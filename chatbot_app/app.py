from __future__ import annotations
from flask import Flask, render_template, request, jsonify
import re, time
from typing import Dict, Any, Tuple
from ai_model import get_generator

edu_gen = get_generator()  # Puede ser None si aún no has entrenado/copiado el modelo
app = Flask(__name__)

# ---------- Memoria simple en RAM (demo)
profiles: Dict[str, Dict[str, Any]] = {}

def get_or_create_profile(student_id: str) -> Dict[str, Any]:
    if student_id not in profiles:
        profiles[student_id] = {
            "created_at": time.time(),
            "level": "basico",
            "pref_explicacion": "corta",
            "materias": {
                "matematicas": {"aciertos": 0, "errores": 0, "ultimo_tema": None},
                "robotica": {"aciertos": 0, "errores": 0, "ultimo_tema": None},
            },
        }
    return profiles[student_id]

# ---------- Utilidades de Matemáticas
_ALLOWED_MATH = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def safe_eval_math(expr: str) -> Tuple[bool, str]:
    expr = expr.strip()
    if not expr:
        return False, "Ingresa una expresión, por ejemplo: 12 + 7 * 3."
    if not _ALLOWED_MATH.match(expr):
        return False, "Solo evalúo números y operaciones básicas (+, -, *, /) con paréntesis."
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return True, f"El resultado de **{expr}** es **{result}**."
    except ZeroDivisionError:
        return False, "No se puede dividir para cero."
    except Exception:
        return False, "No pude evaluar esa expresión. Revisa la sintaxis (ej: (3+5)*2 )."

def next_math_exercise(level: str) -> str:
    if level == "basico":
        return "Resuelve: 36 + 27 - 8"
    if level == "intermedio":
        return "Resuelve: (45 - 12) * 3"
    return "Resuelve: (120 / (6 + 4)) * (15 - 3)"

# ---------- Robótica (mini FAQ)
ROBOTICS_FAQ = {
    "sensor ultrasonico": "El HC-SR04 mide distancia por ultrasonido: d ≈ (tiempo_eco * v_sonido) / 2.",
    "arduino pwm": "PWM en Arduino: analogWrite(pin, valor) de 0 a 255.",
    "puente h": "Permite invertir el sentido de un motor DC con dos entradas lógicas.",
    "servo": "Pulso ~1ms ≈ 0°, ~1.5ms ≈ 90°, ~2ms ≈ 180° a ~50Hz.",
    "esp32 vs arduino": "ESP32 trae WiFi/BLE y más potencia; Arduino UNO es ideal para iniciar.",
}

def answer_robotics(query: str) -> str | None:
    q = query.lower()
    for key, ans in ROBOTICS_FAQ.items():
        if key in q:
            return ans
    return None

# ---------- Motor de respuesta
def personalize_text(text: str, profile: Dict[str, Any]) -> str:
    if profile.get("pref_explicacion") == "corta":
        return text
    return text + "\n\nTip: si quieres, puedo darte un ejemplo adicional o un ejercicio guiado."

def adapt_difficulty(profile: Dict[str, Any], materia: str, correcto: bool) -> None:
    stats = profile["materias"][materia]
    stats["aciertos" if correcto else "errores"] += 1
    a, e = stats["aciertos"], stats["errores"]
    if a >= 3 and profile["level"] == "basico":
        profile["level"] = "intermedio"
    if a >= 6 and profile["level"] == "intermedio":
        profile["level"] = "avanzado"
    if e >= 3 and profile["level"] == "avanzado":
        profile["level"] = "intermedio"
    if e >= 5 and profile["level"] == "intermedio":
        profile["level"] = "basico"

def tutor_matematicas(message: str, profile: Dict[str, Any]) -> str:
    ok, resp = safe_eval_math(message)
    if ok:
        adapt_difficulty(profile, "matematicas", True)
        extra = f"\n¿Quieres otro ejercicio {profile['level']}? {next_math_exercise(profile['level'])}"
        return personalize_text(resp + extra, profile)
    adapt_difficulty(profile, "matematicas", False)
    hint = "Pista: usa solo números, +, -, *, / y paréntesis. Ej: (3+5)*2"
    exercise = next_math_exercise(profile["level"])
    return personalize_text(f"{resp}\n\n{hint}\n\nPrueba este: {exercise}", profile)

def tutor_robotica(message: str, profile: Dict[str, Any]) -> str:
    ans = answer_robotics(message)
    if ans:
        adapt_difficulty(profile, "robotica", True)
        return personalize_text(ans, profile)
    adapt_difficulty(profile, "robotica", False)
    keys = ", ".join(ROBOTICS_FAQ.keys())
    return personalize_text(
        "No encontré una respuesta directa. ¿Puedes darme más contexto?\n"
        f"Temas que puedo explicar: {keys}.", profile
    )

def route_message(subject: str, message: str, profile: Dict[str, Any]) -> str:
    subject = (subject or "").strip().lower()
    if subject in ("matematicas", "matemáticas", "mate", "math"):
        profile["materias"]["matematicas"]["ultimo_tema"] = "general"
        return tutor_matematicas(message, profile)
    if subject in ("robotica", "robótica", "robotics"):
        profile["materias"]["robotica"]["ultimo_tema"] = "general"
        return tutor_robotica(message, profile)
    if _ALLOWED_MATH.match(message.strip()):
        return tutor_matematicas(message, profile)
    if any(k in message.lower() for k in ROBOTICS_FAQ.keys()):
        return tutor_robotica(message, profile)
    return personalize_text(
        "Puedo ayudarte en **Matemáticas** o **Robótica**.\n"
        "Ejemplos:\n- Matemáticas: (12+8)/5\n- Robótica: ¿Cómo usar un sensor ultrasónico?",
        profile
    )

# ---------- Endpoints
@app.route("/api/generate", methods=["POST"])
def api_generate():
    if edu_gen is None:
        return jsonify({"ok": False, "error": "Modelo no cargado. Entrena o verifica 'models/flan_t5_edu_lora'."}), 503
    data = request.get_json(silent=True) or {}
    instruction = (data.get("instruction") or "Eres un tutor en español. Explica paso a paso.").strip()
    inp = (data.get("input") or "").strip()
    if not inp:
        return jsonify({"ok": False, "error": "Falta 'input' en el cuerpo JSON."}), 400
    text = edu_gen.generate(instruction, inp)
    return jsonify({"ok": True, "text": text})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "chatbot-educativo", "time": time.time()})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    student_id = (data.get("student_id") or "anonimo").strip()
    subject = (data.get("subject") or "").strip().lower()
    pref = (data.get("pref_explicacion") or "").strip().lower()

    if not message:
        return jsonify({"ok": False, "error": "Falta 'message' en el cuerpo JSON."}), 400

    profile = get_or_create_profile(student_id)
    if pref in ("corta", "detallada"):
        profile["pref_explicacion"] = pref

    reply = route_message(subject, message, profile)
    return jsonify({
        "ok": True,
        "student_id": student_id,
        "level": profile["level"],
        "subject": subject or "auto",
        "reply": reply
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
