# -*- coding: utf-8 -*-
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

class EduGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        lora_dir: str = "models/flan_t5_edu_lora",
        max_new_tokens: int = 196,
        temperature: float = 0.7,
        num_beams: int = 1,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_beams = num_beams

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        base.config.use_cache = True  # inferencia
        self.model = PeftModel.from_pretrained(base, lora_dir)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def generate(self, instruction: str, input_text: str) -> str:
        prompt = f"Instrucción: {instruction}\nEntrada: {input_text}\nSalida:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.num_beams == 1),
            temperature=self.temperature,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# Carga perezosa (opcional)
_generator: Optional[EduGenerator] = None

def get_generator() -> Optional[EduGenerator]:
    global _generator
    if _generator is None:
        try:
            _generator = EduGenerator(
                model_name="google/flan-t5-small",
                lora_dir="models/flan_t5_edu_lora",
                max_new_tokens=160,
            )
        except Exception as e:
            print("⚠️ No se pudo cargar el modelo entrenado:", e)
            _generator = None
    return _generator
