# -*- coding: utf-8 -*-
"""
Entrena un adaptador LoRA sobre FLAN-T5 usando JSONL con campos:
{"instruction": "...", "input": "...", "output": "..."}

Uso (PowerShell, desde la carpeta del proyecto):
  python train_lora_flan_t5.py `
    --model_name google/flan-t5-small `
    --train_files .\data\matematicas_quinto.jsonl .\data\tecnologia_robotica_1ro_bgu.jsonl `
    --out_dir .\models\flan_t5_edu_lora `
    --epochs 2 --bsz 4 --grad_accum 4 --max_len 256 --lr 5e-4
"""
import argparse, os, sys
from typing import Dict, List
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
)
from peft import LoraConfig, TaskType, get_peft_model

def build_prompt(example: Dict) -> Dict:
    src = f"Instrucción: {example['instruction']}\nEntrada: {example['input']}\nSalida:"
    tgt = example["output"]
    return {"source": src, "target": tgt}

def _check_paths(files: List[str]) -> None:
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print("❌ No se encontraron estos archivos:", *missing, sep="\n - ")
        print("💡 Verifica rutas relativas. Si estás en otra carpeta, usa rutas absolutas.")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google/flan-t5-small",
                    help="Usa flan-t5-small en CPU; flan-t5-base si tienes más RAM")
    ap.add_argument("--train_files", nargs="+", required=True, help="Uno o más .jsonl")
    ap.add_argument("--out_dir", default="models/flan_t5_edu_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    _check_paths(args.train_files)
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer y modelo base
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    base.config.use_cache = False  # recomendado para entrenamiento

    # LoRA económico en VRAM/RAM
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(base, peft_cfg)

    # Cargar datasets y concatenar
    datasets = []
    for f in args.train_files:
        ds = load_dataset("json", data_files={"train": f})["train"]
        ds = ds.map(build_prompt, remove_columns=ds.column_names)
        datasets.append(ds)
    train_ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    def tokenize(ex):
        x = tok(ex["source"], truncation=True, max_length=args.max_len)
        y = tok(ex["target"], truncation=True, max_length=args.max_len)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(tokenize, batched=False)
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    # Entrenamiento orientado a CPU
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,  # Windows friendly
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        data_collator=collator,
        tokenizer=tok,  # (FutureWarning aceptable en 4.45)
    )
    trainer.train()

    trainer.model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)
    print(f"✅ Adapter LoRA guardado en: {args.out_dir}")

if __name__ == "__main__":
    main()
