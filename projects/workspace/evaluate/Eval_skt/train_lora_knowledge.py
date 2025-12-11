import os, json, random
from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a"
DATA_PATH  = "/workspace/evaluate/Eval_skt/knowledge/gold_val_std.jsonl"
OUTPUT_DIR = "/workspace/outputs/qlora-sft-knowledge-law-skt"

MAX_LEN = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 16
LR = 2e-4
NUM_EPOCHS = 1
SAVE_STEPS = 1000
SEED = 42

def set_seed(seed):
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class QADataset(Dataset):
    def __init__(self, rows: List[Dict], tok: AutoTokenizer):
        self.rows = rows
        self.tok = tok

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        q = r["question"].strip()
        a = r["answer"].strip()
        prompt = f"{q}\n### 답변:"
        in_ids  = self.tok(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LEN).input_ids
        ans_ids = self.tok(a, add_special_tokens=False, truncation=True, max_length=MAX_LEN - len(in_ids)).input_ids
        ids = in_ids + ans_ids + [self.tok.eos_token_id]
        if len(ids) > MAX_LEN: ids = ids[:MAX_LEN]
        labels = [-100]*len(in_ids) + ans_ids + [self.tok.eos_token_id]
        labels = labels[:len(ids)]
        attn = [1]*len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

def load_rows(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line)
            q=o.get("question","").strip(); a=o.get("answer","").strip()
            if q and a: rows.append({"question":q, "answer":a})
    return rows

# --- 간단 커스텀 collator (pad + tensor 스택만 수행) ---
@dataclass
class SimpleCollator:
    pad_id: int
    def __call__(self, features):
        # pad to max len in batch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            att = f["attention_mask"]
            lab = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(torch.nn.functional.pad(ids, (0,pad_len), value=self.pad_id))
            attention_mask.append(torch.nn.functional.pad(att, (0,pad_len), value=0))
            labels.append(torch.nn.functional.pad(lab, (0,pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

def main():
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    rows = load_rows(DATA_PATH)
    random.shuffle(rows)
    n_total = len(rows)
    n_val = min(500, max(1, n_total//20))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_ds = QADataset(train_rows, tok)
    val_ds   = QADataset(val_rows, tok)

    collator = SimpleCollator(pad_id=tok.pad_token_id)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=50,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,   # ← 커스텀 collator 사용
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint-final"))
    print(f"[DONE] LoRA saved in {OUTPUT_DIR}")
    try:
        print("Checkpoints:", [p for p in os.listdir(OUTPUT_DIR) if p.startswith("checkpoint-")])
    except Exception:
        pass
    print("Final:", os.path.join(OUTPUT_DIR, "checkpoint-final"))

if __name__ == "__main__":
    main()
