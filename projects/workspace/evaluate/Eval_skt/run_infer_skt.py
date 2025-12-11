import os, json, argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    ap = argparse.ArgumentParser(description="Generate predictions (answer-only) with base+LoRA")
    ap.add_argument("--base", required=True, help="Base model path (e.g., /workspace/models/.../snapshots/...)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (e.g., /workspace/outputs/.../checkpoint-XXXXX)")
    ap.add_argument("--input_jsonl", required=True, help="Gold JSONL path")
    ap.add_argument("--out_jsonl", required=True, help="Where to write predictions (id,prediction)")
    ap.add_argument("--id_field", default="id", help="ID field name in gold (default: id)")
    ap.add_argument("--input_field", default="question", help="Prompt/question field name in gold (default: question)")
    ap.add_argument("--prefix", default="", help="Optional string prepended before question")
    ap.add_argument("--suffix", default="\n### 답변:", help="String appended after question to cue answer start")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--no_sample", action="store_true", help="Use greedy decoding (default)")
    return ap.parse_args()

def _dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]

def load_pairs(path: str, id_field: str, input_field: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            if id_field not in o or input_field not in o:
                # 허용: taskinfo.input 구조를 자동 탐지
                if input_field == "question" and "taskinfo" in o and "input" in o["taskinfo"]:
                    q = o["taskinfo"]["input"]
                    rid = o.get(id_field, len(rows))
                else:
                    continue
            else:
                rid, q = o[id_field], o[input_field]
            rows.append({"id": rid, "question": str(q)})
    return rows

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] tokenizer: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    print(f"[load] base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[peft] attach adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    print(f"[data] read: {args.input_jsonl}")
    pairs = load_pairs(args.input_jsonl, args.id_field, args.input_field)
    print(f"[data] N={len(pairs)}")

    Path(os.path.dirname(args.out_jsonl)).mkdir(parents=True, exist_ok=True)

    # 배치 인퍼런스
    B = max(1, args.batch_size)
    do_sample = not args.no_sample  # 기본: sampling off가 더 재현성 좋음 → 아래서 강제 False
    do_sample = False

    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for i in range(0, len(pairs), B):
            batch = pairs[i:i+B]
            prompts = [f"{args.prefix}{ex['question']}{args.suffix}" for ex in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                    do_sample=do_sample,
                )

            # 핵심: 입력 길이 이후(new tokens)만 디코딩 → 정답만!
            new_tokens = gen_ids[:, input_len:]
            outs = tok.batch_decode(new_tokens, skip_special_tokens=True)

            for ex, ans in zip(batch, outs):
                ans = ans.strip()
                w.write(json.dumps({"id": ex["id"], "prediction": ans}, ensure_ascii=False) + "\n")

    print(f"[done] Wrote predictions: {args.out_jsonl}")

if __name__ == "__main__":
    main()
