import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def resolve_base_dir(base:str)->str:
    p = Path(base)
    if (p/"tokenizer.json").exists():  # 스냅샷 바로 지정된 경우
        return str(p)
    snaps = p/"snapshots"
    if snaps.is_dir():
        # 가장 최근 스냅샷 중 tokenizer.json 있는 것 선택
        cands = sorted(snaps.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for c in cands:
            if (c/"tokenizer.json").exists():
                return str(c)
    raise FileNotFoundError(f"tokenizer.json not found under {base}")

def build_prompt(sample):
    # ✅ 키워드 한 줄, 쉼표만. 불필요한 문장/판례 금지.
    q = sample.get("question") or sample.get("prompt") or sample.get("input") or sample.get("query") or ""
    inst = ("지시: 질문에 대해 정답 키워드만 쉼표(,)로 나열하라. "
            "불필요한 문장·예시·판례번호·괄호·따옴표·끝마침표 금지. 한 줄로만 출력.")
    return f"{inst}\n질문: {q}\n정답:"

def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    # ✅ SKT 기본값
    ap.add_argument("--base", default="/workspace/models/models--skt--A.X-3.1-Light")
    ap.add_argument("--adapter", default="/workspace/outputs/qlora-sft-civil-law-skt/checkpoint-13500")
    ap.add_argument("--input_jsonl", default="/workspace/data/gold_val.jsonl")
    ap.add_argument("--out_jsonl", default="/workspace/outputs/qlora-sft-civil-law-skt/predictions.jsonl")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--dtype", choices=["auto","bf16","fp16"], default="bf16")
    ap.add_argument("--no_strip_prompt_echo", action="store_true")
    ap.add_argument("--tokenizer_base", default=None, help="토크나이저만 별도 경로에서 로드(옵션)")
    args = ap.parse_args()

    # dtype
    torch_dtype = "auto"
    if args.dtype=="bf16": torch_dtype=torch.bfloat16
    elif args.dtype=="fp16": torch_dtype=torch.float16

    base_resolved = resolve_base_dir(args.base)
    tok_base = args.tokenizer_base or base_resolved

    # 먼저 legacy=False 시도 → 안 되면 기본
    try:
        tok = AutoTokenizer.from_pretrained(tok_base, use_fast=True, legacy=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(tok_base, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(base_resolved, torch_dtype=torch_dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    outp = Path(args.out_jsonl); outp.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with Path(outp).open("w", encoding="utf-8") as fout:
        for ex in load_jsonl(args.input_jsonl):
            _id = ex.get(args.id_field, n)
            prompt = build_prompt(ex)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature>0.0),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(gen[0], skip_special_tokens=True)
            if not args.no_strip_prompt_echo and text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            # 후처리: 줄바꿈/마침표 제거, 양끝 공백 제거
            text = text.replace("\n"," ").strip().rstrip(" .")
            fout.write(json.dumps({"id": _id, "prediction": text}, ensure_ascii=False)+"\n")
            n+=1
    print(f"Wrote predictions: {outp} (N={n})")

if __name__ == "__main__":
    main()
