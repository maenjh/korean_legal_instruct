# /workspace/evaluate/evaluate_token_f1.py
"""
Token-F1 evaluator for JSONL prediction/gold files.

- Aligns by id if present; otherwise evaluates in order.
- Outputs: per-sample CSV + overall summary JSON.
- Default fields: id, prediction (pred), answer (gold)
Usage:
  python3 evaluate_token_f1.py \
    --pred /workspace/outputs/qlora-sft-civil-law-llama/predictions.jsonl \
    --gold /workspace/data/gold_val.jsonl \
    --out_csv /workspace/evaluate/token_f1_details.csv \
    --out_json /workspace/evaluate/token_f1_summary.json
"""

import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w가-힣一-龥ぁ-んァ-ンー・％％％%.,:;!?(){}\[\]「」『』《》〈〉“”\"'`´·\-–—/]+")

def normalize_text(s: str, lower: bool=True, strip_punct: bool=True) -> str:
    if s is None:
        return ""
    s = s.strip()
    if lower:
        s = s.lower()
    # normalize whitespace-like punctuation to space
    s = s.replace("\u00A0", " ").replace("\u200b", "")
    # optional: remove exotic control chars
    s = "".join(ch for ch in s if ch.isprintable())
    if strip_punct:
        # keep basic punctuation that may act as tokens (.,%/‐) by spacing them
        s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    # Split on whitespace; treat ., % and / as separate tokens if spaced
    # Example: "제3조(1)" -> "제3조", "1"
    s = re.sub(r"([.,/%])", r" \1 ", s)
    s = _ws.sub(" ", s).strip()
    return s.split() if s else []

def prf1(pred_tokens: List[str], gold_tokens: List[str]) -> Tuple[float,float,float,int,int,int]:
    pc = Counter(pred_tokens)
    gc = Counter(gold_tokens)
    overlap = sum((pc & gc).values())
    p = overlap / max(1, sum(pc.values()))
    r = overlap / max(1, sum(gc.values()))
    f1 = 0.0 if p+r==0 else 2*p*r/(p+r)
    return p, r, f1, overlap, sum(pc.values()), sum(gc.values())

def align_examples(pred_rows, gold_rows, id_field, pred_field, gold_field):
    # If all rows have id, align by id; else align by index with min length
    if all(id_field in r for r in pred_rows) and all(id_field in r for r in gold_rows):
        gmap = {r[id_field]: r for r in gold_rows}
        aligned = []
        for pr in pred_rows:
            _id = pr[id_field]
            if _id in gmap:
                aligned.append((pr, gmap[_id]))
        return aligned
    else:
        n = min(len(pred_rows), len(gold_rows))
        return [(pred_rows[i], gold_rows[i]) for i in range(n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--pred_field", default="prediction")
    ap.add_argument("--gold_field", default="answer")
    ap.add_argument("--no_lower", action="store_true", help="disable lowercasing")
    ap.add_argument("--keep_punct", action="store_true", help="keep punctuation (no strip)")
    args = ap.parse_args()

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)
    pairs = align_examples(pred_rows, gold_rows, args.id_field, args.pred_field, args.gold_field)

    if not pairs:
        print("No aligned pairs found. Check id_field/pred_field/gold_field.", file=sys.stderr)
        sys.exit(2)

    per_rows = []
    micro_tp = micro_pred = micro_gold = 0
    macro_f1s, macro_ps, macro_rs = [], [], []

    for pr, gr in pairs:
        pid = pr.get(args.id_field, len(per_rows))
        pred_text = pr.get(args.pred_field, "")
        gold_text = gr.get(args.gold_field, "")
        pn = normalize_text(pred_text, lower=not args.no_lower, strip_punct=not args.keep_punct)
        gn = normalize_text(gold_text, lower=not args.no_lower, strip_punct=not args.keep_punct)
        ptoks = tokenize(pn)
        gtoks = tokenize(gn)
        p, r, f1, tp, pcount, gcount = prf1(ptoks, gtoks)

        per_rows.append({
            "id": pid,
            "precision": round(p,6),
            "recall": round(r,6),
            "f1": round(f1,6),
            "overlap_tokens": tp,
            "pred_tokens": pcount,
            "gold_tokens": gcount,
            "pred_text": pred_text,
            "gold_text": gold_text,
        })
        micro_tp += tp
        micro_pred += pcount
        micro_gold += gcount
        macro_ps.append(p); macro_rs.append(r); macro_f1s.append(f1)

    micro_p = micro_tp / max(1, micro_pred)
    micro_r = micro_tp / max(1, micro_gold)
    micro_f1 = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)

    def _avg(xs): 
        return float(sum(xs)/max(1,len(xs)))

    summary = {
        "num_examples": len(per_rows),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": _avg(macro_ps), "recall": _avg(macro_rs), "f1": _avg(macro_f1s)},
        "options": {
            "lower": not args.no_lower,
            "strip_punct": not args.keep_punct,
            "id_field": args.id_field,
            "pred_field": args.pred_field,
            "gold_field": args.gold_field,
        }
    }

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    # write JSON
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Token-F1] micro_f1={summary['micro']['f1']:.6f} "
          f"macro_f1={summary['macro']['f1']:.6f} "
          f"(N={summary['num_examples']})")
    print(f"Details: {args.out_csv}")
    print(f"Summary: {args.out_json}")

if __name__ == "__main__":
    main()
