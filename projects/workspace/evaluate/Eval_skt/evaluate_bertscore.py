import argparse, csv, json, sys
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def align_examples(pred_rows, gold_rows, id_field):
    if all(id_field in r for r in pred_rows) and all(id_field in r for r in gold_rows):
        gmap = {r[id_field]: r for r in gold_rows}
        return [(pr, gmap[pr[id_field]]) for pr in pred_rows if pr[id_field] in gmap]
    n = min(len(pred_rows), len(gold_rows))
    return [(pred_rows[i], gold_rows[i]) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="BERTScore evaluator")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--pred_field", default="prediction")
    ap.add_argument("--gold_field", default="answer")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--rescale", action="store_true", help="rescale_with_baseline=True")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (auto if None)")
    args = ap.parse_args()

    try:
        from bert_score import score as bertscore
    except Exception:
        print("bert-score 패키지가 필요합니다. 설치: pip install -U bert-score", file=sys.stderr)
        sys.exit(2)

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)
    pairs = align_examples(pred_rows, gold_rows, args.id_field)
    if not pairs:
        print("No aligned pairs found. Check id_field and inputs.", file=sys.stderr); sys.exit(2)

    ids, cands, refs, raw_preds, raw_golds = [], [], [], [], []
    for i,(pr,gr) in enumerate(pairs):
        pid = pr.get(args.id_field, i)
        p = str(pr.get(args.pred_field, "")).strip()
        g = str(gr.get(args.gold_field, "")).strip()
        ids.append(pid); cands.append(p); refs.append(g); raw_preds.append(p); raw_golds.append(g)

    if args.device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    P, R, F1 = bertscore(
        cands, refs,
        model_type=args.model,
        lang=args.lang,
        rescale_with_baseline=args.rescale,
        device=device,
        batch_size=args.batch_size,
        verbose=True,
    )

    per_rows = []
    for i, pid in enumerate(ids):
        per_rows.append({
            "id": pid,
            "P": round(float(P[i]), 6),
            "R": round(float(R[i]), 6),
            "F1": round(float(F1[i]), 6),
            "pred_text": raw_preds[i],
            "gold_text": raw_golds[i],
        })

    def _avg(t): 
        return float(t.mean().item()) if hasattr(t, "mean") else float(sum(t)/len(t))
    summary = {
        "num_examples": len(per_rows),
        "precision": _avg(P),
        "recall": _avg(R),
        "f1": _avg(F1),
        "options": {
            "model": args.model,
            "lang": args.lang,
            "rescale_with_baseline": args.rescale,
            "batch_size": args.batch_size,
            "device": device,
            "id_field": args.id_field,
            "pred_field": args.pred_field,
            "gold_field": args.gold_field,
        }
    }

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader(); w.writerows(per_rows)

    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[BERTScore] F1={summary['f1']:.6f}  (P={summary['precision']:.6f}, R={summary['recall']:.6f})")
    print(f"Details: {args.out_csv}")
    print(f"Summary: {args.out_json}")

if __name__ == "__main__":
    main()
