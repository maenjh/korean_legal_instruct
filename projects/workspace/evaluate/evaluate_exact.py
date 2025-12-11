import argparse, csv, json, re, sys
from pathlib import Path
from typing import Any, Dict, List

# 공백/문장부호 정규화용
_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w가-힣一-龥ぁ-んァ-ンー・％%.,:;!?(){}\[\]\-–—/\"'“”‘’·`´]+")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_text(s: str, lower=True, strip_punct=True, collapse_ws=True) -> str:
    if s is None: return ""
    s = s.strip().replace("\u00A0"," ").replace("\u200b","")
    s = "".join(ch for ch in s if ch.isprintable())
    if lower: s = s.lower()
    if strip_punct: s = _punct.sub(" ", s)
    if collapse_ws: s = _ws.sub(" ", s)
    return s.strip()

def align_examples(pred_rows, gold_rows, id_field):
    # id가 있으면 id 기준 정렬, 없으면 인덱스 정렬
    if all(id_field in r for r in pred_rows) and all(id_field in r for r in gold_rows):
        gmap = {r[id_field]: r for r in gold_rows}
        return [(pr, gmap[pr[id_field]]) for pr in pred_rows if pr[id_field] in gmap]
    n = min(len(pred_rows), len(gold_rows))
    return [(pred_rows[i], gold_rows[i]) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="Exact Match evaluator (0/1)")
    ap.add_argument("--pred", required=True, help="predictions.jsonl (id, prediction)")
    ap.add_argument("--gold", required=True, help="gold.jsonl (id, answer)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--pred_field", default="prediction")
    ap.add_argument("--gold_field", default="answer")
    # 정규화 토글
    ap.add_argument("--no_lower", action="store_true", help="소문자화 비활성화")
    ap.add_argument("--keep_punct", action="store_true", help="문장부호 제거 안 함")
    ap.add_argument("--keep_ws", action="store_true", help="연속 공백 축약 안 함")
    args = ap.parse_args()

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)
    pairs = align_examples(pred_rows, gold_rows, args.id_field)
    if not pairs:
        print("No aligned pairs. Check fields/paths.", file=sys.stderr); sys.exit(2)

    per, correct = [], 0
    for i, (pr, gr) in enumerate(pairs):
        pid = pr.get(args.id_field, i)
        p_raw = pr.get(args.pred_field, "")
        g_raw = gr.get(args.gold_field, "")
        p = normalize_text(p_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)
        g = normalize_text(g_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)
        em = int(p == g)
        correct += em
        per.append({
            "id": pid,
            "exact_match": em,
            "pred_norm": p,
            "gold_norm": g,
            "pred_text": p_raw,
            "gold_text": g_raw,
        })

    acc = correct / max(1, len(per))
    summary = {
        "num_examples": len(per),
        "exact_match": acc,
        "correct": correct,
        "options": {
            "lower": not args.no_lower,
            "strip_punct": not args.keep_punct,
            "collapse_ws": not args.keep_ws,
            "id_field": args.id_field,
            "pred_field": args.pred_field,
            "gold_field": args.gold_field,
        },
    }

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ExactMatch] accuracy={acc:.6f} (correct={correct}/{len(per)})")
    print(f"Details: {args.out_csv}")
    print(f"Summary: {args.out_json}")

if __name__ == "__main__":
    main()
