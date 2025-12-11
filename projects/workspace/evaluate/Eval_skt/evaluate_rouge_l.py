import argparse, csv, json, re, sys
from pathlib import Path
from typing import List, Dict, Any

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w가-힣一-龥ぁ-んァ-ンー・％%.,:;!?(){}\[\]「」『』《》〈〉“”\"'`´·\-–—/]+")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
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
    if all(id_field in r for r in pred_rows) and all(id_field in r for r in gold_rows):
        gmap = {r[id_field]: r for r in gold_rows}
        return [(pr, gmap[pr[id_field]]) for pr in pred_rows if pr[id_field] in gmap]
    n = min(len(pred_rows), len(gold_rows))
    return [(pred_rows[i], gold_rows[i]) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="ROUGE-L evaluator")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--pred_field", default="prediction")
    ap.add_argument("--gold_field", default="answer")
    ap.add_argument("--no_lower", action="store_true")
    ap.add_argument("--keep_punct", action="store_true")
    ap.add_argument("--keep_ws", action="store_true")
    ap.add_argument("--char_level", action="store_true", help="문자 단위 토크나이저(한국어 권장)")
    args = ap.parse_args()

    try:
        from rouge_score import rouge_scorer, tokenizers
    except Exception:
        print("rouge-score 패키지가 필요합니다. 설치: pip install -U rouge-score", file=sys.stderr)
        sys.exit(2)

    # rouge-score 버전 호환: CharacterTokenizer 없을 수 있어 직접 구현
    class CharTokenizer:
        def tokenize(self, text: str): return list(text)

    tokenizer = CharTokenizer() if args.char_level else tokenizers.DefaultTokenizer()

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)
    pairs = align_examples(pred_rows, gold_rows, args.id_field)
    if not pairs:
        print("No aligned pairs found. Check id_field and inputs.", file=sys.stderr); sys.exit(2)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=tokenizer)

    per_rows, P_list, R_list, F1_list = [], [], [], []
    for i,(pr,gr) in enumerate(pairs):
        pid = pr.get(args.id_field, i)
        p_raw = str(pr.get(args.pred_field, "") or "")
        g_raw = str(gr.get(args.gold_field, "") or "")

        p = normalize_text(p_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)
        g = normalize_text(g_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)

        s = scorer.score(g, p)['rougeL']  # reference=g, prediction=p
        P_list.append(s.precision); R_list.append(s.recall); F1_list.append(s.fmeasure)
        per_rows.append({
            "id": pid,
            "precision": round(float(s.precision), 6),
            "recall": round(float(s.recall), 6),
            "f1": round(float(s.fmeasure), 6),
            "pred_text": p_raw,
            "gold_text": g_raw,
        })

    def _avg(xs): return float(sum(xs)/max(1,len(xs)))
    summary = {
        "num_examples": len(per_rows),
        "precision": _avg(P_list),
        "recall": _avg(R_list),
        "f1": _avg(F1_list),
        "options": {
            "char_level": args.char_level,
            "lower": not args.no_lower,
            "strip_punct": not args.keep_punct,
            "collapse_ws": not args.keep_ws,
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

    print(f"[ROUGE-L] F1={summary['f1']:.6f} (P={summary['precision']:.6f}, R={summary['recall']:.6f})")
    print(f"Details: {args.out_csv}")
    print(f"Summary: {args.out_json}")

if __name__ == "__main__":
    main()
