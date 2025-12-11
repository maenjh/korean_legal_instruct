import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ws = re.compile(r"\s+")
# 최소 기호(.,%/)는 보존
_punct = re.compile(r"[^\w가-힣一-龥ぁ-んァ-ンー・％%.,:;!?(){}\[\]\-–—/\"'“”‘’·`´]+")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
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

def tokenize(s: str) -> List[str]:
    # 공백 기준 토큰화(한국어 키워드형 평가에 안정적)
    return s.split() if s else []

def prf_counts(pred_tokens: List[str], gold_tokens: List[str]) -> Tuple[int,int,int]:
    pc = Counter(pred_tokens); gc = Counter(gold_tokens)
    overlap = pc & gc
    overlap_n = sum(overlap.values())
    return overlap_n, sum(pc.values()), sum(gc.values())

def f1_from(overlap, p_total, g_total) -> Tuple[float,float,float]:
    precision = overlap / p_total if p_total else 0.0
    recall    = overlap / g_total if g_total else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    return precision, recall, f1

def align_examples(pred_rows, gold_rows, id_field):
    if all(id_field in r for r in pred_rows) and all(id_field in r for r in gold_rows):
        gmap = {r[id_field]: r for r in gold_rows}
        return [(pr, gmap[pr[id_field]]) for pr in pred_rows if pr[id_field] in gmap]
    n = min(len(pred_rows), len(gold_rows))
    return [(pred_rows[i], gold_rows[i]) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="Token-level F1 evaluator")
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
    args = ap.parse_args()

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)
    pairs = align_examples(pred_rows, gold_rows, args.id_field)
    if not pairs:
        print("No aligned pairs. Check fields/paths.", file=sys.stderr); sys.exit(2)

    # per-sample & aggregation
    per = []
    micro_overlap = micro_pred = micro_gold = 0
    f1_list, p_list, r_list = [], [], []

    for i,(pr,gr) in enumerate(pairs):
        pid = pr.get(args.id_field, i)
        p_raw = pr.get(args.pred_field, "")
        g_raw = gr.get(args.gold_field, "")

        p_norm = normalize_text(p_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)
        g_norm = normalize_text(g_raw, lower=not args.no_lower, strip_punct=not args.keep_punct, collapse_ws=not args.keep_ws)

        p_toks = tokenize(p_norm); g_toks = tokenize(g_norm)
        ov, p_tot, g_tot = prf_counts(p_toks, g_toks)
        P, R, F1 = f1_from(ov, p_tot, g_tot)

        micro_overlap += ov; micro_pred += p_tot; micro_gold += g_tot
        p_list.append(P); r_list.append(R); f1_list.append(F1)

        per.append({
            "id": pid,
            "precision": round(P,6),
            "recall": round(R,6),
            "f1": round(F1,6),
            "overlap_tokens": ov,
            "pred_tokens": p_tot,
            "gold_tokens": g_tot,
            "pred_text": p_raw,
            "gold_text": g_raw,
        })

    microP, microR, microF1 = f1_from(micro_overlap, micro_pred, micro_gold)
    macroP = sum(p_list)/len(p_list) if p_list else 0.0
    macroR = sum(r_list)/len(r_list) if r_list else 0.0
    macroF1= sum(f1_list)/len(f1_list) if f1_list else 0.0

    summary = {
        "num_examples": len(per),
        "micro": {"precision": microP, "recall": microR, "f1": microF1},
        "macro": {"precision": macroP, "recall": macroR, "f1": macroF1},
        "options": {
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
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Token-F1] micro_f1={microF1:.6f} macro_f1={macroF1:.6f} (N={len(per)})")
    print(f"Details: {args.out_csv}")
    print(f"Summary: {args.out_json}")

if __name__ == "__main__":
    main()
