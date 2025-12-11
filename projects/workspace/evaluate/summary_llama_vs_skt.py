import json, pandas as pd
from pathlib import Path

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def r(x):
    try: return None if x is None else round(float(x), 6)
    except: return None

llama_dir = Path("/workspace/evaluate")
skt_dir   = Path("/workspace/evaluate/Eval_skt")

paths = {
    "LLaMA": {
        "Exact Match": llama_dir/"exact_match_summary.json",
        "Token-F1":    llama_dir/"token_f1_summary.json",
        "BERTScore":   llama_dir/"bertscore_summary.json",
        "ROUGE-L":     llama_dir/"rouge_l_summary.json",
    },
    "SKT": {
        "Exact Match": skt_dir/"exact_match_summary.json",
        "Token-F1":    skt_dir/"token_f1_summary.json",
        "BERTScore":   skt_dir/"bertscore_summary.json",
        "ROUGE-L":     skt_dir/"rouge_l_summary.json",
    },
}

rows = []
for model, d in paths.items():
    em = load_json(d["Exact Match"]).get("exact_match")
    tf = load_json(d["Token-F1"])
    bs = load_json(d["BERTScore"]).get("f1")
    rl = load_json(d["ROUGE-L"]).get("f1")

    micro = (tf.get("micro") or {}).get("f1")
    macro = (tf.get("macro") or {}).get("f1")

    rows.append({
        "Model": model,
        "Exact Match": r(em),
        "Token-F1 (micro)": r(micro),
        "Token-F1 (macro)": r(macro),
        "BERTScore F1": r(bs),
        "ROUGE-L F1": r(rl),
    })

df = pd.DataFrame(rows, columns=["Model","Exact Match","Token-F1 (micro)","Token-F1 (macro)","BERTScore F1","ROUGE-L F1"])
out_csv = "/workspace/evaluate/summary_llama_vs_skt.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print(f"\nSaved: {out_csv}")
