#!/usr/bin/env bash
set -euo pipefail

# 고정: 베이스 모델
BASE="/workspace/models/models--skt--A.X-3.1-Light"

# 도메인 라벨(표시용)
declare -A DOM_LABEL=(
  ["civil"]="민사"
  ["knowledge"]="지식"
  ["administrative"]="행정"
  ["criminal"]="형사"
)

# GOLD 경로 (도메인별 평가셋)
declare -A GOLD=(
  ["civil"]="/workspace/evaluate/Eval_skt/civil/gold_val.jsonl"
  ["knowledge"]="/workspace/evaluate/Eval_skt/knowledge/gold_val.jsonl"
  ["administrative"]="/workspace/evaluate/Eval_skt/administrative/gold_val.jsonl"
  ["criminal"]="/workspace/evaluate/Eval_skt/criminal/gold_val.jsonl"
)

# ===== 실제 LoRA 체크포인트 목록 (네가 준 경로로 반영) =====
# 민사
declare -A LORAS_civil=(
  [ckpt9000]="/workspace/outputs/qlora-sft-civil-law-skt/checkpoint-9000"
  [ckpt13500]="/workspace/outputs/qlora-sft-civil-law-skt/checkpoint-13500"
  [ckpt13518]="/workspace/outputs/qlora-sft-civil-law-skt/checkpoint-13518"
)

# 지식(= 지식재산법; intellectual property)
declare -A LORAS_knowledge=(
  [ckpt9000]="/workspace/outputs/qlora-sft-intellectual-property-law-skt/checkpoint-9000"
  [ckpt13500]="/workspace/outputs/qlora-sft-intellectual-property-law-skt/checkpoint-13500"
  [ckpt13596]="/workspace/outputs/qlora-sft-intellectual-property-law-skt/checkpoint-13596"
)

# 행정
declare -A LORAS_administrative=(
  [ckpt9000]="/workspace/outputs/qlora-sft-administrative-law-skt/checkpoint-9000"
  [ckpt13000]="/workspace/outputs/qlora-sft-administrative-law-skt/checkpoint-13000"
  [ckpt13500]="/workspace/outputs/qlora-sft-administrative-law-skt/checkpoint-13500"
)

# 형사
declare -A LORAS_criminal=(
  [ckpt9000]="/workspace/outputs/qlora-sft-criminal-law-skt/checkpoint-9000"
  [ckpt13000]="/workspace/outputs/qlora-sft-criminal-law-skt/checkpoint-13000"
  [ckpt13500]="/workspace/outputs/qlora-sft-criminal-law-skt/checkpoint-13500"
)

# 도메인→어댑터 맵
get_loras() {
  local dom="$1"
  case "$dom" in
    civil)           declare -n M=LORAS_civil ;;
    knowledge)       declare -n M=LORAS_knowledge ;;
    administrative)  declare -n M=LORAS_administrative ;;
    criminal)        declare -n M=LORAS_criminal ;;
    *) return 1 ;;
  esac
  for k in "${!M[@]}"; do
    echo "$k:::${M[$k]}"
  done
}

EVAL_ROOT="/workspace/evaluate/Eval_skt"
mkdir -p "$EVAL_ROOT"

# 필요 패키지
pip show bert-score  >/dev/null 2>&1 || pip install -U bert-score
pip show rouge-score >/dev/null 2>&1 || pip install -U rouge-score

SUMMARY="$EVAL_ROOT/summary_skt_4domains.csv"
echo "Domain,Adapter,TokenF1_micro,TokenF1_macro,ROUGE_L_F1,BERT_F1,Num" > "$SUMMARY"

# 도메인 루프
for dom in civil knowledge administrative criminal; do
  GOLD_PATH="${GOLD[$dom]:-}"
  if [[ -z "${GOLD_PATH}" || ! -f "$GOLD_PATH" ]]; then
    echo "[WARN] Skip '$dom' (gold not found: $GOLD_PATH)" >&2
    continue
  fi

  mapfile -t LINES < <(get_loras "$dom" || true)
  if [[ ${#LINES[@]} -eq 0 ]]; then
    echo "[WARN] Skip '$dom' (no LoRA configured)" >&2
    continue
  fi

  for line in "${LINES[@]}"; do
    ALIAS="${line%%:::*}"
    ADAPTER="${line#*:::}"
    if [[ ! -d "$ADAPTER" ]]; then
      echo "[WARN] Skip '$dom/$ALIAS' (adapter not found: $ADAPTER)" >&2
      continue
    fi

    RUN_DIR="$EVAL_ROOT/${dom}/${ALIAS}"
    mkdir -p "$RUN_DIR"
    PRED="$RUN_DIR/predictions.jsonl"

    # 1) 예측 생성 (없으면 생성)
    if [[ ! -s "$PRED" ]]; then
      echo "[${DOM_LABEL[$dom]} / ${ALIAS}] Generating predictions..."
      python3 /workspace/evaluate/Eval_skt/run_infer_skt.py \
        --base "$BASE" \
        --adapter "$ADAPTER" \
        --input_jsonl "$GOLD_PATH" \
        --out_jsonl "$PRED" \
        --max_new_tokens 48 --temperature 0.0 --top_p 1.0
    else
      echo "[${DOM_LABEL[$dom]} / ${ALIAS}] predictions.jsonl exists. Skip generation."
    fi

    # 2) Token-F1
    echo "[${DOM_LABEL[$dom]} / ${ALIAS}] Token-F1..."
    python3 /workspace/evaluate/Eval_skt/evaluate_token_f1.py \
      --pred "$PRED" \
      --gold "$GOLD_PATH" \
      --out_csv "$RUN_DIR/token_f1_details.csv" \
      --out_json "$RUN_DIR/token_f1_summary.json"

    # 3) ROUGE-L
    echo "[${DOM_LABEL[$dom]} / ${ALIAS}] ROUGE-L..."
    python3 /workspace/evaluate/Eval_skt/evaluate_rouge_l.py \
      --pred "$PRED" \
      --gold "$GOLD_PATH" \
      --out_csv "$RUN_DIR/rouge_l_details.csv" \
      --out_json "$RUN_DIR/rouge_l_summary.json" \
      --char_level

    # 4) BERTScore
    echo "[${DOM_LABEL[$dom]} / ${ALIAS}] BERTScore..."
    python3 /workspace/evaluate/Eval_skt/evaluate_bertscore.py \
      --pred "$PRED" \
      --gold "$GOLD_PATH" \
      --out_csv "$RUN_DIR/bertscore_details.csv" \
      --out_json "$RUN_DIR/bertscore_summary.json" \
      --model xlm-roberta-large --lang ko --batch_size 16 --rescale

    # 5) 요약 합치기
    python3 - <<'PY' "$dom" "$ALIAS" "$RUN_DIR" "$SUMMARY"
import json, sys, os
dom, alias, ddir, summary = sys.argv[1:]
def J(p):
    try: return json.load(open(p, encoding="utf-8"))
    except: return {}
tf = J(os.path.join(ddir, "token_f1_summary.json"))
rg = J(os.path.join(ddir, "rouge_l_summary.json"))
bs = J(os.path.join(ddir, "bertscore_summary.json"))
t_micro = tf.get("micro",{}).get("f1", 0.0)
t_macro = tf.get("macro",{}).get("f1", 0.0)
r_f1    = rg.get("f1", 0.0)
b_f1    = bs.get("f1", 0.0)
num     = tf.get("num_examples", rg.get("num_examples", bs.get("num_examples", 0)))
with open(summary, "a", encoding="utf-8") as w:
    w.write(f"{dom},{alias},{t_micro},{t_macro},{r_f1},{b_f1},{num}\n")
PY

  done
done

echo "[DONE] Summary → $SUMMARY"

# 보기 힌트
echo
echo "Quick view:"
echo "python3 - <<'PY'"
echo "import pandas as pd; df=pd.read_csv('$SUMMARY');"
echo "dom_map={'civil':'민사','knowledge':'지식','administrative':'행정','criminal':'형사'};"
echo "df['Domain']=df['Domain'].map(dom_map).fillna(df['Domain']);"
echo "print(df.to_string(index=False))"
echo "PY"
