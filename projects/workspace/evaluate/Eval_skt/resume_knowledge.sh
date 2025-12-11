#!/usr/bin/env bash
set -euo pipefail

# ---- 경로 설정 ----
SCRIPT=/workspace/evaluate/Eval_skt/train_lora_knowledge_minimal.py
BASE_MODEL="/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a"
DATA_JSONL="/workspace/evaluate/Eval_skt/knowledge/gold_val_std.jsonl"
OUT_DIR="/workspace/outputs/qlora-sft-knowledge-law-skt"
LOG_DIR="/workspace/outputs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/train_knowledge_${TS}.log"

# ---- 사전 체크 ----
[ -f "$SCRIPT" ] || { echo "[ERR] $SCRIPT 없음"; exit 1; }
[ -f "$DATA_JSONL" ] || { echo "[ERR] gold 데이터 없음: $DATA_JSONL"; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- 최신 체크포인트 탐색 ----
LATEST_CKPT=$(ls -dt ${OUT_DIR}/checkpoint-* 2>/dev/null | head -n1 || true)

echo "[INFO] Base Model : $BASE_MODEL"
echo "[INFO] Dataset    : $DATA_JSONL"
echo "[INFO] Output Dir : $OUT_DIR"
echo "[INFO] Latest Ckpt: ${LATEST_CKPT:-<none>}"
echo "[INFO] Log        : $LOG"

# ---- 실행 (resume or fresh) ----
if [ -n "${LATEST_CKPT:-}" ]; then
  echo "[RUN] RESUME from ${LATEST_CKPT}"
  nohup python3 "$SCRIPT" \
    --base "$BASE_MODEL" \
    --train_jsonl "$DATA_JSONL" \
    --out_dir "$OUT_DIR" \
    --resume_from_checkpoint "$LATEST_CKPT" \
    > "$LOG" 2>&1 &
else
  echo "[RUN] FRESH START"
  nohup python3 "$SCRIPT" \
    --base "$BASE_MODEL" \
    --train_jsonl "$DATA_JSONL" \
    --out_dir "$OUT_DIR" \
    > "$LOG" 2>&1 &
fi

PID=$!
echo "[OK] Started. PID=${PID}"
echo "[TIP] 진행 상황 보기:   tail -f \"$LOG\""
echo "[TIP] GPU 사용 확인:    nvidia-smi"
echo "[TIP] 프로세스 확인:     ps -ef | grep train_lora_knowledge_minimal.py | grep -v grep"
