#!/usr/bin/env bash
set -euo pipefail

BASE=/workspace/evaluate/Eval_skt
BASE_MODEL="/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a"

# 이번에 돌릴 도메인 3개
DOMAINS=("knowledge" "administrative" "criminal")

# 체크포인트(필요 시 추가 가능)
CKPTS=("ckpt9000")

SUM="$BASE/summary_skt_4domains.csv"
# 요약표가 없으면 헤더 생성 (EM 제외)
[ -f "$SUM" ] || echo "Domain,Checkpoint,TokenF1,ROUGE-L,BERTScoreF1,Num" > "$SUM"

mk_remaining () {
  # $1=DOM $2=CKPT -> remaining 파일 생성하고 남은 개수 출력
  python3 - <<'PY1'
import json, pathlib, sys
base = pathlib.Path("/workspace/evaluate/Eval_skt")
dom, ckpt = sys.argv[1], sys.argv[2]
pred = base / dom / ckpt / "predictions.jsonl"
gold = base / dom / "gold_val_std.jsonl"
rem  = base / dom / "gold_val_std.remaining.jsonl"

done = set()
if pred.exists():
    with pred.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

todo = 0
with gold.open(encoding="utf-8") as g, rem.open("w", encoding="utf-8") as w:
    for line in g:
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("id") in done:
            continue
        w.write(line)
        todo += 1
print(todo)
PY1
}

dedup_predictions () {
  # $1 = PRED 파일 경로
  python3 - <<'PY2'
import json, collections, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    sys.exit(0)
out = p.with_name("predictions.dedup.jsonl")
d = collections.OrderedDict()
with p.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                o = json.loads(line)
                d[o["id"]] = o
            except Exception:
                pass
with out.open("w", encoding="utf-8") as w:
    for o in d.values():
        w.write(json.dumps(o, ensure_ascii=False) + "\n")
bak = p.with_suffix(p.suffix + ".bak")
p.rename(bak)
out.rename(p)
print("[dedup] backup ->", bak)
PY2
}

for DOM in "${DOMAINS[@]}"; do
  GOLD="$BASE/$DOM/gold_val_std.jsonl"
  if [ ! -f "$GOLD" ]; then
    echo "[WARN] Skip '$DOM' (gold not found: $GOLD)"
    continue
  fi

  for CKPT in "${CKPTS[@]}"; do
    OUTDIR="$BASE/$DOM/$CKPT"
    PRED="$OUTDIR/predictions.jsonl"
    ADAPTER="/workspace/outputs/qlora-sft-${DOM}-law-skt/${CKPT/ckpt/checkpoint-}"
    if [ ! -d "$ADAPTER" ]; then
      echo "[WARN] Skip '$DOM' / '$CKPT' (adapter not found: $ADAPTER)"
      continue
    fi
    mkdir -p "$OUTDIR"

    echo
    echo "===================="
    echo "[RUN] $DOM / $CKPT"
    echo "===================="

    # 1) 남은 목록
    REM_COUNT=$(mk_remaining "$DOM" "$CKPT")
    echo "[INFO] remaining: $REM_COUNT"

    # 2) 인퍼런스(남은 것만)
    if [ "$REM_COUNT" -gt 0 ]; then
      python3 "$BASE/run_infer_skt.py" \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --input_jsonl "$BASE/$DOM/gold_val_std.remaining.jsonl" \
        --out_jsonl "$OUTDIR/pred_tmp.jsonl" \
        --id_field id --input_field question \
        --suffix $'\n### 답변:' --max_new_tokens 128 --batch_size 8 --dtype bfloat16 --no_sample

      # 3) 이어붙이고 임시 삭제
      cat "$OUTDIR/pred_tmp.jsonl" >> "$PRED"
      rm -f "$OUTDIR/pred_tmp.jsonl"
    else
      echo "[INFO] already complete. skip inference."
    fi

    # (옵션) 중복 정리
    [ -f "$PRED" ] && dedup_predictions "$PRED" || true

    # 4) 전량 여부(정보용)
    echo "[COUNT] gold vs pred:"
    wc -l "$GOLD" || true
    wc -l "$PRED" || true

    # 5) 지표 3종
    python3 "$BASE/evaluate_token_f1.py" \
      --pred "$PRED" --gold "$GOLD" \
      --out_csv "$OUTDIR/token_f1_details.csv" \
      --out_json "$OUTDIR/token_f1_summary.json" || true

    python3 "$BASE/evaluate_rouge_l.py" \
      --pred "$PRED" --gold "$GOLD" \
      --out_csv "$OUTDIR/rouge_l_details.csv" \
      --out_json "$OUTDIR/rouge_l_summary.json" \
      --char_level || true

    python3 "$BASE/evaluate_bertscore.py" \
      --pred "$PRED" --gold "$GOLD" \
      --out_csv "$OUTDIR/bertscore_details.csv" \
      --out_json "$OUTDIR/bertscore_summary.json" \
      --model xlm-roberta-large --lang ko --batch_size 16 --rescale || true

    # 6) 요약표 반영(EM 제외)
    TF1=$(jq -r '.micro.f1 // empty'  "$OUTDIR/token_f1_summary.json")
    RGF=$(jq -r '.f1 // empty'        "$OUTDIR/rouge_l_summary.json")
    BSF=$(jq -r '.f1 // empty'        "$OUTDIR/bertscore_summary.json")
    NUM=$(jq -r '.num_examples // empty' "$OUTDIR/token_f1_summary.json")
    if [ -z "${NUM:-}" ] && [ -f "$PRED" ]; then NUM=$(wc -l < "$PRED"); fi
    echo "$DOM,$CKPT,${TF1:-},${RGF:-},${BSF:-},${NUM:-}" >> "$SUM"
    echo "[OK] summarized: $DOM / $CKPT"
  done
done

echo
echo "[DONE] knowledge/administrative/criminal evaluated."
echo "[SUM] $SUM"
column -s, -t "$SUM" 2>/dev/null || cat "$SUM"
