"""
민사법 Instruction Tuning 모델 비교 평가 코드
- 평가 대상: LLaMA 3.1 vs SKT A.X-3.1-Light
- 지표: Exact Match, Token F1, ROUGE-L, BERTScore-F1
"""

import json
import re
import os
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import pandas as pd

# ==================== 공통 설정 ====================
DATA_PATH = "/workspace/data/01.민사법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/test_split.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32
SAMPLE_SIZE = 500

# ==================== 모델별 경로 ====================
MODELS = [
    {
        "name": "LLaMA-3.1-8B-Instruct",
        "base": "/workspace/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "lora": "/workspace/outputs/qlora-sft-civil-law-llama",
    },
    {
        "name": "SKT-A.X-3.1-Light",
        "base": "/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a",
        "lora": "/workspace/outputs/qlora-sft-civil-law-skt",
    },
]

# ==================== 유틸 함수 ====================
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ==================== 평가 함수 ====================
def evaluate_model(base_model_path, lora_model_path, model_name):
    print(f"\n📘 {model_name} 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("⚙️ pad_token이 설정되지 않아 eos_token으로 대체했습니다.")
    print(f"✅ {model_name} 토크나이저 로드 완료")

    print(f"📗 {model_name} 모델 로드 중 (LoRA 적용): {lora_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        lora_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"✅ {model_name} 모델 로드 완료")

    # 데이터 로드
    print(f"\n📄 평가 데이터 로드 중: {DATA_PATH}")
    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("text", "")
            if "### 답변:" in text:
                parts = text.split("### 답변:")
                prompt = parts[0].strip()
                gold = parts[1].strip()
                data.append({"prompt": prompt, "gold": gold})
    print(f"총 평가 샘플 수: {len(data)}")

    if SAMPLE_SIZE and SAMPLE_SIZE < len(data):
        data = data[:SAMPLE_SIZE]
        print(f"⚡ 샘플링 적용: 상위 {SAMPLE_SIZE}개 데이터만 평가")

    em_scores, f1_scores, preds, golds = [], [], [], []

    # 배치 평가
    print(f"\n🚀 {model_name} 배치 평가 시작 (batch_size={BATCH_SIZE})")
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"Evaluating {model_name}"):
        batch = data[i:i + BATCH_SIZE]
        prompts = [d["prompt"].strip() + "\n### 답변:" for d in batch]
        gold_batch = [d["gold"].strip() for d in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        pred_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred, gold, prompt in zip(pred_batch, gold_batch, prompts):
            if prompt in pred:
                pred = pred.replace(prompt, "").strip()
            em_scores.append(exact_match(pred, gold))
            f1_scores.append(token_f1(pred, gold))
            preds.append(pred)
            golds.append(gold)

    # 의미 기반 평가
    print(f"\n📐 {model_name} 의미 기반 평가 (ROUGE-L, BERTScore) 계산 중...")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(golds, preds)]

    P, R, F1 = bert_score(preds, golds, lang="ko", verbose=False, rescale_with_baseline=True)
    bert_f1_avg = float(torch.mean(F1))

    # 평균 계산
    result = {
        "Model": model_name,
        "Exact Match": sum(em_scores) / len(em_scores),
        "Token F1": sum(f1_scores) / len(f1_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "BERTScore-F1": bert_f1_avg,
    }

    print("\n====================== 📊 평가 결과 ======================")
    print(f"📘 모델: {model_name}")
    print(f"📊 Exact Match: {result['Exact Match']:.4f}")
    print(f"📊 Token F1:    {result['Token F1']:.4f}")
    print(f"📊 ROUGE-L:     {result['ROUGE-L']:.4f}")
    print(f"📊 BERTScore-F1:{result['BERTScore-F1']:.4f}")
    print("==========================================================")

    return result


# ==================== 실행 ====================
if __name__ == "__main__":
    results = []

    for model_info in MODELS:
        result = evaluate_model(model_info["base"], model_info["lora"], model_info["name"])
        results.append(result)

    # CSV 저장
    df = pd.DataFrame(results)
    save_path = "/workspace/eval_moon/results_civil_law_compare.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 평가 완료! 결과가 CSV로 저장되었습니다:")
    print(f"📄 {save_path}")
    print(df)

