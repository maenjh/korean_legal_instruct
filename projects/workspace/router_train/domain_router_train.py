#!/usr/bin/env python3
"""
Domain Router 학습 스크립트 (v2 - NVIDIA 23.10 환경 호환)
- Input: /workspace/router_train/domain_router_train.jsonl
- Output: /workspace/models/domain_router_classifier/
"""

import os
import json
import numpy as np
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# ====================
# 설정
# ====================
DATA_PATH = "/workspace/router_train/domain_router_train.jsonl"
MODEL_NAME = "/workspace/models/klue-bert-base"
OUTPUT_DIR = "/workspace/models/domain_router_classifier"
SEED = 42
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

set_seed(SEED)

# ====================
# 1️⃣ 데이터 로드
# ====================
print("📂 데이터 로드 중...")

texts, labels = [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        labels.append(obj["label"])

label_names = sorted(list(set(labels)))
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for name, i in label2id.items()}

print(f"✅ 라벨 목록: {label_names}")
print(f"✅ 총 샘플 수: {len(texts)}")

# Dataset 변환
dataset = Dataset.from_dict({
    "text": texts,
    "label": [label2id[l] for l in labels]
})
dataset = dataset.shuffle(seed=SEED)
split = dataset.train_test_split(test_size=0.1)
train_dataset, eval_dataset = split["train"], split["test"]

# ====================
# 2️⃣ 토크나이징
# ====================
print("🔤 토크나이징 중...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ====================
# 3️⃣ 모델 정의
# ====================
print("🧠 모델 초기화 중...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# ====================
# 4️⃣ 평가 함수
# ====================
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": np.mean([report[l]["f1-score"] for l in label_names])
    }

# ====================
# 5️⃣ 학습 설정
# ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",            # ✅ 최신 버전 호환 (evaluation_strategy → eval_strategy)
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=2,
    report_to="none",                 # WandB나 TensorBoard 미사용 시 none
    fp16=True,                        # H100 GPU에서 자동 mixed precision 사용
)

# ====================
# 6️⃣ Trainer 정의
# ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ====================
# 7️⃣ 학습 시작
# ====================
print("🚀 학습 시작!")
trainer.train()

# ====================
# 8️⃣ 모델 저장
# ====================
print("💾 모델 저장 중...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ====================
# 9️⃣ 평가 결과 출력 및 저장
# ====================
print("📊 최종 평가 결과:")
results = trainer.evaluate()
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# 결과 JSON 파일 저장
report_path = os.path.join(OUTPUT_DIR, "results.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 완료! 모델 및 결과 저장 경로:\n{OUTPUT_DIR}")
print(f"📁 평가 리포트: {report_path}")
