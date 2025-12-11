#!/usr/bin/env python3
"""
Domain Router 테스트 스크립트
- 학습된 모델: /workspace/models/domain_router_classifier
- 입력 문장: text (문자열)
- 출력: 예측된 도메인(label)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====================
# 설정
# ====================
MODEL_PATH = "/workspace/models/domain_router_classifier"

# 모델/토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# GPU 사용 가능 시 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ====================
# 테스트용 함수
# ====================
def predict_domain(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label = model.config.id2label[pred]
    return label

# ====================
# 테스트 실행
# ====================
if __name__ == "__main__":
    print("🧠 Domain Router 테스트 시작!")
    while True:
        text = input("\n문장을 입력하세요 (종료하려면 q 입력): ")
        if text.lower() == "q":
            break
        pred_label = predict_domain(text)
        print(f"➡️ 예측된 도메인: {pred_label}")
