# Korean Legal Instruct Project

이 프로젝트는 한국어 법률 상담을 위한 AI 시스템 구축 프로젝트입니다. AI Hub의 법률 데이터를 활용하여 4가지 주요 법률 도메인(민사, 형사, 행정, 지식재산권)에 특화된 모델을 학습시키고, 사용자 질문을 적절한 도메인으로 분류하여 전문적인 답변을 제공하는 시스템을 구현했습니다.

## 📁 프로젝트 구조

```
korean_legal_instruct/
├── legal_mvp_2/       # 웹 애플리케이션 및 서빙 코드 (Flask)
│   ├── app.py         # 메인 애플리케이션
│   ├── router.py      # 도메인 분류 라우터
│   ├── generator.py   # 답변 생성기
│   └── ...
└── projects/          # 모델 학습 및 데이터 처리 코드
    └── workspace/
        ├── lora_train/    # LoRA 파인튜닝 스크립트
        ├── router_train/  # 라우터 모델 학습 및 데이터 생성 스크립트
        ├── evaluate/      # 모델 평가 스크립트
        └── ...
```

## 📊 데이터셋 (Data)

본 프로젝트에서 사용된 데이터는 **AI Hub (aihub.or.kr)** 에서 제공하는 법률 관련 데이터셋을 활용하였습니다.

### 데이터 구성
총 4가지 법률 도메인의 데이터를 사용했습니다:

1. **민사법 (Civil Law)**
   - 원천 데이터: `01.민사법 LLM 사전학습 및 Instruction Tuning 데이터`
2. **지식재산권법 (Intellectual Property Law)**
   - 원천 데이터: `02.지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터`
3. **행정법 (Administrative Law)**
   - 원천 데이터: `03.행정법 LLM 사전학습 및 Instruction Tuning 데이터`
4. **형사법 (Criminal Law)**
   - 원천 데이터: `04.형사법 LLM 사전학습 및 Instruction Tuning 데이터`

### 데이터 전처리 (Preprocessing)

AI Hub의 원천 데이터(JSON)를 모델 학습에 적합한 형태로 가공하는 과정을 거쳤습니다.

1. **데이터 추출 및 통일**:
   - 각 도메인별로 상이한 JSON 구조(`taskinfo` 계열 vs `label` 계열)에서 질문(`input`), 지시사항(`instruction`), 답변(`output`)을 추출하여 통일된 포맷으로 변환했습니다.
   - **Router 학습용 데이터**: 각 도메인별 질문을 추출하여 `(질문, 도메인 라벨)` 쌍으로 구성된 데이터셋을 생성했습니다 (`projects/workspace/router_train/generate_router_data.py`).
   - **Instruction Tuning 데이터**: `(지시사항, 질문, 답변)` 쌍으로 구성된 JSONL 파일을 생성하여 LLM 학습에 활용했습니다.

2. **프롬프트 포맷팅**:
   학습 시 다음과 같은 구조의 프롬프트를 사용했습니다:
   ```text
   ### 지시사항:
   {instruction}

   ### 질문:
   {input}

   ### 답변:
   {output}
   ```

## 🚀 모델 학습 (Training)

### 1. 도메인 라우터 (Domain Router)
- 사용자 질문이 어떤 법률 도메인에 속하는지 분류하는 BERT 기반 분류 모델입니다.
- 4개 클래스(민사, 형사, 행정, 지식재산)로 분류하도록 학습되었습니다.

### 2. 법률 특화 LLM (Legal LLM)
- **Base Model**: `skt/A.X-3.1-Light` (또는 Llama-3.1-8B-Instruct)
- **학습 방법**: QLoRA (Quantized Low-Rank Adaptation) + TRL SFTTrainer
- **설정**:
  - 4-bit Quantization (메모리 효율성)
  - LoRA Rank: 16, Alpha: 32
  - 각 도메인별로 별도의 LoRA 어댑터를 학습시켜 도메인 특화 성능을 확보했습니다.

## 💻 실행 방법 (Usage)

### 웹 애플리케이션 실행
`legal_mvp_2` 폴더에서 Flask 앱을 실행하여 웹 인터페이스를 통해 상담 시스템을 사용할 수 있습니다.

```bash
cd legal_mvp_2
pip install -r requirements.txt
python app.py
```

## 📜 라이선스

이 프로젝트의 데이터는 AI Hub의 라이선스 정책을 따르며, 코드는 오픈소스로 공개됩니다.
