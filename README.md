# Korean Legal Instruction Tuning Project

한국어 법률 데이터를 활용한 LoRA 기반 법률 도메인 특화 모델 학습 및 도메인 라우터 구현 프로젝트

## 프로젝트 개요

AIHub의 한국어 법률 데이터를 활용하여 4개의 법률 도메인에 특화된 LoRA 모델을 학습하고, 입력 질의를 적절한 도메인으로 분류하는 도메인 라우터를 구현한 MVP입니다.

### 지원 법률 도메인
- **민사법** (Civil Law)
- **형사법** (Criminal Law)
- **지식재산권법** (Intellectual Property Law)
- **행정법** (Administrative Law)

## 프로젝트 구조

```
korean_legal_instruct/
├── src/
│   ├── data/              # 데이터 로딩 및 전처리
│   ├── models/            # 모델 정의
│   ├── training/          # 학습 스크립트
│   └── inference/         # 추론 및 라우팅 로직
├── scripts/               # 실행 스크립트
├── examples/              # 사용 예제
└── requirements.txt       # 의존성 패키지
```

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/maenjh/korean_legal_instruct.git
cd korean_legal_instruct

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비
AIHub에서 한국어 법률 데이터를 다운로드하여 `data/raw/` 디렉토리에 배치합니다.

```bash
python scripts/prepare_data.py --data_dir data/raw --output_dir data/processed
```

### 2. LoRA 모델 학습
4개의 도메인별 LoRA 모델을 학습합니다.

```bash
# 민사법 모델 학습
python scripts/train_lora.py --domain civil --config configs/civil_config.yaml

# 형사법 모델 학습
python scripts/train_lora.py --domain criminal --config configs/criminal_config.yaml

# 지식재산권법 모델 학습
python scripts/train_lora.py --domain ip --config configs/ip_config.yaml

# 행정법 모델 학습
python scripts/train_lora.py --domain administrative --config configs/administrative_config.yaml
```

### 3. 도메인 라우터 학습
입력 질의를 적절한 법률 도메인으로 분류하는 라우터를 학습합니다.

```bash
python scripts/train_router.py --data_dir data/processed --output_dir models/router
```

### 4. MVP 실행
학습된 모델을 사용하여 질의에 답변합니다.

```bash
python examples/mvp_demo.py --query "계약 해지에 대한 법적 근거는?"
```

## 모델 아키텍처

- **Base Model**: Polyglot-ko 또는 KoGPT 계열 모델
- **LoRA**: Low-Rank Adaptation으로 각 도메인별 효율적 파인튜닝
- **Router**: 텍스트 분류 모델 기반 도메인 분류기

## 주요 기능

1. **도메인 자동 인식**: 입력 질의를 분석하여 적절한 법률 도메인 자동 선택
2. **전문화된 답변**: 각 도메인에 특화된 LoRA 모델을 통한 정확한 법률 정보 제공
3. **효율적 학습**: LoRA를 활용한 파라미터 효율적 학습
4. **확장 가능**: 새로운 법률 도메인 추가 용이

## 성능

(학습 후 업데이트 예정)

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트를 환영합니다.

## 참고 자료

- [AIHub 한국어 법률 데이터](https://aihub.or.kr/)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
- [Transformers 라이브러리](https://github.com/huggingface/transformers)