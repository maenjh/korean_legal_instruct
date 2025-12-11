# Quick Start Guide

이 가이드는 한국 법률 상담 시스템을 빠르게 시작하는 방법을 안내합니다.

## 사전 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장, CPU에서도 실행 가능하지만 느림)
- 최소 16GB RAM (모델 로딩 시)

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/maenjh/korean_legal_instruct.git
cd korean_legal_instruct
```

### 2. 가상환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

### AIHub 데이터 다운로드 (선택사항)

실제 AIHub 데이터를 사용하려면:

1. [AIHub](https://aihub.or.kr/)에서 법률 데이터 다운로드
2. `data/raw/` 디렉토리에 도메인별로 배치:
   ```
   data/raw/
   ├── civil/          # 민사 데이터
   ├── criminal/       # 형사 데이터
   ├── ip/             # 지식재산권 데이터
   └── administrative/ # 행정 데이터
   ```

### 데이터 전처리

```bash
python scripts/prepare_data.py --data_dir data/raw --output_dir data/processed
```

데이터가 없는 경우 자동으로 샘플 데이터가 생성됩니다.

## 모델 학습

### 1. 도메인 라우터 학습

```bash
python scripts/train_router.py --config configs/router_config.yaml --data_dir data/processed/router
```

### 2. LoRA 모델 학습 (4개 도메인)

각 법률 도메인에 대해 LoRA 모델을 학습합니다:

```bash
# 민사법 (Civil Law)
python scripts/train_lora.py --domain civil --config configs/civil_config.yaml

# 형사법 (Criminal Law)
python scripts/train_lora.py --domain criminal --config configs/criminal_config.yaml

# 지식재산권법 (IP Law)
python scripts/train_lora.py --domain ip --config configs/ip_config.yaml

# 행정법 (Administrative Law)
python scripts/train_lora.py --domain administrative --config configs/administrative_config.yaml
```

**참고**: 각 모델 학습에는 GPU에서 약 30분-2시간 소요될 수 있습니다.

## MVP 실행

### 단일 질의 모드

```bash
python examples/mvp_demo.py --query "계약 해지에 대한 법적 근거는?"
```

### 대화형 모드

```bash
python examples/mvp_demo.py
```

대화형 모드에서는 계속해서 질문을 입력할 수 있습니다. 종료하려면 `quit` 또는 `exit`를 입력하세요.

### 특정 도메인 지정

자동 라우팅 대신 특정 도메인을 사용하려면:

```bash
python examples/mvp_demo.py --query "손해배상 청구 방법은?" --domain civil
```

## 라우터 테스트

도메인 라우터의 성능을 테스트하려면:

```bash
python examples/test_router.py
```

## 디렉토리 구조

```
korean_legal_instruct/
├── configs/                 # 설정 파일
│   ├── base_config.yaml
│   ├── civil_config.yaml
│   ├── criminal_config.yaml
│   ├── ip_config.yaml
│   ├── administrative_config.yaml
│   └── router_config.yaml
├── data/                    # 데이터 디렉토리
│   ├── raw/                # 원본 데이터
│   └── processed/          # 전처리된 데이터
├── models/                  # 학습된 모델
│   ├── lora/               # LoRA 체크포인트
│   │   ├── civil/
│   │   ├── criminal/
│   │   ├── ip/
│   │   └── administrative/
│   └── router/             # 라우터 모델
├── src/                     # 소스 코드
│   ├── data/               # 데이터 로딩
│   ├── models/             # 모델 정의
│   ├── training/           # 학습 유틸리티
│   └── inference/          # 추론 파이프라인
├── scripts/                 # 실행 스크립트
│   ├── prepare_data.py
│   ├── train_lora.py
│   └── train_router.py
└── examples/                # 사용 예제
    ├── mvp_demo.py
    └── test_router.py
```

## 문제 해결

### GPU 메모리 부족

모델 학습 시 GPU 메모리가 부족한 경우:

1. `configs/base_config.yaml`에서 `batch_size`를 줄입니다 (예: 4 → 2)
2. `gradient_accumulation_steps`를 늘립니다 (예: 8 → 16)

### 모델 로딩 실패

베이스 모델 다운로드에 실패하는 경우:

1. 인터넷 연결 확인
2. Hugging Face 캐시 디렉토리 확인
3. 다른 베이스 모델 사용 (configs에서 `base_model` 변경)

### 데이터 없음 경고

실제 데이터 없이도 시스템이 작동하도록 샘플 데이터가 자동 생성됩니다. 
실제 성능을 보려면 AIHub에서 데이터를 다운로드하세요.

## 다음 단계

1. 실제 AIHub 데이터로 모델 재학습
2. 하이퍼파라미터 튜닝으로 성능 향상
3. 추가 법률 도메인 확장
4. 웹 인터페이스 구축

## 지원

문제가 발생하면 GitHub Issues에 등록해주세요.
