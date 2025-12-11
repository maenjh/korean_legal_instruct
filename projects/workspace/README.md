# QLoRA + TRL SFTTrainer를 사용한 Instruct 학습

이 프로젝트는 QLoRA (Quantized Low-Rank Adaptation) 방식과 TRL의 SFTTrainer를 사용하여 민사법 관련 instruction 데이터로 모델을 파인튜닝합니다.

## 📁 프로젝트 구조

```
/workspace/
├── train_qlora_sft.py      # QLoRA + SFT 학습 메인 스크립트
├── inference.py             # 학습된 모델 추론 스크립트
├── requirements.txt         # 필요한 라이브러리 목록
├── setup.sh                 # 환경 설정 스크립트
├── data/
│   └── 3.개방데이터/1.데이터/Training/
│       └── civil_law_instruct.jsonl
├── models/
│   └── models--skt--A.X-3.1-Light/
└── outputs/                 # 학습된 모델 저장 위치
```

## 🎯 주요 기능

### 1. **QLoRA (Quantized LoRA)**
- **4-bit 양자화**: 메모리 사용량을 대폭 줄여 큰 모델도 학습 가능
- **LoRA 파라미터**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 2. **TRL SFTTrainer**
- Supervised Fine-Tuning에 최적화된 트레이너
- Instruction 형식 데이터 처리
- 효율적인 메모리 관리

### 3. **데이터 형식**
```json
{
  "info": {...},
  "taskinfo": {
    "instruction": "지시사항",
    "input": "질문",
    "output": "답변"
  }
}
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 실행 권한 부여
chmod +x setup.sh

# 라이브러리 설치
./setup.sh
```

또는 직접 설치:

```bash
pip install -r requirements.txt
```

### 2. 학습 실행

**방법 1: tmux 세션에서 실행 (권장)**

```bash
# tmux 세션에서 학습 시작
./run_training.sh

# 학습 진행 상황 확인 (세션 접속)
tmux attach -t qlora_training

# 세션에서 나오기 (Ctrl+B, 그 다음 D)
# 또는 학습 로그만 확인
tail -f /workspace/training.log
```

**방법 2: 직접 실행**

```bash
python train_qlora_sft.py
```

### 3. 학습 모니터링

```bash
# TensorBoard로 학습 과정 모니터링
tensorboard --logdir=/workspace/logs
```

### 4. 추론 실행

```bash
python inference.py
```

## ⚙️ 학습 설정

### 하드웨어 요구사항
- **GPU**: CUDA 지원 GPU (최소 16GB VRAM 권장)
- **메모리**: 32GB 이상 권장
- **저장공간**: 50GB 이상

### 학습 하이퍼파라미터
```python
BATCH_SIZE = 4                      # 배치 크기
GRADIENT_ACCUMULATION_STEPS = 4     # 그래디언트 누적 스텝
LEARNING_RATE = 2e-4                # 학습률
NUM_EPOCHS = 3                      # 에폭 수
MAX_SEQ_LENGTH = 2048               # 최대 시퀀스 길이
WARMUP_RATIO = 0.03                 # 웜업 비율
```

### 모델 경로
- **데이터**: `/workspace/data/3.개방데이터/1.데이터/Training/civil_law_instruct.jsonl`
- **베이스 모델**: `/workspace/models/models--skt--A.X-3.1-Light`
- **출력 디렉토리**: `/workspace/outputs/qlora-sft-civil-law`

## 📊 학습 결과

학습이 완료되면 다음 위치에 저장됩니다:
- **모델 체크포인트**: `/workspace/outputs/qlora-sft-civil-law/`
- **학습 로그**: `/workspace/logs/`

## 🔧 커스터마이징

### 1. LoRA 설정 변경
`train_qlora_sft.py`에서 다음 파라미터를 수정:
```python
LORA_R = 16          # LoRA rank (낮을수록 파라미터 수 감소)
LORA_ALPHA = 32      # LoRA alpha (일반적으로 r의 2배)
LORA_DROPOUT = 0.05  # Dropout 비율
```

### 2. 학습 파라미터 조정
```python
BATCH_SIZE = 4                    # GPU 메모리에 맞게 조정
GRADIENT_ACCUMULATION_STEPS = 4   # 유효 배치 크기 = BATCH_SIZE × 이 값
LEARNING_RATE = 2e-4              # 학습률
NUM_EPOCHS = 3                    # 에폭 수
```

### 3. 프롬프트 템플릿 변경
`format_instruction_data()` 함수에서 프롬프트 형식 수정 가능

## 💡 주요 기술

### QLoRA의 장점
1. **메모리 효율성**: 4-bit 양자화로 메모리 사용량 75% 감소
2. **학습 속도**: LoRA로 인해 전체 파인튜닝 대비 3-4배 빠름
3. **성능 유지**: 전체 파인튜닝과 유사한 성능

### TRL SFTTrainer의 특징
1. **Instruction 튜닝 최적화**
2. **자동 데이터 전처리**
3. **효율적인 배치 처리**
4. **Gradient Checkpointing 지원**

## 📝 참고사항

- **양자화**: 4-bit NF4 양자화 사용
- **옵티마이저**: paged_adamw_8bit (메모리 효율적)
- **Mixed Precision**: BFloat16 사용
- **Gradient Checkpointing**: 메모리 절약을 위해 활성화

## � tmux 사용법

### 기본 명령어
```bash
# 세션 목록 확인
tmux ls

# 세션 접속
tmux attach -t qlora_training

# 세션에서 나오기 (학습은 계속 실행됨)
Ctrl+B, 그 다음 D 키 입력

# 세션 종료 (학습도 중단됨)
tmux kill-session -t qlora_training

# 학습 로그 실시간 확인
tail -f /workspace/training.log

# 학습 로그 전체 확인
cat /workspace/training.log
```

### 세션 내에서 스크롤
- `Ctrl+B` 그 다음 `[` 키를 누르면 스크롤 모드
- 방향키나 Page Up/Down으로 스크롤
- `q` 키로 스크롤 모드 종료

## �🐛 문제 해결

### CUDA Out of Memory
```python
# train_qlora_sft.py에서 조정
BATCH_SIZE = 2  # 배치 크기 줄이기
MAX_SEQ_LENGTH = 1024  # 시퀀스 길이 줄이기
```

### 학습 속도가 느림
```python
# 그래디언트 누적 줄이기
GRADIENT_ACCUMULATION_STEPS = 2
# 또는 데이터 로더 워커 증가
dataloader_num_workers = 4
```

## 📚 참고 자료

- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [TRL 문서](https://huggingface.co/docs/trl)
- [PEFT 문서](https://huggingface.co/docs/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## 📄 라이센스

이 코드는 학습 및 연구 목적으로 자유롭게 사용 가능합니다.
