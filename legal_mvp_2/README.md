# 법률 AI 상담 시스템 (Legal MVP 2)

법률 상담을 위한 AI 챗봇 시스템입니다. 사용자의 질문을 자동으로 법률 도메인으로 분류하고, 해당 도메인에 특화된 AI 모델을 사용하여 전문적인 법률 상담 응답을 생성합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 실행](#설치-및-실행)
- [설정](#설정)
- [사용 방법](#사용-방법)

## 🎯 프로젝트 개요

이 프로젝트는 4가지 법률 도메인(민사법, 형사법, 행정법, 지식재산권법)에 특화된 AI 모델을 활용하여 법률 상담 서비스를 제공하는 웹 애플리케이션입니다.

### 지원 법률 도메인

- **민사법 (Civil Law)**: 민사 분쟁, 계약, 손해배상 등
- **형사법 (Criminal Law)**: 형사 사건, 범죄 관련 법률 문제
- **행정법 (Administrative Law)**: 행정 처분, 행정 소송 등
- **지식재산권법 (IP Law)**: 특허, 상표, 저작권 등

## ✨ 주요 기능

1. **자동 도메인 분류**: 사용자 질문을 분석하여 적절한 법률 도메인을 자동으로 감지
2. **수동 도메인 선택**: 사용자가 직접 법률 도메인을 선택할 수 있는 옵션 제공
3. **도메인별 특화 모델**: 각 법률 도메인에 맞춤화된 LoRA 어댑터를 사용하여 전문적인 응답 생성
4. **대화 컨텍스트 유지**: 이전 대화 내역을 기반으로 맥락을 이해한 응답 생성
5. **사용자 친화적 UI**: 직관적인 웹 인터페이스를 통한 실시간 상담

## 📁 프로젝트 구조

```
legal_mvp_2/
├── app.py                 # Flask 웹 애플리케이션 메인 파일
├── config.py              # 설정 파일 (모델 경로, 디바이스 설정 등)
├── generator.py            # 응답 생성기 (도메인 분류 및 프롬프트 구성)
├── llm_manager.py         # LLM 모델 관리자 (베이스 모델 및 LoRA 어댑터 관리)
├── router.py              # 도메인 라우터 (질문을 법률 도메인으로 분류)
├── requirements.txt       # Python 패키지 의존성
├── templates/
│   └── chat.html          # 웹 인터페이스 HTML 템플릿
└── README.md              # 프로젝트 문서
```

## 🏗️ 시스템 아키텍처

이 시스템은 **"Dynamic LoRA Adapter Switching (동적 어댑터 전환)"** 방식을 사용하는 **Multi-Domain Legal Chatbot**입니다. 사용자의 질문 의도를 파악하여 가장 적합한 법률 분야(민사, 형사, 행정, 지재권)의 전문 지식을 가진 어댑터(Adapter)를 실시간으로 교체하여 답변을 생성합니다.

### 아키텍처 다이어그램

```mermaid
graph TD
    %% 스타일 정의
    classDef user fill:#f9f,stroke:#333,stroke-width:2px;
    classDef flask fill:#ff9,stroke:#333,stroke-width:2px;
    classDef logic fill:#9cf,stroke:#333,stroke-width:2px;
    classDef model fill:#f96,stroke:#333,stroke-width:2px;
    classDef storage fill:#ddd,stroke:#333,stroke-width:2px;

    %% 노드 정의
    User((User / Client)):::user
    
    subgraph "Backend Server (Flask)"
        App[app.py<br/>(API Endpoint)]:::flask
        Gen[generator.py<br/>(ResponseGenerator)]:::logic
    end

    subgraph "AI Engine"
        Router[router.py<br/>(DomainRouter)]:::logic
        LLM_Mgr[llm_manager.py<br/>(LLMManager)]:::logic
        
        subgraph "Models"
            RouterModel[Router Model<br/>(Fine-tuned BERT)]:::model
            BaseLLM[Base LLM<br/>(skt/A.X-3.1-Light)]:::model
            
            subgraph "LoRA Adapters"
                Civil[Civil Law Adapter]:::storage
                Criminal[Criminal Law Adapter]:::storage
                Admin[Admin Law Adapter]:::storage
                IP[IP Law Adapter]:::storage
            end
        end
    end

    %% 흐름 연결
    User -->|POST /api/chat| App
    App -->|1. Generate Request| Gen
    
    Gen -->|2. Predict Domain| Router
    Router -->|Inference| RouterModel
    RouterModel -->|Result: 'civil'| Router
    Router -->|Return Domain| Gen
    
    Gen -->|3. Set Adapter('civil')| LLM_Mgr
    LLM_Mgr -->|Switch Active Adapter| BaseLLM
    
    %% 어댑터 선택 로직 시각화
    LLM_Mgr -.->|Select| Civil
    LLM_Mgr -.->|Select| Criminal
    LLM_Mgr -.->|Select| Admin
    LLM_Mgr -.->|Select| IP
    
    Gen -->|4. Generate(Prompt)| LLM_Mgr
    LLM_Mgr -->|Inference with Adapter| BaseLLM
    BaseLLM -->|Generated Text| LLM_Mgr
    LLM_Mgr -->|Response| Gen
    Gen -->|JSON Response| App
    App -->|Display| User
```

### 데이터 처리 흐름 (Data Flow)

1.  **사용자 입력**: 사용자가 법률 질문을 입력합니다.
2.  **도메인 분류 (Router)**: `DomainRouter`가 BERT 모델을 사용하여 질문을 4가지 카테고리(민사, 형사, 행정, 지식재산권) 중 하나로 분류합니다.
3.  **어댑터 교체 (Adapter Switching)**: `LLMManager`가 분류된 도메인에 맞는 LoRA 어댑터를 Base LLM에 활성화(Active)합니다.
4.  **답변 생성 (Generation)**: 해당 도메인에 맞는 프롬프트와 어댑터가 적용된 LLM이 전문적인 법률 답변을 생성합니다.
5.  **응답 반환**: 생성된 답변이 사용자에게 전달됩니다.

### 컴포넌트 설명

#### 1. **app.py** - Flask 웹 애플리케이션
- Flask 서버 초기화 및 라우팅 처리
- `/`: 메인 채팅 인터페이스 제공
- `/api/chat`: POST 요청을 받아 AI 응답 생성

#### 2. **config.py** - 설정 관리
- 모델 경로 설정 (베이스 모델, 라우터, LoRA 어댑터)
- 디바이스 설정 (CUDA/CPU)
- 생성 파라미터 설정 (temperature, top_p, max_new_tokens 등)
- 도메인 매핑 설정

#### 3. **router.py** - 도메인 라우터
- `DomainRouter` 클래스: 사용자 질문을 4가지 법률 도메인으로 분류
- Transformers 기반 시퀀스 분류 모델 사용
- 자동으로 적절한 법률 도메인 감지

#### 4. **llm_manager.py** - LLM 관리자
- `LLMManager` 클래스: 싱글톤 패턴으로 구현
- 베이스 LLM 모델 로드 (SKT A.X-3.1-Light)
- 4가지 LoRA 어댑터 동적 로드 및 전환
- 텍스트 생성 기능 제공

#### 5. **generator.py** - 응답 생성기
- `ResponseGenerator` 클래스: 전체 응답 생성 파이프라인 관리
- 도메인 라우터와 LLM 관리자 통합
- 프롬프트 템플릿 구성 (법률 상담 변호사 역할 설정)
- 대화 히스토리 관리 및 포맷팅

#### 6. **templates/chat.html** - 웹 인터페이스
- 반응형 채팅 UI
- 실시간 메시지 전송 및 수신
- 도메인 선택 드롭다운
- 법률 조항 하이라이팅 및 포맷팅

### 데이터 흐름

```
사용자 질문
    ↓
[app.py] POST /api/chat
    ↓
[generator.py] ResponseGenerator.generate()
    ↓
[router.py] DomainRouter.predict() → 도메인 분류
    ↓
[llm_manager.py] LLMManager.set_adapter() → 도메인별 어댑터 전환
    ↓
[generator.py] 프롬프트 구성
    ↓
[llm_manager.py] LLMManager.generate_response() → 응답 생성
    ↓
[app.py] JSON 응답 반환
    ↓
[chat.html] UI 업데이트
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- `flask`: 웹 프레임워크
- `torch`: PyTorch (딥러닝 프레임워크)
- `transformers`: Hugging Face Transformers 라이브러리
- `peft`: Parameter-Efficient Fine-Tuning (LoRA 지원)
- `accelerate`: 모델 가속화
- `sentencepiece`: 토크나이저
- `protobuf`: 프로토콜 버퍼

### 2. 모델 경로 확인

`config.py` 파일에서 다음 경로들이 올바르게 설정되어 있는지 확인하세요:

- `BASE_MODEL_PATH`: 베이스 LLM 모델 경로
- `ROUTER_PATH`: 도메인 라우터 모델 경로
- `LORA_ADAPTERS`: 각 도메인별 LoRA 어댑터 경로

### 3. 애플리케이션 실행

```bash
python app.py
```

서버가 시작되면 브라우저에서 `http://localhost:7860`으로 접속하세요.

## ⚙️ 설정

### config.py 주요 설정 항목

#### 모델 경로
```python
BASE_MODEL_PATH = "/raid/home/nlp4/workspace/models/..."
ROUTER_PATH = "/raid/home/nlp4/workspace/models/domain_router_classifier/..."
LORA_ADAPTERS = {
    "civil": "...",
    "criminal": "...",
    "admin": "...",
    "ip": "..."
}
```

#### 디바이스 설정
```python
DEVICE = "cuda:4"  # GPU 사용 시
# 또는
DEVICE = "cpu"     # CPU 사용 시
```

#### 생성 파라미터
```python
GENERATION_CONFIG = {
    "max_new_tokens": 512,      # 최대 생성 토큰 수
    "do_sample": True,           # 샘플링 활성화
    "top_p": 0.9,               # Nucleus sampling
    "temperature": 0.7,         # 생성 온도
    "repetition_penalty": 1.1   # 반복 패널티
}
```

## 💻 사용 방법

### 웹 인터페이스 사용

1. 브라우저에서 `http://localhost:7860` 접속
2. 질문 입력창에 법률 관련 질문 입력
3. **자동 감지** 또는 **수동 도메인 선택**:
   - 자동 감지: 도메인 선택을 "🧠 자동 감지"로 두면 AI가 자동으로 분류
   - 수동 선택: 드롭다운에서 원하는 법률 도메인 선택
4. "전송" 버튼 클릭 또는 Enter 키 입력
5. AI 응답 확인

### API 사용

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "계약 위반 시 손해배상을 받을 수 있나요?",
    "history": [],
    "manual_domain": null
  }'
```

응답 예시:
```json
{
  "predicted_domain": "civil",
  "answer": "네, 계약 위반 시 손해배상을 받을 수 있습니다...",
  "history": [
    {"role": "user", "content": "계약 위반 시 손해배상을 받을 수 있나요?"},
    {"role": "assistant", "content": "네, 계약 위반 시 손해배상을 받을 수 있습니다..."}
  ]
}
```

## 📝 주요 특징

- **도메인 특화 모델**: 각 법률 도메인에 맞춤화된 LoRA 어댑터 사용
- **동적 어댑터 전환**: 사용자 질문에 따라 실시간으로 적절한 모델로 전환
- **컨텍스트 인식**: 대화 히스토리를 고려한 맥락적 응답 생성
- **사용자 친화적 프롬프트**: 법률 전문가 역할의 친절하고 명확한 응답 생성
- **법률 조항 하이라이팅**: UI에서 법률 조항을 시각적으로 강조 표시

## 🔧 기술 스택

- **Backend**: Flask (Python)
- **AI/ML**: 
  - PyTorch
  - Hugging Face Transformers
  - PEFT (LoRA)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Model**: SKT A.X-3.1-Light (베이스 모델)

## 📌 주의사항

1. **모델 파일 경로**: `config.py`에서 모델 경로가 올바르게 설정되어 있어야 합니다.
2. **GPU 메모리**: 여러 LoRA 어댑터를 동시에 로드하므로 충분한 GPU 메모리가 필요합니다.
3. **초기 로딩 시간**: 첫 실행 시 모든 모델을 로드하므로 시간이 걸릴 수 있습니다.
4. **법률 상담 한계**: 이 시스템은 참고용 정보를 제공하며, 실제 법률 문제는 전문 변호사와 상담하시기 바랍니다.

## 📄 라이선스

이 프로젝트의 라이선스 정보는 별도로 확인하시기 바랍니다.

