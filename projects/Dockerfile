# NVIDIA PyTorch 컨테이너 이미지 기반
FROM nvcr.io/nvidia/pytorch:23.10-py3

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    bash \
    dash \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업그레이드 (옵션)
RUN pip install --upgrade pip setuptools wheel

# requirements.txt 복사 및 설치 (필요 시)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY . .

# 기본 실행 명령
CMD ["/bin/bash"]
# 필요하면 아래 주석 풀어서 JupyterLab으로 변경 가능
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
