import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = "/raid/home/nlp4/workspace"

# Model Paths
BASE_MODEL_PATH = os.getenv(
    "BASE_MODEL_PATH",
    os.path.join(WORKSPACE_ROOT, "models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a")
)

ROUTER_PATH = os.getenv(
    "ROUTER_PATH",
    os.path.join(WORKSPACE_ROOT, "models/domain_router_classifier/checkpoint-2700")
)

# LoRA Adapters
# Corrected mapping based on folder names:
# admin -> administrative-law-skt
# ip -> intellectual-property-law-skt
LORA_ADAPTERS = {
    "civil": os.path.join(WORKSPACE_ROOT, "projects/workspace/outputs/qlora-sft-civil-law-skt"),
    "criminal": os.path.join(WORKSPACE_ROOT, "projects/workspace/outputs/qlora-sft-criminal-law-skt"),
    "admin": os.path.join(WORKSPACE_ROOT, "projects/workspace/outputs/qlora-sft-administrative-law-skt"),
    "ip": os.path.join(WORKSPACE_ROOT, "projects/workspace/outputs/qlora-sft-intellectual-property-law-skt"),
}

# Domain Mapping
DOMAIN_MAP = {
    0: "admin",      # 행정법
    1: "civil",      # 민사법
    2: "criminal",   # 형사법
    3: "ip"          # 지식재산권법
}

ID2LABEL = {i: v.upper() for i, v in DOMAIN_MAP.items()}
LABEL2ID = {v.upper(): i for i, v in DOMAIN_MAP.items()}

# Device Configuration
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
ROUTER_DEVICE = DEVICE
LLM_DEVICE = DEVICE

# Generation Config
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "repetition_penalty": 1.1
}
