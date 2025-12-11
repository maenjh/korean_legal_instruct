"""
QLoRA + TRL SFTTrainer를 사용한 Instruct 학습 코드 (행정법 버전)
Train: /workspace/data/03.행정법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/Training/administrative_law_instruct_train.jsonl
Valid: /workspace/data/03.행정법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/Validation/administrative_law_instruct_val.jsonl
Model: /workspace/models/models--skt--A.X-3.1-Light
"""

import os
import gc
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# ==================== 설정 ====================
MODEL_PATH = "/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a"

# 행정법 학습 데이터
DATA_PATH = "/workspace/data/03.행정법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/Training/administrative_law_instruct_train.jsonl"

# 행정법 검증 데이터
VAL_DATA_PATH = "/workspace/data/03.행정법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/Validation/administrative_law_instruct_val.jsonl"

OUTPUT_DIR = "/workspace/outputs/qlora-sft-administrative-law-skt"
LOGS_DIR = "/workspace/logs"

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 학습 설정
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.03
SAVE_STEPS = 500
LOGGING_STEPS = 10


# ==================== QLoRA 설정 ====================
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ==================== LoRA 설정 ====================
def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ==================== 데이터 전처리 ====================
def format_instruction_prompt(instruction, input_text, output):
    if input_text:
        prompt = f"""### 지시사항:
{instruction}

### 질문:
{input_text}

### 답변:
{output}"""
    else:
        prompt = f"""### 지시사항:
{instruction}

### 답변:
{output}"""
    return prompt


def load_and_prepare_data_generator(data_path, max_samples=None):
    print(f"데이터 파일 읽기 시작: {data_path}")
    data_list = []
    valid_count = 0
    total_count = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1

                instruction = ""
                input_text = ""
                output = ""

                # 지원 포맷 1: {"taskinfo": {"instruction":..., "input":..., "output":...}}
                # 지원 포맷 2: {"label": {"instruction":..., "input":..., "output":...}}
                if isinstance(data.get("taskinfo"), dict) and data.get("taskinfo"):
                    taskinfo = data.get("taskinfo", {})
                    instruction = str(taskinfo.get("instruction", "")).strip()
                    input_text = str(taskinfo.get("input", "")).strip()
                    output = str(taskinfo.get("output", "")).strip()
                elif isinstance(data.get("label"), dict) and data.get("label"):
                    label = data.get("label", {})
                    instruction = str(label.get("instruction", "")).strip()
                    input_text = str(label.get("input", "")).strip()
                    output = str(label.get("output", "")).strip()
                else:
                    continue

                if not output:
                    continue

                text = format_instruction_prompt(instruction, input_text, output)
                data_list.append({"text": text})
                valid_count += 1

                if line_num % 10000 == 0:
                    print(f"  처리 중: {line_num:,} 줄 ({valid_count:,} 유효)")

                if max_samples and valid_count >= max_samples:
                    break

            except json.JSONDecodeError as e:
                print(f"  경고: {line_num}번째 줄 JSON 파싱 실패: {e}")
                continue
            except Exception as e:
                print(f"  경고: {line_num}번째 줄 처리 중 오류: {e}")
                continue

    print(f"\n데이터 로딩 완료:\n  전체: {total_count:,} 줄\n  유효: {valid_count:,} 샘플")
    if valid_count == 0:
        raise ValueError("유효한 데이터가 없습니다!")
    return data_list


def create_dataset_from_list(data_list):
    print("Dataset 객체 생성 중...")
    dataset = Dataset.from_dict({"text": [item["text"] for item in data_list]})
    print(f"Dataset 크기: {len(dataset):,}")
    return dataset


# ==================== 메인 학습 함수 ====================
def train():
    print("=" * 50)
    print("QLoRA + TRL SFTTrainer 학습 시작 (행정법)")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1) 토크나이저
    print(f"\n토크나이저 로드 중: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2) 모델 (4bit)
    print(f"\n모델 로드 중 (4bit 양자화): {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("\n모델을 k-bit training에 맞게 준비 중...")
    model = prepare_model_for_kbit_training(model)

    print("\nLoRA 적용 중...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3) Train 데이터 로드
    print(f"\n학습 데이터 로드 중: {DATA_PATH}")
    train_data_list = load_and_prepare_data_generator(DATA_PATH)
    gc.collect()
    full_dataset = create_dataset_from_list(train_data_list)
    del train_data_list; gc.collect()

    print("\n학습/테스트 데이터셋 분리 중 (90% train, 10% test)...")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    print(f"학습 데이터셋 크기: {len(train_dataset):,}")
    print(f"테스트 데이터셋 크기: {len(test_dataset):,}")

    del full_dataset; gc.collect()

    # 4) Validation 데이터 로드
    print(f"\n검증 데이터 로드 중: {VAL_DATA_PATH}")
    val_data_list = load_and_prepare_data_generator(VAL_DATA_PATH)
    gc.collect()
    val_dataset = create_dataset_from_list(val_data_list)
    del val_data_list; gc.collect()

    # 5) 학습 인자
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_dir=LOGS_DIR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )

    # 6) 토큰화
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False
        )

    print("\n학습 데이터셋 토큰화 중...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    print("\n검증 데이터셋 토큰화 중...")
    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )

    # 7) Trainer 설정 및 학습
    print("\nSFTTrainer 설정 중...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 50)
    print("학습 시작 (행정법)")
    print("=" * 50)
    trainer.train()

    print("\n모델 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print(f"학습 완료! 모델 저장 위치: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    train()
