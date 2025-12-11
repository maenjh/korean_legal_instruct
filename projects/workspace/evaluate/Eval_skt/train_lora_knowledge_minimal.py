from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch, json, os

BASE_MODEL = "/workspace/models/models--skt--A.X-3.1-Light/snapshots/9b41bb2406472634d8812c0b8931fa40fa9a6c3a"
TRAIN_JSON = "/workspace/evaluate/Eval_skt/knowledge/gold_val_std.jsonl"
OUT_DIR = "/workspace/outputs/qlora-sft-knowledge-law-skt"

# 로딩
print("[load] model & tokenizer")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05)
model = get_peft_model(model, peft_config)

# 데이터 로드
print("[load] dataset")
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                yield {"text": j.get("question","") + "\n### 답변:\n" + j.get("answer","")}
data = list(load_jsonl(TRAIN_JSON))
dataset = Dataset.from_list(data)

# 학습 설정
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_steps=9000,
    logging_steps=100,
    save_steps=3000,
    bf16=True,
    optim="paged_adamw_32bit"
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args, dataset_text_field="text")
trainer.train()
model.save_pretrained(os.path.join(OUT_DIR, "checkpoint-final"))
print("[done] saved ->", OUT_DIR)
