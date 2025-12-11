import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config

class LLMManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.tokenizer = None
        self.model = None
        self.device = config.LLM_DEVICE
        self.load_models()
        self.initialized = True

    def load_models(self):
        print(f"--- Loading Base LLM from {config.BASE_MODEL_PATH} ---")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_PATH,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Base Model
        try:
            # Try loading with bfloat16 and auto device map
            self.model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device == "cuda" else {"": self.device},
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Warning] Failed to load with device_map='auto': {e}. Falling back to standard load.")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_PATH,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
            )
            self.model.to(self.device)

        print("--- Base LLM Loaded ---")

        # Load Adapters
        self._load_adapters()

    def _load_adapters(self):
        print("--- Loading LoRA Adapters ---")
        if not config.LORA_ADAPTERS:
            print("[Warning] No adapters configured.")
            return

        # Load the first adapter to initialize PeftModel
        first_key = list(config.LORA_ADAPTERS.keys())[0]
        first_path = config.LORA_ADAPTERS[first_key]
        
        print(f"Attaching first adapter: {first_key} from {first_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            first_path,
            adapter_name=first_key
        )

        # Load remaining adapters
        for key, path in config.LORA_ADAPTERS.items():
            if key == first_key:
                continue
            try:
                print(f"Loading adapter: {key} from {path}")
                self.model.load_adapter(path, adapter_name=key)
            except Exception as e:
                print(f"[Error] Failed to load adapter {key}: {e}")

        self.model.eval()
        print("--- All Adapters Loaded ---")

    def set_adapter(self, domain: str):
        if domain in config.LORA_ADAPTERS:
            try:
                self.model.set_adapter(domain)
                print(f"[LLMManager] Switched adapter to: {domain}")
            except Exception as e:
                print(f"[Error] Could not switch adapter to {domain}: {e}")
        else:
            print(f"[Warning] Adapter {domain} not found. Using current adapter.")

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_config = config.GENERATION_CONFIG.copy()
        gen_config["eos_token_id"] = self.tokenizer.eos_token_id
        gen_config["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_config
            )
        
        # Decode only the new tokens
        input_length = inputs.input_ids.shape[1]
        response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response_text
