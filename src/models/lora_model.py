"""
LoRA model definitions and utilities.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional


class LoRAModel:
    """Wrapper for LoRA-adapted language models."""
    
    def __init__(
        self,
        base_model_name: str,
        lora_config: Optional[LoraConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize LoRA model.
        
        Args:
            base_model_name: Name or path of base model
            lora_config: LoRA configuration
            device: Device to load model on
        """
        self.device = device
        self.base_model_name = base_model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=True if device == "cuda" else False
        )
        
        # Apply LoRA if config provided
        if lora_config:
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def load_lora_weights(self, lora_path: str):
        """
        Load trained LoRA weights.
        
        Args:
            lora_path: Path to LoRA checkpoint
        """
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            device_map="auto" if self.device == "cuda" else None
        )
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate response for given prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated response (remove prompt)
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Create LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability
        bias: Bias training mode
        task_type: Task type
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
