"""
Training utilities for LoRA models.
"""
import os
import yaml
from typing import Dict, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model
from datasets import Dataset

from ..models.lora_model import create_lora_config
from ..data.data_loader import format_instruction


class LoRATrainer:
    """Trainer for domain-specific LoRA models."""
    
    def __init__(self, config_path: str, base_config_path: str = None):
        """
        Initialize LoRA trainer.
        
        Args:
            config_path: Path to domain-specific config
            base_config_path: Path to base config (optional)
        """
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if base_config_path:
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
                # Merge configs (domain config overrides base)
                base_config.update(self.config)
                self.config = base_config
        
        self.domain = self.config.get('domain', 'unknown')
        self.output_dir = self.config.get('output_dir', f'models/lora/{self.domain}')
        
        # Initialize model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize base model and apply LoRA."""
        base_model_name = self.config.get('base_model', 'beomi/KoAlpaca-Polyglot-5.8B')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            load_in_8bit=True if self.device == "cuda" else False
        )
        
        # Create and apply LoRA config
        lora_config = self.config.get('lora_config', {})
        peft_config = create_lora_config(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules'),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
            task_type=lora_config.get('task_type', 'CAUSAL_LM')
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess examples for training.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        # Format instructions
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            response = examples.get('response', [None])[i]
            texts.append(format_instruction(instruction, response))
        
        # Tokenize
        model_inputs = self.tokenizer(
            texts,
            max_length=self.config.get('model_max_length', 512),
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        # Set labels (for causal LM, labels are the same as input_ids)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Train LoRA model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Training arguments
        training_config = self.config.get('training', {})
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 2e-4),
            warmup_steps=training_config.get('warmup_steps', 100),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            logging_steps=self.config.get('output', {}).get('logging_steps', 100),
            save_steps=self.config.get('output', {}).get('save_steps', 500),
            eval_steps=self.config.get('output', {}).get('eval_steps', 500),
            save_total_limit=self.config.get('output', {}).get('save_total_limit', 3),
            fp16=self.device == "cuda",
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"Training {self.domain} LoRA model...")
        trainer.train()
        
        # Save final model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
