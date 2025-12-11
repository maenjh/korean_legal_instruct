"""
Training utilities for domain router.
"""
import os
import yaml
import torch
from typing import Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RouterTrainer:
    """Trainer for domain classification router."""
    
    def __init__(self, config_path: str):
        """
        Initialize router trainer.
        
        Args:
            config_path: Path to router config
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.labels = self.config.get('labels', ['civil', 'criminal', 'ip', 'administrative'])
        self.num_labels = len(self.labels)
        self.output_dir = self.config.get('output_dir', 'models/router')
        
        # Label mappings
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Initialize model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize classification model."""
        model_name = self.config.get('model', 'klue/roberta-base')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels
        ).to(self.device)
    
    def preprocess_function(self, examples):
        """
        Preprocess examples for training.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples with labels
        """
        # Tokenize texts
        result = self.tokenizer(
            examples['text'],
            padding='max_length',
            max_length=512,
            truncation=True
        )
        
        # Add labels
        if 'label_id' in examples:
            result['labels'] = examples['label_id']
        
        return result
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Train router model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ['labels']]
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=[col for col in eval_dataset.column_names if col not in ['labels']]
            )
        
        # Training arguments
        training_config = self.config.get('training', {})
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=training_config.get('num_epochs', 5),
            per_device_train_batch_size=training_config.get('batch_size', 16),
            learning_rate=training_config.get('learning_rate', 3e-5),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            weight_decay=training_config.get('weight_decay', 0.01),
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="accuracy" if eval_dataset else None,
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("Training domain router...")
        trainer.train()
        
        # Save final model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save label mappings
        import json
        label_config = {
            'labels': self.labels,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }
        with open(os.path.join(self.output_dir, 'label_config.json'), 'w') as f:
            json.dump(label_config, f, ensure_ascii=False, indent=2)
        
        print(f"Router model saved to {self.output_dir}")
