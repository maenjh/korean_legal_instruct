#!/usr/bin/env python3
"""
Script to train domain-specific LoRA models.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from src.training.lora_trainer import LoRATrainer


def main():
    parser = argparse.ArgumentParser(description="Train domain-specific LoRA model")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["civil", "criminal", "ip", "administrative"],
        help="Legal domain to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to domain config file"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data"
    )
    
    args = parser.parse_args()
    
    # Load datasets
    domain_data_dir = os.path.join(args.data_dir, args.domain)
    
    print(f"Loading {args.domain} dataset from {domain_data_dir}...")
    if os.path.exists(domain_data_dir):
        dataset = load_from_disk(domain_data_dir)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    else:
        print(f"Warning: Dataset not found at {domain_data_dir}")
        print("Please run prepare_data.py first or ensure data directory is correct.")
        return
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Initialize trainer
    print(f"\nInitializing LoRA trainer for {args.domain}...")
    trainer = LoRATrainer(
        config_path=args.config,
        base_config_path=args.base_config
    )
    
    # Train model
    print(f"\nStarting training for {args.domain} domain...")
    trainer.train(train_dataset, eval_dataset)
    
    print(f"\n{args.domain} LoRA model training complete!")


if __name__ == "__main__":
    main()
