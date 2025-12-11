#!/usr/bin/env python3
"""
Script to train domain router model.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from src.training.router_trainer import RouterTrainer


def main():
    parser = argparse.ArgumentParser(description="Train domain classification router")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_config.yaml",
        help="Path to router config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/router",
        help="Directory containing processed router data"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading router dataset from {args.data_dir}...")
    if os.path.exists(args.data_dir):
        dataset = load_from_disk(args.data_dir)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    else:
        print(f"Warning: Dataset not found at {args.data_dir}")
        print("Please run prepare_data.py first or ensure data directory is correct.")
        return
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Initialize trainer
    print("\nInitializing router trainer...")
    trainer = RouterTrainer(config_path=args.config)
    
    # Train model
    print("\nStarting router training...")
    trainer.train(train_dataset, eval_dataset)
    
    print("\nRouter model training complete!")


if __name__ == "__main__":
    main()
