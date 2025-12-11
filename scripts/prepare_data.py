#!/usr/bin/env python3
"""
Script to prepare data for training.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import LegalDataLoader


def main():
    parser = argparse.ArgumentParser(description="Prepare legal instruction data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = LegalDataLoader(args.data_dir)
    
    # Process each domain
    domains = ["civil", "criminal", "ip", "administrative"]
    
    for domain in domains:
        print(f"\nProcessing {domain} data...")
        
        # Prepare dataset
        dataset = data_loader.prepare_dataset(domain)
        
        # Save processed data
        domain_output_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        dataset.save_to_disk(domain_output_dir)
        
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['validation'])}")
        print(f"  Saved to: {domain_output_dir}")
    
    # Prepare router dataset
    print("\nPreparing router dataset...")
    router_dataset = data_loader.prepare_router_dataset()
    router_output_dir = os.path.join(args.output_dir, "router")
    os.makedirs(router_output_dir, exist_ok=True)
    router_dataset.save_to_disk(router_output_dir)
    
    print(f"  Train samples: {len(router_dataset['train'])}")
    print(f"  Validation samples: {len(router_dataset['validation'])}")
    print(f"  Saved to: {router_output_dir}")
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
