#!/usr/bin/env python3
"""
MVP Demo: Korean Legal Assistant
Simple demonstration of the legal instruction system with router and LoRA models.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import LegalAssistant


def main():
    parser = argparse.ArgumentParser(description="Korean Legal Assistant MVP Demo")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Legal query to answer (if not provided, starts interactive mode)"
    )
    parser.add_argument(
        "--router_path",
        type=str,
        default="models/router",
        help="Path to trained router model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="models/lora",
        help="Base path to LoRA models directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="beomi/KoAlpaca-Polyglot-5.8B",
        help="Base language model name"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["civil", "criminal", "ip", "administrative"],
        help="Specific domain (optional, will auto-detect if not provided)"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*60)
    print("한국 법률 상담 시스템 MVP")
    print("Korean Legal Instruction System - MVP Demo")
    print("="*60)
    print()
    
    # Initialize assistant
    print("초기화 중... (Initializing...)")
    try:
        assistant = LegalAssistant(
            router_path=args.router_path,
            lora_base_path=args.lora_path,
            base_model_name=args.base_model
        )
        print("초기화 완료! (Initialization complete!)\n")
    except Exception as e:
        print(f"초기화 실패: {e}")
        print("\n참고: 모델이 학습되지 않은 경우 다음 명령어를 실행하세요:")
        print("  1. python scripts/prepare_data.py")
        print("  2. python scripts/train_router.py")
        print("  3. python scripts/train_lora.py --domain [civil|criminal|ip|administrative] --config configs/[domain]_config.yaml")
        return
    
    # Single query mode
    if args.query:
        print(f"질문: {args.query}\n")
        
        try:
            result = assistant.generate_response(
                args.query,
                domain=args.domain
            )
            
            print(f"도메인: {result['domain_kr']} (신뢰도: {result['confidence']:.2%})")
            print(f"\n답변:\n{result['response']}")
            
            if 'routing_info' in result and 'probabilities' in result['routing_info']:
                print("\n[도메인 분류 확률]")
                for domain, prob in result['routing_info']['probabilities'].items():
                    domain_kr = assistant.router.label_kr.get(domain, domain)
                    print(f"  {domain_kr}: {prob:.2%}")
        
        except Exception as e:
            print(f"오류 발생: {e}")
    
    # Interactive mode
    else:
        assistant.generate_interactive()


if __name__ == "__main__":
    main()
