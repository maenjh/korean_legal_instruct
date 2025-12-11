#!/usr/bin/env python3
"""
Example: Test domain router predictions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.router import DomainRouter


def main():
    # Initialize router
    print("Initializing domain router...")
    router = DomainRouter()
    
    # Try to load trained model
    router_path = "models/router"
    if os.path.exists(router_path):
        router.load_checkpoint(router_path)
        print(f"Loaded trained router from {router_path}\n")
    else:
        print("Using untrained router (for demonstration)\n")
    
    # Test queries
    test_queries = [
        "계약서 작성 시 주의사항은 무엇인가요?",
        "사기죄의 구성요건은 무엇인가요?",
        "특허권 침해 소송을 제기하려면 어떻게 해야 하나요?",
        "행정심판을 청구할 수 있는 기간은?",
        "손해배상 청구권의 소멸시효는?",
        "형사고소를 하려면 어떤 절차가 필요한가요?",
        "저작권 등록은 필수인가요?",
        "공무원의 부당한 행정처분에 대해 어떻게 대응하나요?"
    ]
    
    print("="*60)
    print("도메인 라우터 테스트")
    print("="*60)
    
    for query in test_queries:
        print(f"\n질문: {query}")
        
        result = router.predict(query, return_probabilities=True)
        
        print(f"예측 도메인: {result['domain_kr']} (신뢰도: {result['confidence']:.2%})")
        print("확률 분포:")
        for domain, prob in result['probabilities'].items():
            domain_kr = router.label_kr.get(domain, domain)
            bar = "█" * int(prob * 30)
            print(f"  {domain_kr:12s} {bar} {prob:.2%}")


if __name__ == "__main__":
    main()
