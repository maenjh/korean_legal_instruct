"""
Inference pipeline integrating router and LoRA models.
"""
import os
from typing import Dict, Optional
import torch

from ..models.router import DomainRouter
from ..models.lora_model import LoRAModel
from ..data.data_loader import format_instruction


class LegalAssistant:
    """
    Legal assistant that routes queries to domain-specific LoRA models.
    """
    
    def __init__(
        self,
        router_path: str,
        lora_base_path: str,
        base_model_name: str = "beomi/KoAlpaca-Polyglot-5.8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize legal assistant.
        
        Args:
            router_path: Path to trained router model
            lora_base_path: Base path to LoRA models directory
            base_model_name: Name of base language model
            device: Device to run models on
        """
        self.device = device
        self.lora_base_path = lora_base_path
        self.base_model_name = base_model_name
        
        # Initialize router
        print("Loading domain router...")
        self.router = DomainRouter(device=device)
        if os.path.exists(router_path):
            self.router.load_checkpoint(router_path)
        
        # Domain-specific LoRA models (lazy loading)
        self.lora_models = {}
        self.domains = ["civil", "criminal", "ip", "administrative"]
    
    def _load_lora_model(self, domain: str) -> LoRAModel:
        """
        Load LoRA model for specific domain (lazy loading).
        
        Args:
            domain: Legal domain
            
        Returns:
            LoRA model for the domain
        """
        if domain not in self.lora_models:
            lora_path = os.path.join(self.lora_base_path, domain)
            
            if not os.path.exists(lora_path):
                print(f"Warning: LoRA model for {domain} not found at {lora_path}")
                print(f"Using base model without LoRA adaptation.")
                # Load base model without LoRA
                model = LoRAModel(self.base_model_name, device=self.device)
            else:
                print(f"Loading {domain} LoRA model...")
                model = LoRAModel(self.base_model_name, device=self.device)
                model.load_lora_weights(lora_path)
            
            self.lora_models[domain] = model
        
        return self.lora_models[domain]
    
    def route_query(self, query: str) -> Dict:
        """
        Route query to appropriate domain.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with routing information
        """
        return self.router.predict(query, return_probabilities=True)
    
    def generate_response(
        self,
        query: str,
        domain: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict:
        """
        Generate response for query.
        
        Args:
            query: User query
            domain: Specific domain (optional, will be auto-detected if None)
            max_length: Maximum response length
            temperature: Generation temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Route query if domain not specified
        if domain is None:
            routing_info = self.route_query(query)
            domain = routing_info['domain']
            confidence = routing_info['confidence']
        else:
            routing_info = None
            confidence = 1.0
        
        # Load appropriate LoRA model
        lora_model = self._load_lora_model(domain)
        
        # Format prompt
        prompt = format_instruction(query)
        
        # Generate response
        response = lora_model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Return result
        result = {
            "query": query,
            "domain": domain,
            "domain_kr": self.router.label_kr.get(domain, domain),
            "confidence": confidence,
            "response": response
        }
        
        if routing_info:
            result["routing_info"] = routing_info
        
        return result
    
    def generate_interactive(self):
        """
        Interactive mode for continuous Q&A.
        """
        print("\n" + "="*60)
        print("한국 법률 상담 시스템 (Korean Legal Assistant)")
        print("="*60)
        print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
        
        while True:
            try:
                query = input("\n질문: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print("\n상담을 종료합니다.")
                    break
                
                if not query:
                    continue
                
                # Generate response
                result = self.generate_response(query)
                
                # Display result
                print(f"\n도메인: {result['domain_kr']} (신뢰도: {result['confidence']:.2%})")
                print(f"\n답변: {result['response']}")
                
                if 'routing_info' in result and 'probabilities' in result['routing_info']:
                    print("\n[도메인 분류 확률]")
                    for domain, prob in result['routing_info']['probabilities'].items():
                        domain_kr = self.router.label_kr.get(domain, domain)
                        print(f"  {domain_kr}: {prob:.2%}")
            
            except KeyboardInterrupt:
                print("\n\n상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                continue
