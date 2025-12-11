"""
Data loading and preprocessing utilities for Korean legal instruction data.
"""
import json
import os
from typing import Dict, List, Tuple
from datasets import Dataset, DatasetDict
import pandas as pd


class LegalDataLoader:
    """Load and preprocess Korean legal instruction data from AIHub."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing raw legal data
        """
        self.data_dir = data_dir
        self.domains = ["civil", "criminal", "ip", "administrative"]
        self.domain_mapping = {
            "civil": "민사",
            "criminal": "형사", 
            "ip": "지식재산권",
            "administrative": "행정"
        }
    
    def load_domain_data(self, domain: str) -> List[Dict]:
        """
        Load data for a specific legal domain.
        
        Args:
            domain: Legal domain (civil, criminal, ip, administrative)
            
        Returns:
            List of instruction-response pairs
        """
        domain_dir = os.path.join(self.data_dir, domain)
        data = []
        
        # Check if domain directory exists
        if not os.path.exists(domain_dir):
            print(f"Warning: {domain_dir} does not exist. Generating sample data.")
            return self._generate_sample_data(domain)
        
        # Load JSON files from domain directory
        for filename in os.listdir(domain_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(domain_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    domain_data = json.load(f)
                    if isinstance(domain_data, list):
                        data.extend(domain_data)
                    else:
                        data.append(domain_data)
        
        return data
    
    def _generate_sample_data(self, domain: str, num_samples: int = 100) -> List[Dict]:
        """
        Generate sample data for demonstration purposes.
        
        Args:
            domain: Legal domain
            num_samples: Number of samples to generate
            
        Returns:
            List of sample instruction-response pairs
        """
        domain_kr = self.domain_mapping.get(domain, domain)
        samples = []
        
        # Domain-specific sample templates
        templates = {
            "civil": [
                ("계약서 작성 시 주의사항은?", "계약서 작성 시에는 당사자의 권리와 의무를 명확히 기재해야 합니다."),
                ("손해배상청구의 요건은?", "손해배상청구는 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 경우 성립합니다."),
            ],
            "criminal": [
                ("형사처벌의 종류는?", "형사처벌에는 징역, 금고, 벌금, 구류, 과료, 몰수 등이 있습니다."),
                ("정당방위의 요건은?", "정당방위는 자기 또는 타인의 법익에 대한 현재의 부당한 침해를 방위하기 위한 행위입니다."),
            ],
            "ip": [
                ("특허권의 존속기간은?", "특허권은 설정등록한 날부터 특허출원일 후 20년이 되는 날까지 존속합니다."),
                ("저작권 침해의 요건은?", "저작권 침해는 저작권자의 허락 없이 저작물을 복제, 배포, 공연, 전시하는 경우 성립합니다."),
            ],
            "administrative": [
                ("행정처분 불복방법은?", "행정처분에 불복하는 경우 행정심판 또는 행정소송을 제기할 수 있습니다."),
                ("행정청의 의무는?", "행정청은 법령에 따라 공정하고 투명하게 직무를 수행해야 합니다."),
            ]
        }
        
        domain_templates = templates.get(domain, templates["civil"])
        
        for i in range(num_samples):
            template_idx = i % len(domain_templates)
            instruction, response = domain_templates[template_idx]
            
            samples.append({
                "instruction": f"[{domain_kr}] {instruction}",
                "response": response,
                "domain": domain
            })
        
        return samples
    
    def prepare_dataset(self, domain: str, train_split: float = 0.9) -> DatasetDict:
        """
        Prepare training and validation datasets for a domain.
        
        Args:
            domain: Legal domain
            train_split: Proportion of data for training
            
        Returns:
            DatasetDict with train and validation splits
        """
        data = self.load_domain_data(domain)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * train_split)
        
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        # Convert to datasets
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df)
        })
        
        return dataset_dict
    
    def prepare_router_dataset(self, train_split: float = 0.9) -> DatasetDict:
        """
        Prepare dataset for training the domain router.
        
        Args:
            train_split: Proportion of data for training
            
        Returns:
            DatasetDict with train and validation splits containing all domains
        """
        all_data = []
        
        for domain in self.domains:
            domain_data = self.load_domain_data(domain)
            for item in domain_data:
                all_data.append({
                    "text": item.get("instruction", ""),
                    "label": domain,
                    "domain": domain
                })
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create label encoding
        label_to_id = {label: idx for idx, label in enumerate(self.domains)}
        df['label_id'] = df['label'].map(label_to_id)
        
        # Split
        split_idx = int(len(df) * train_split)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df)
        })
        
        return dataset_dict


def format_instruction(instruction: str, response: str = None) -> str:
    """
    Format instruction-response pair for training.
    
    Args:
        instruction: User instruction
        response: Model response (optional)
        
    Returns:
        Formatted prompt string
    """
    if response:
        return f"### 질문:\n{instruction}\n\n### 답변:\n{response}"
    else:
        return f"### 질문:\n{instruction}\n\n### 답변:\n"
