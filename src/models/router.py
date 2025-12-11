"""
Domain router for classifying legal queries.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Tuple


class DomainRouter:
    """Domain classification model for routing legal queries."""
    
    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        num_labels: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize domain router.
        
        Args:
            model_name: Name or path of classification model
            num_labels: Number of legal domain classes
            device: Device to load model on
        """
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        # Domain labels
        self.labels = ["civil", "criminal", "ip", "administrative"]
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Korean labels for display
        self.label_kr = {
            "civil": "민사법",
            "criminal": "형사법",
            "ip": "지식재산권법",
            "administrative": "행정법"
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load trained router checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=self.num_labels
        ).to(self.device)
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        Predict legal domain for given text.
        
        Args:
            text: Input query text
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with predicted domain and optionally probabilities
        """
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = self.id_to_label[predicted_id]
        
        result = {
            "domain": predicted_label,
            "domain_kr": self.label_kr[predicted_label],
            "confidence": probabilities[0][predicted_id].item()
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.labels[i]: probabilities[0][i].item()
                for i in range(self.num_labels)
            }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict domains for batch of texts.
        
        Args:
            texts: List of input queries
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Process results
        results = []
        for i in range(len(texts)):
            predicted_id = torch.argmax(probabilities[i]).item()
            predicted_label = self.id_to_label[predicted_id]
            
            results.append({
                "domain": predicted_label,
                "domain_kr": self.label_kr[predicted_label],
                "confidence": probabilities[i][predicted_id].item()
            })
        
        return results
