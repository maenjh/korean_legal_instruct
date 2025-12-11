import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

class DomainRouter:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = config.ROUTER_DEVICE
        self.load_model()

    def load_model(self):
        print(f"--- Loading Domain Router from {config.ROUTER_PATH} ---")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.ROUTER_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.ROUTER_PATH,
                num_labels=len(config.DOMAIN_MAP),
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
            )
            self.model.to(self.device)
            self.model.eval()
            print("--- Domain Router Loaded ---")
        except Exception as e:
            print(f"Error loading router: {e}")
            raise e

    def predict(self, text: str) -> str:
        """
        Predicts the legal domain for the given text.
        Returns the domain key (e.g., 'civil', 'criminal').
        """
        if not text:
            return "civil" # Default fallback

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_class_id = torch.argmax(logits, dim=1).item()
        domain_key = config.DOMAIN_MAP.get(predicted_class_id, "civil")
        
        print(f"[Router] Input: {text[:50]}... -> Predicted: {domain_key}")
        return domain_key
