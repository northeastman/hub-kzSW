from __future__ import annotations

from logging import Logger

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from knowledge_assistant.core.settings import ModelSettings


class QueryTypeClassifier:
    labels = {0: "general", 1: "domain"}

    def __init__(self, settings: ModelSettings, logger: Logger):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(str(settings.classifier_base_model_path))
        self.model = BertForSequenceClassification.from_pretrained(str(settings.classifier_model_path), num_labels=2)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Query classifier loaded on {self.device}")

    def classify(self, query: str) -> str:
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = int(torch.argmax(outputs.logits, dim=1).item())
        return self.labels.get(prediction, "general")

