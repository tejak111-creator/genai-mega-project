from __future__ import annotations

from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class HFEmbeddingProvider:
    """
    Native HuggingFace embedding provider using AutoTokenizer + AutoModel.
    """
    def __init__(self, model_name:str = " sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return [emb.numpy() for emb in embeddings]
        