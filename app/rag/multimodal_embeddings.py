from __future__ import annotations

from typing import List
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbeddingProvider:
    """
    Multimodal embedding provider using OpenAI CLIP.
    Supports:
    - Text embeddings
    - Image embeddings
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_text(self, texts: List[str]) -> List[np.ndarray]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        embeddings = outputs.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return [embeddings[i].astype(np.float32) for i in range(len(embeddings))]

    def embed_images(self, image_paths: List[str]) -> List[np.ndarray]:
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        embeddings = outputs.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return [embeddings[i].astype(np.float32) for i in range(len(embeddings))]