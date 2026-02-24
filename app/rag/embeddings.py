from __future__ import annotations
from typing import List
import numpy as np
from app.core.config import settings
from app.rag.hf_embeddings import HFEmbeddingProvider
from sentence_transformers import SentenceTransformer


#EmbeddingProvider will be an interface
#We separate Pipeline Logic and Model Logic
class EmbeddingProvider:
    #Implementations may use OpenAI,local models, etc.
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError
    

class DummyEmbeddingProvider(EmbeddingProvider):
    """
    Temporary embedding provider for development/testing
    Returns random vectors.
    """
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        EMBEDDING_DIM = 8
        return [np.random.rand(EMBEDDING_DIM).astype(np.float32) for _ in texts]
    

class SentenceTransformerProvider(EmbeddingProvider):
    """ 
    Real embedding provider using SentenceTransformers.
    Model: all-MiniLM-L6-v2
    
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        sample = self.model.encode(["dimension probe"], convert_to_numpy = True, normalize_embeddings = True)
        self.embedding_dim = int(sample.shape[1])
        #Loads embedding model only once, since repeated is slow
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("all items in texts must be strings")
        
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
       #vectors = vectors.astype(np.float32)

        return [vectors[i] for i in range(len(vectors))]
    
def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "sentence":
        return SentenceTransformerProvider(settings.embedding_model)
    if settings.embedding_provider == "hf":
        return HFEmbeddingProvider(settings.embedding_model)
    
    raise ValueError("Unsupported embedding provider")


