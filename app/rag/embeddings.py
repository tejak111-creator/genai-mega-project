from typing import List
import numpy as np

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
        return [np.random.rand(EMBEDDING_DIM) for _ in texts]