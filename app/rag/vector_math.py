import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


v1 = np.random.rand(5)
v2 = np.random.rand(5)
print(cosine_similarity(v1,v2))
