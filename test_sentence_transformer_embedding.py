from app.rag.embeddings import SentenceTransformerProvider

p = SentenceTransformerProvider()
vecs = p.embed(["hello world", "machine learning"])
print(len(vecs))
print(vecs[0].shape)
print(p.embedding_dim)