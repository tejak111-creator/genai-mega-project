from typing import List, Tuple
from app.rag.models import DocumentChunk

"""
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        #we keep some overlap between consecutive chunks
    return chunks


text = "hello world " *200
chunks = chunk_text(text)
print(len(chunks))
"""

#returns spans
def chunk_spans(text: str, chunk_size: int = 500, overlap: int =50) -> List[Tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size muts be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    
    spans = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start += step
    
    return spans

# Building structured chunks || Returns List of DocChunks
def chunk_document(text: str, doc_id: str, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
    spans = chunk_spans(text, chunk_size=chunk_size, overlap=overlap)
    chunks: List[DocumentChunk] = []

    for i, (start, end) in enumerate(spans):
        chunks.append(
            DocumentChunk(
                text=text[start:end],
                doc_id=doc_id,
                chunk_id=i,
                start_char=start,
                end_char=end,
            )
        )
        
    return chunks
