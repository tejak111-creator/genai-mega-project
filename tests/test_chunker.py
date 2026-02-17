from app.rag.chunker import chunk_document

def test_chunk_document():
    text = "hello world " * 200
    chunks = chunk_document(text,doc_id="doc1")

    assert len(chunks)>0
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].chunk_id == 0
    assert isinstance(chunks[0].text, str)
    