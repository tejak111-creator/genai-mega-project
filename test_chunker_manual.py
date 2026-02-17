from app.rag.chunker import chunk_document

text = "hello world " * 200

chunks= chunk_document(text, doc_id="sample.txt")

print("Num chunks:", len(chunks))
print(chunks[0])
print(chunks[0].text[:50])
