#We attach the metadata to chunks

from dataclasses import dataclass

# frozen = True, makes the dataclass immutable( read-only) after creation
#once the object is created, you cannot change its fields
@dataclass(frozen=True)
class DocumentChunk:
    text: str
    doc_id: str
    chunk_id: int
    start_char: int
    end_char: int

#Modify chunker to return structured objects instead of strings

