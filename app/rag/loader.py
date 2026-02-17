from pathlib import Path

#Document Loader, we take file path, read file, return text
def load_text_files(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    
    return p.read_text(encoding="utf-8")

result = load_text_files("data/sample.txt")
print(result)
