
import json
from pathlib import Path

CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

targets = {
    "Beatil 4mg/5mg",
    "Medsolu 16 mg",
    "Nexium 10mg dạng cốm pha hỗn dịch uống",
}

keywords = (
    "liều",
    "cách dùng",
    "tương tác",
    "chống chỉ định",
)

with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        metadata = chunk.get("metadata", {})
        drug_name = metadata.get("drug_name", "")
        section = metadata.get("section", "")
        content = chunk.get("text", "")

        if drug_name not in targets:
            continue

        if any(k in section.lower() for k in keywords):
            print("=" * 80)
            print(f"Drug: {drug_name}")
            print(f"Section: {section}")
            print(f"Chunk index: {metadata.get('chunk_index')}")
            print(f"URL: {metadata.get('source_url')}")
            print(content[:1200])
