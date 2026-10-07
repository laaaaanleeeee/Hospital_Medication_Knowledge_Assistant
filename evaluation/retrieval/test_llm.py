import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.rag.llm import GeminiLLM


def main():

    print("=" * 70)
    print("GEMINI LLM TEST")
    print("=" * 70)

    context = """
[Document 1]
Thuốc: Voltaren 75mg/3ml Novartis
Phần: Chống chỉ định

Loét dạ dày, quá mẫn cảm với hoạt chất.
"""

    question = "Voltaren có giá bao nhiêu?"

    prompt = f"""
Bạn là trợ lý tra cứu thông tin thuốc.

QUY TẮC:
- Chỉ sử dụng thông tin trong CONTEXT.
- Không sử dụng kiến thức bên ngoài CONTEXT.
- Không tự suy đoán hoặc bổ sung thông tin không có trong CONTEXT.
- Nếu CONTEXT không chứa thông tin để trả lời,
  hãy nói:
  "Không tìm thấy thông tin này trong tài liệu được cung cấp."
- Trả lời bằng tiếng Việt.
- Trả lời ngắn gọn và trực tiếp.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = GeminiLLM()

    answer = llm.generate(prompt)

    print("\nANSWER:")
    print(answer)

    assert answer.strip()

    print("\nSTATUS: PASSED")


if __name__ == "__main__":
    main()