import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.rag.prompt_builder import PromptBuilder


def main():

    print("=" * 70)
    print("PROMPT BUILDER TEST")
    print("=" * 70)

    context = """
[Document 1]
Thuốc: Voltaren 75mg/3ml Novartis
Phần: Chống chỉ định

Loét dạ dày, quá mẫn cảm với hoạt chất.
"""

    question = "Voltaren có những chống chỉ định nào?"

    builder = PromptBuilder()

    prompt = builder.build(
        context=context,
        question=question
    )

    print(prompt)

    assert "CONTEXT:" in prompt
    assert "QUESTION:" in prompt
    assert "Voltaren" in prompt
    assert question in prompt
    assert "Không sử dụng kiến thức bên ngoài CONTEXT." in prompt
    assert "Không tìm thấy thông tin này trong tài liệu được cung cấp." in prompt

    print("\nSTATUS: PASSED")


if __name__ == "__main__":
    main()