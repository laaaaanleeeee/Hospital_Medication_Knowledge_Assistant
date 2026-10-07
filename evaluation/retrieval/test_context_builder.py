import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.rag.context_builder import ContextBuilder


def main():

    sample_results = [
        {
            "text": (
                "Loét dạ dày, quá mẫn cảm với hoạt chất. "
                "Chống chỉ định cho bệnh nhân..."
            ),
            "metadata": {
                "drug_name": "Voltaren 75mg/3ml Novartis",
                "section": "Chống chỉ định",
                "source_url": (
                    "https://thuocbietduoc.com.vn/"
                    "thuoc-10030/voltaren.aspx"
                )
            }
        },
        {
            "text": (
                "Trường hợp ngoại lệ: Phát ban có bọng nước..."
            ),
            "metadata": {
                "drug_name": "Voltaren 75mg/3ml Novartis",
                "section": "Tác dụng ngoài ý muốn",
                "source_url": (
                    "https://thuocbietduoc.com.vn/"
                    "thuoc-10030/voltaren.aspx"
                )
            }
        }
    ]

    builder = ContextBuilder()

    context = builder.build(
        sample_results
    )

    print("=" * 70)
    print("CONTEXT BUILDER TEST")
    print("=" * 70)

    print(context)

    assert "[Document 1]" in context
    assert "Voltaren 75mg/3ml Novartis" in context
    assert "Chống chỉ định" in context
    assert "Source:" in context

    print("\nSTATUS: PASSED")


if __name__ == "__main__":
    main()