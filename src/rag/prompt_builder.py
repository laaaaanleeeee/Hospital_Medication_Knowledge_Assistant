class PromptBuilder:

    SYSTEM_PROMPT = """
Bạn là trợ lý tra cứu thông tin thuốc.

Nhiệm vụ:
- Trả lời câu hỏi dựa CHỈ trên CONTEXT được cung cấp.
- Không sử dụng kiến thức bên ngoài CONTEXT.
- Không tự suy đoán hoặc bổ sung thông tin không có trong CONTEXT.
- Nếu CONTEXT không chứa thông tin để trả lời, hãy nói:
  "Không tìm thấy thông tin này trong tài liệu được cung cấp."
- Trả lời bằng tiếng Việt.
- Trả lời ngắn gọn, rõ ràng và trực tiếp.
"""

    def build(
        self,
        context: str,
        question: str
    ) -> str:

        prompt = f"""
{self.SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        return prompt.strip()