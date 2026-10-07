import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiLLM:

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash"
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

    def generate(
        self,
        prompt: str
    ) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text