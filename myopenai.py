import os
from venv import logger
import warnings
from openai import BadRequestError, OpenAI
from dotenv import load_dotenv
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse


class MyOpenAI:

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512):
        """
        Initializes the GhvOpenAI class, setting the model and dimensions.

        Args:
            model (str): The embedding model to use.
            dimensions (int): The number of dimensions for the embeddings.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.user = os.getenv("OPENAI_USER", "default_user")
        self.completions_model = os.getenv(
            "OPENAI_COMPLETIONS_MODEL", "o3-mini")
        self.client = OpenAI(api_key=self.api_key)
        # Exponential backoff delays in seconds
        self.backoff_delays = [1, 2, 4, 8, 16]

    def sendPrompt(self, prompt: str, model_override: str = None) -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.

        Returns:
            str: The completion result as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=(
                    self.completions_model if model_override is None else model_override
                ),
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract the response from the completion
            completion = response.choices[0].message.content

            # If the content seems truncated, send a follow-up request or handle continuation
            while response.choices[0].finish_reason == "length":
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {"role": "system", "content": "Continue."},
                    ],
                )
                completion += response.choices[0].message.content
            return completion

        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise
