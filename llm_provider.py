
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
CHAT_MODEL = "nex-agi/nex-n2-pro:free"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

class LLMProvider:
    def __init__(self, provider_name):
        self.provider_name = provider_name
        if provider_name == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("key"),
            )

    def create_embedding(self, text):
        if self.provider_name == "openrouter":
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            embedding = response.data[0].embedding
    
            return embedding
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")


    def chat(self,user_message,details):
        # import prompt.text using dotenv
        prompt_file = Path(__file__).parent / "prompt.txt"
        prompt = prompt_file.read_text()
        # replace {details} in prompt with details
        prompt = prompt.replace("{details}", details)
        prompt = prompt.replace("{user_message}", user_message)
        if self.provider_name == "openrouter":
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": prompt},
                {"role": "user", "content": user_message}],
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
