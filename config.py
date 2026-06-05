from openai import OpenAI

MODEL = "qwen2.5-coder:7b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
