from Mistral import HuggingFaceLLM
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize LLM (only once)
llm = HuggingFaceLLM(
        api_key = os.getenv("HUGGINGFACE_API_KEY")
)

def generate(prompt):
    return llm.generate(prompt)
