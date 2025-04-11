from huggingface_hub import InferenceClient
import requests

class HuggingFaceLLM:
    def __init__(self, api_key):
        self.client = InferenceClient(token=api_key)

    def generate(self, prompt):
        try:
            response = self.client.text_generation(
                model="mistralai/Mistral-7B-Instruct-v0.1",
                prompt=f"write an eassy on {prompt}",
                max_new_tokens=500,
                temperature=0.7,
            )
            return response
        except requests.exceptions.HTTPError as e:
            return f"HTTP error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
