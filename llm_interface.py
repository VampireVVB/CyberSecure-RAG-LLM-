import requests
from config import config

class LLaMAGenerator:
    def __init__(self):
        self.api_url = f"{config.api_base_url}/chat/completions"

    def generate(self, prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "mistral",  
            "messages": [
                {"role": "system", "content": "You are a secure coding tutor. Answer using only the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            print("🚀 Sending prompt to LLaMA (Ollama)...")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=180)
            response_json = response.json()

            if "choices" not in response_json:
                print("⚠️ LLM API Error:", response_json)
                return "Error: LLM did not return a valid response."

            return response_json["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {e}"



