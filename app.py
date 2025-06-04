import requests
import json
from typing import List, Dict, Optional

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "meta-llama/llama-3.1-8b-instruct:free",
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict:
        """
        Send a chat completion request to OpenRouter
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (default is free Llama 3.1 8B)
            temperature: Randomness (0-1)
            max_tokens: Max response length
        
        Returns:
            API response as dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode failed: {str(e)}"}
    
    def simple_prompt(self, prompt: str) -> str:
        """
        Send a simple prompt and get the response text using Llama 3.1 8B
        
        Args:
            prompt: Your question/prompt
            
        Returns:
            Response text or error message
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.chat_completion(messages, "meta-llama/llama-3.1-8b-instruct:free")
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "Error: Unexpected response format"

# Usage Example
if __name__ == "__main__":
    # Initialize client with explicit API key
    api_key = "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"
    client = OpenRouterClient(api_key)
    
    # Single prompt example with Llama 3.1 8B
    response = client.simple_prompt("do you know mar baselios college tvm?")
    print("Response:", response)