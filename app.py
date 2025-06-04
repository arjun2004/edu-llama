import requests
import json
from typing import List, Dict, Optional
import PyPDF2
import io
from pathlib import Path

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Store PDF content for follow-up questions
        self.pdf_content = ""
    
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
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text_content = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def extract_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text content from PDF bytes
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text content
        """
        try:
            text_content = ""
            
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and store PDF content for future queries
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Success message or error
        """
        if not Path(pdf_path).exists():
            return f"Error: PDF file not found at {pdf_path}"
        
        self.pdf_content = self.extract_pdf_text(pdf_path)
        
        if self.pdf_content.startswith("Error"):
            return self.pdf_content
        
        return f"PDF loaded successfully. Content length: {len(self.pdf_content)} characters"
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> str:
        """
        Load and store PDF content from bytes for future queries
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Optional filename for reference
            
        Returns:
            Success message or error
        """
        self.pdf_content = self.extract_pdf_from_bytes(pdf_bytes)
        
        if self.pdf_content.startswith("Error"):
            return self.pdf_content
        
        return f"PDF '{filename}' loaded successfully. Content length: {len(self.pdf_content)} characters"
    
    def summarize_pdf(self, pdf_path: str = None, custom_prompt: str = None) -> str:
        """
        Generate a summary of the loaded PDF or a specific PDF
        
        Args:
            pdf_path: Optional path to PDF (if not using pre-loaded content)
            custom_prompt: Optional custom summary instructions
            
        Returns:
            PDF summary or error message
        """
        # Load PDF if path provided and no content stored
        if pdf_path and not self.pdf_content:
            load_result = self.load_pdf(pdf_path)
            if load_result.startswith("Error"):
                return load_result
        
        # Check if we have PDF content
        if not self.pdf_content:
            return "Error: No PDF content loaded. Please load a PDF first using load_pdf() method."
        
        # Prepare summary prompt
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nPDF Content:\n{self.pdf_content}"
        else:
            prompt = f"Please provide a comprehensive summary of the following PDF content. Include the main topics, key points, and important details:\n\n{self.pdf_content}"
        
        # Handle long content by truncating if necessary
        max_content_length = 12000  # Leave room for prompt and response
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            if custom_prompt:
                prompt = f"{custom_prompt}\n\nPDF Content (truncated):\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
            else:
                prompt = f"Please provide a comprehensive summary of the following PDF content (truncated due to length). Include the main topics, key points, and important details:\n\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
        
        return self.simple_prompt(prompt)
    
    def ask_pdf_question(self, question: str, pdf_path: str = None) -> str:
        """
        Ask a question about the loaded PDF content
        
        Args:
            question: Question to ask about the PDF
            pdf_path: Optional path to PDF (if not using pre-loaded content)
            
        Returns:
            Answer based on PDF content or error message
        """
        # Load PDF if path provided and no content stored
        if pdf_path and not self.pdf_content:
            load_result = self.load_pdf(pdf_path)
            if load_result.startswith("Error"):
                return load_result
        
        # Check if we have PDF content
        if not self.pdf_content:
            return "Error: No PDF content loaded. Please load a PDF first using load_pdf() method."
        
        # Prepare question prompt
        prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content:\n{self.pdf_content}\n\nIf the answer is not found in the PDF content, please say so clearly."
        
        # Handle long content by truncating if necessary
        max_content_length = 12000  # Leave room for prompt and response
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            prompt = f"Based on the following PDF content (truncated), please answer this question: {question}\n\nPDF Content:\n{truncated_content}\n\n[Note: Content was truncated due to length limits]\n\nIf the answer is not found in the available PDF content, please say so clearly."
        
        return self.simple_prompt(prompt)
    
    def clear_pdf_content(self):
        """Clear the stored PDF content"""
        self.pdf_content = ""
        return "PDF content cleared from memory."

# Usage Example
if __name__ == "__main__":
    # Initialize client with explicit API key
    api_key = "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"
    client = OpenRouterClient(api_key)
    
    # Original functionality still works
    # response = client.simple_prompt("do you know mar baselios college tvm?")
    # print("Response:", response)
    
    # PDF FUNCTIONALITY EXAMPLE - UNCOMMENT AND MODIFY PATH:
    
    # STEP 1: Put your PDF file path here
    pdf_path = "C:/Users/arjun/edu-llama/intel.pdf"  # Windows example
    # pdf_path = "/home/username/Documents/sample.pdf"      # Linux example
    # pdf_path = "/Users/username/Documents/sample.pdf"     # Mac example
    
    # STEP 2: Uncomment the lines below to test PDF functionality

    print(f"\nTrying to load PDF: {pdf_path}")
    load_result = client.load_pdf(pdf_path)
    print("Load result:", load_result)
    
    if not load_result.startswith("Error"):
        # Get summary
        print("\nGenerating PDF summary...")
        summary = client.summarize_pdf()
        print("\nPDF SUMMARY:")
        print("-" * 40)
        print(summary)
        
        # Ask questions about the PDF
        print("\n" + "="*50)
        print("ASKING QUESTIONS ABOUT THE PDF:")
        print("="*50)
        
        question1 = "What are the main topics covered in this document?"
        print(f"\nQuestion 1: {question1}")
        answer1 = client.ask_pdf_question(question1)
        print(f"Answer: {answer1}")
        
        question2 = "What are the key conclusions or recommendations?"
        print(f"\nQuestion 2: {question2}")
        answer2 = client.ask_pdf_question(question2)
        print(f"Answer: {answer2}")
        
        question3 = "Can you extract any important numbers or statistics?"
        print(f"\nQuestion 3: {question3}")
        answer3 = client.ask_pdf_question(question3)
        print(f"Answer: {answer3}")
    else:
        print("Failed to load PDF. Please check the file path.")

    
  