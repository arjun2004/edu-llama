import streamlit as st
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
        """Send a chat completion request to OpenRouter"""
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
        """Send a simple prompt and get the response text using Llama 3.1 8B"""
        messages = [{"role": "user", "content": prompt}]
        result = self.chat_completion(messages, "meta-llama/llama-3.1-8b-instruct:free")
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "Error: Unexpected response format"
    
    def extract_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF bytes"""
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
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> str:
        """Load and store PDF content from bytes for future queries"""
        self.pdf_content = self.extract_pdf_from_bytes(pdf_bytes)
        
        if self.pdf_content.startswith("Error"):
            return self.pdf_content
        
        return f"PDF '{filename}' loaded successfully. Content length: {len(self.pdf_content)} characters"
    
    def summarize_pdf(self, custom_prompt: str = None) -> str:
        """Generate a summary of the loaded PDF"""
        if not self.pdf_content:
            return "Error: No PDF content loaded. Please upload a PDF first."
        
        # Prepare summary prompt
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nPDF Content:\n{self.pdf_content}"
        else:
            prompt = f"Please provide a comprehensive summary of the following PDF content. Include the main topics, key points, and important details:\n\n{self.pdf_content}"
        
        # Handle long content by truncating if necessary
        max_content_length = 12000
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            if custom_prompt:
                prompt = f"{custom_prompt}\n\nPDF Content (truncated):\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
            else:
                prompt = f"Please provide a comprehensive summary of the following PDF content (truncated due to length). Include the main topics, key points, and important details:\n\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
        
        return self.simple_prompt(prompt)
    
    def ask_pdf_question(self, question: str) -> str:
        """Ask a question about the loaded PDF content"""
        if not self.pdf_content:
            return "Error: No PDF content loaded. Please upload a PDF first."
        
        # Prepare question prompt
        prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content:\n{self.pdf_content}\n\nIf the answer is not found in the PDF content, please say so clearly."
        
        # Handle long content by truncating if necessary
        max_content_length = 12000
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            prompt = f"Based on the following PDF content (truncated), please answer this question: {question}\n\nPDF Content:\n{truncated_content}\n\n[Note: Content was truncated due to length limits]\n\nIf the answer is not found in the available PDF content, please say so clearly."
        
        return self.simple_prompt(prompt)
    
    def clear_pdf_content(self):
        """Clear the stored PDF content"""
        self.pdf_content = ""
        return "PDF content cleared from memory."

# Streamlit UI
def main():
    st.set_page_config(
        page_title="OpenRouter PDF Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 OpenRouter PDF Analyzer")
    st.markdown("Upload a PDF and analyze it using AI models from OpenRouter")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key",
            placeholder="sk-or-v1-..."
        )
        
        # Model selection
        model_options = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.1-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        
        selected_model = st.selectbox(
            "AI Model",
            model_options,
            help="Choose the AI model for analysis"
        )
        
        # Temperature setting
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses (0 = focused, 1 = creative)"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Enter your OpenRouter API key
        2. Upload a PDF file
        3. Get an AI summary
        4. Ask questions about the content
        """)
    
    # Initialize session state
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = ""
    
    # Check if API key is provided
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to continue.")
        st.info("💡 You can get a free API key from [OpenRouter](https://openrouter.ai/)")
        return
    
    # Initialize client
    if st.session_state.client is None or st.session_state.client.api_key != api_key:
        st.session_state.client = OpenRouterClient(api_key)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to analyze"
        )
        
        if uploaded_file is not None:
            # Load PDF
            if st.button("📂 Load PDF", type="primary"):
                with st.spinner("Loading PDF..."):
                    pdf_bytes = uploaded_file.read()
                    result = st.session_state.client.load_pdf_from_bytes(
                        pdf_bytes, 
                        uploaded_file.name
                    )
                    
                    if result.startswith("Error"):
                        st.error(result)
                        st.session_state.pdf_loaded = False
                    else:
                        st.success(result)
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_filename = uploaded_file.name
            
            # Show PDF status
            if st.session_state.pdf_loaded:
                st.success(f"✅ PDF loaded: {st.session_state.pdf_filename}")
                
                # Clear PDF button
                if st.button("🗑️ Clear PDF"):
                    st.session_state.client.clear_pdf_content()
                    st.session_state.pdf_loaded = False
                    st.session_state.pdf_filename = ""
                    st.success("PDF content cleared!")
                    st.rerun()
    
    with col2:
        st.header("🤖 General Chat")
        
        # General chat input
        general_prompt = st.text_area(
            "Ask anything (not related to PDF)",
            placeholder="e.g., Explain quantum physics...",
            height=100
        )
        
        if st.button("💬 Send Chat", type="secondary"):
            if general_prompt:
                with st.spinner("Getting response..."):
                    response = st.session_state.client.simple_prompt(general_prompt)
                    st.write("**Response:**")
                    st.write(response)
            else:
                st.warning("Please enter a prompt first.")
    
    # PDF Analysis Section
    if st.session_state.pdf_loaded:
        st.markdown("---")
        st.header("📋 PDF Analysis")
        
        # Create tabs for different analysis options
        tab1, tab2, tab3 = st.tabs(["📄 Summary", "❓ Q&A", "🎯 Custom Analysis"])
        
        with tab1:
            st.subheader("Document Summary")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                summary_type = st.selectbox(
                    "Summary Type",
                    ["Comprehensive", "Key Points", "Executive Summary", "Technical Details"]
                )
            
            with col2:
                if st.button("📋 Generate Summary", key="summary_btn"):
                    with st.spinner("Generating summary..."):
                        if summary_type == "Comprehensive":
                            summary = st.session_state.client.summarize_pdf()
                        elif summary_type == "Key Points":
                            summary = st.session_state.client.summarize_pdf(
                                "Please extract and list the key points from this document in bullet format."
                            )
                        elif summary_type == "Executive Summary":
                            summary = st.session_state.client.summarize_pdf(
                                "Please provide an executive summary suitable for business leaders."
                            )
                        else:  # Technical Details
                            summary = st.session_state.client.summarize_pdf(
                                "Please focus on technical details, specifications, and methodologies in this document."
                            )
                        
                        st.markdown("### Summary")
                        st.write(summary)
        
        with tab2:
            st.subheader("Ask Questions")
            
            # Predefined questions
            st.markdown("**Quick Questions:**")
            quick_questions = [
                "What are the main topics covered?",
                "What are the key conclusions?",
                "Are there any important numbers or statistics?",
                "What recommendations are made?",
                "Who are the key people mentioned?"
            ]
            
            selected_quick = st.selectbox(
                "Choose a quick question",
                [""] + quick_questions
            )
            
            if st.button("🚀 Ask Quick Question") and selected_quick:
                with st.spinner("Getting answer..."):
                    answer = st.session_state.client.ask_pdf_question(selected_quick)
                    st.markdown("### Answer")
                    st.write(answer)
            
            st.markdown("---")
            
            # Custom question
            custom_question = st.text_area(
                "Or ask your own question:",
                placeholder="e.g., What methodology was used in this research?",
                height=80
            )
            
            if st.button("❓ Ask Custom Question"):
                if custom_question:
                    with st.spinner("Getting answer..."):
                        answer = st.session_state.client.ask_pdf_question(custom_question)
                        st.markdown("### Answer")
                        st.write(answer)
                else:
                    st.warning("Please enter a question first.")
        
        with tab3:
            st.subheader("Custom Analysis")
            
            custom_prompt = st.text_area(
                "Custom analysis prompt:",
                placeholder="e.g., Analyze this document from a marketing perspective and identify opportunities...",
                height=100
            )
            
            if st.button("🎯 Run Custom Analysis"):
                if custom_prompt:
                    with st.spinner("Running analysis..."):
                        result = st.session_state.client.summarize_pdf(custom_prompt)
                        st.markdown("### Analysis Result")
                        st.write(result)
                else:
                    st.warning("Please enter a custom prompt first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** This app uses free models from OpenRouter. "
        "For better performance, consider upgrading to premium models."
    )

if __name__ == "__main__":
    main()