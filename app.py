import streamlit as st
import requests
import json
from typing import List, Dict, Optional
import PyPDF2
import io
from pathlib import Path
import speech_recognition as sr
import pyttsx3
import threading
from io import BytesIO
import base64
import tempfile
import os
import platform
from datetime import datetime

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.is_speaking = False
        self.stop_speaking = False
        self.init_microphone()
        self.init_tts()
    
    def init_microphone(self):
        """Initialize microphone with error handling"""
        try:
            self.microphone = sr.Microphone()
            # Adjust microphone for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            st.warning(f"Microphone initialization warning: {e}")
            self.microphone = None
    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Set properties
            voices = self.tts_engine.getProperty('voices')
            if voices and len(voices) > 0:
                # Try to use a female voice if available, otherwise use first voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level
        except Exception as e:
            st.warning(f"TTS initialization warning: {e}")
            self.tts_engine = None
    
    def listen_for_speech(self, timeout=10, phrase_time_limit=10):
        """Listen for speech and convert to text"""
        if not self.microphone:
            return "Error: Microphone not available"
        
        try:
            with self.microphone as source:
                st.info("🎤 Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            st.info("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Error: Listening timeout - no speech detected"
        except sr.UnknownValueError:
            return "Error: Could not understand the speech"
        except sr.RequestError as e:
            return f"Error: Speech recognition service error - {e}"
        except Exception as e:
            return f"Error: {e}"
    
    def speak_text(self, text):
        """Convert text to speech with stop functionality"""
        if not self.tts_engine:
            return False
        
        try:
            self.is_speaking = True
            self.stop_speaking = False
            
            # Run TTS in a separate thread to avoid blocking
            def speak():
                try:
                    # Split text into sentences for better stop control
                    sentences = text.split('. ')
                    for sentence in sentences:
                        if self.stop_speaking:
                            break
                        if sentence.strip():
                            self.tts_engine.say(sentence + '.')
                            self.tts_engine.runAndWait()
                except Exception as e:
                    pass
                finally:
                    self.is_speaking = False
                    self.stop_speaking = False
            
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            return True
        except Exception as e:
            self.is_speaking = False
            st.error(f"TTS error: {e}")
            return False
    
    def stop_speech(self):
        """Stop the current speech"""
        self.stop_speaking = True
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
    
    def text_to_audio_file(self, text):
        """Convert text to audio file for download"""
        if not self.tts_engine:
            return None
        
        try:
            # Create temporary file with proper extension
            if platform.system() == "Windows":
                temp_path = tempfile.mktemp(suffix='.wav')
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    temp_path = tmp_file.name
            
            # Save speech to file
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # Read the file
            try:
                with open(temp_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
            except FileNotFoundError:
                return None
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors
            
            return audio_bytes
        except Exception as e:
            st.warning(f"Audio file creation warning: {e}")
            return None

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
        """
        Send a prompt to the model using LLaMA 3.1 with a learning assistant persona.
        Politely declines to answer off-topic or irrelevant questions.
        """

        # Optional: Early filter for clearly off-topic questions
        off_topic_keywords = [
            "joke", "celebrity", "politics", "game", "song", "weather", 
            "funny", "gossip", "sports", "movie", "actor", "meme"
        ]
        if any(keyword in prompt.lower() for keyword in off_topic_keywords):
            return "I'm here to help with educational topics. Could you ask something related to your studies?"

        # Full message setup for the learning assistant persona
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a dedicated and friendly AI learning assistant. "
                    "Your primary goal is to support students and educators by answering questions related to academic subjects, "
                    "study materials, and educational topics. You can explain concepts, summarize content, and offer guidance across "
                    "a range of disciplines like math, science, history, literature, and computer science.\n\n"
                    
                    "If a user asks something unrelated to learning—such as about entertainment, politics, or personal opinions—"
                    "you must politely decline by saying:\n"
                    "\"I'm here to help with educational topics. Could you ask something related to your studies?\"\n\n"

                    "Keep your tone clear, respectful, and student-friendly. Provide well-structured, fact-based answers. "
                    "When appropriate, include simple examples, analogies, or step-by-step explanations to aid understanding."
                )
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "system",
                "content": (
                    "Ensure that your response stays on topic, avoids speculation, and avoids humor or casual banter. "
                    "Use bullet points or short paragraphs for clarity when needed. Focus only on the educational value."
                )
            }
        ]

        # Call LLM
        result = self.chat_completion(messages, "meta-llama/llama-3.1-8b-instruct:free")

        # Handle error or missing output
        if not result or "error" in result:
            return f"Error: {result.get('error', 'Unknown error occurred')}"

        try:
            content = result["choices"][0]["message"]["content"].strip()
            return content if content else "Sorry, I couldn't generate a response. Please try rephrasing your question."
        except (KeyError, IndexError, TypeError):
            return "Error: Unexpected response format from the model."

    def extract_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    text_content.append(f"--- Page {page_num + 1} ---\nError extracting text: {e}")
            
            if not text_content:
                return "Error: No readable text found in the PDF"
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
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

def create_audio_download_link(audio_bytes, filename="response.wav"):
    """Create a download link for audio"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">📥 Download Audio</a>'
        return href
    return ""

def display_message(role, content, timestamp=None, show_audio=False, voice_handler=None):
    """Display a chat message with styling similar to ChatGPT"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    # Create columns for avatar and content
    if role == "user":
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.markdown("**👤**")
        with col2:
            st.markdown(f"**You** · {timestamp}")
            st.markdown(content)
    else:
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
        with col1:
            st.markdown("**🤖**")
        with col2:
            st.markdown(f"**AI Assistant** · {timestamp}")
            st.markdown(content)
        with col3:
            if show_audio and voice_handler and voice_handler.tts_engine:
                # Create unique keys for buttons
                speak_key = f"speak_{timestamp}_{hash(content)}"
                stop_key = f"stop_{timestamp}_{hash(content)}"
                
                if voice_handler.is_speaking:
                    if st.button("⏹️ Stop", key=stop_key, help="Stop speaking"):
                        voice_handler.stop_speech()
                        st.rerun()
                else:
                    if st.button("🔊 Play", key=speak_key, help="Speak response"):
                        voice_handler.speak_text(content)
                        st.rerun()

def main():
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for ChatGPT-like styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        border-top: 1px solid #e0e0e0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #10b981;
    }
    
    .status-offline {
        background-color: #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Learning Assistant</h1>
        <p>Upload PDFs and chat with AI using voice and text</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize voice handler
    if 'voice_handler' not in st.session_state:
        try:
            st.session_state.voice_handler = VoiceHandler()
        except Exception as e:
            st.warning(f"Voice initialization warning: {e}")
            st.session_state.voice_handler = None
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = ""
    if 'listening' not in st.session_state:
        st.session_state.listening = False
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = ""
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key",
            placeholder="sk-or-v1-..."
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required")
        
        st.markdown("---")
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        model_options = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.1-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        
        selected_model = st.selectbox("AI Model", model_options)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Voice Settings
        st.subheader("🎤 Voice Features")
        voice_enabled = st.checkbox("Enable Voice", value=True)
        
        if voice_enabled and st.session_state.voice_handler:
            mic_status = "🟢" if st.session_state.voice_handler.microphone else "🔴"
            tts_status = "🟢" if st.session_state.voice_handler.tts_engine else "🔴"
            
            st.markdown(f"**Microphone:** {mic_status}")
            st.markdown(f"**Text-to-Speech:** {tts_status}")
            
            auto_speak = st.checkbox("Auto-speak responses", value=False)
            
            # Global stop button for voice
            if st.session_state.voice_handler.is_speaking:
                if st.button("⏹️ Stop All Speech", type="secondary"):
                    st.session_state.voice_handler.stop_speech()
                    st.success("Speech stopped!")
                    st.rerun()
        else:
            auto_speak = False
        
        st.markdown("---")
        
        # PDF Upload Section
        st.subheader("📄 PDF Upload")
        
        uploaded_file = st.file_uploader("Choose PDF", type="pdf")
        
        if uploaded_file and api_key:
            if st.button("📂 Load PDF", type="primary"):
                if st.session_state.client is None:
                    st.session_state.client = OpenRouterClient(api_key)
                
                with st.spinner("Loading PDF..."):
                    pdf_bytes = uploaded_file.read()
                    result = st.session_state.client.load_pdf_from_bytes(
                        pdf_bytes, uploaded_file.name
                    )
                    
                    if result.startswith("Error"):
                        st.error(result)
                    else:
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_filename = uploaded_file.name
                        st.success(f"✅ {uploaded_file.name} loaded!")
                        
                        # Add system message to chat
                        st.session_state.chat_history.append({
                            "role": "system",
                            "content": f"📄 PDF loaded: **{uploaded_file.name}**\n\nYou can now ask questions about this document!",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        st.rerun()
        
        # PDF Status
        if st.session_state.pdf_loaded:
            st.success(f"📄 **{st.session_state.pdf_filename}**")
            if st.button("🗑️ Clear PDF"):
                if st.session_state.client:
                    st.session_state.client.clear_pdf_content()
                st.session_state.pdf_loaded = False
                st.session_state.pdf_filename = ""
                st.success("PDF cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.session_state.pdf_loaded:
            if st.button("📋 Summarize PDF"):
                if st.session_state.client:
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.client.summarize_pdf()
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": "Please summarize the uploaded PDF",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": summary,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        st.rerun()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            st.rerun()
    
    # Main Chat Interface
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to start chatting.")
        st.info("💡 Get a free API key from [OpenRouter](https://openrouter.ai/)")
        return
    
    # Initialize client
    if st.session_state.client is None:
        st.session_state.client = OpenRouterClient(api_key)
    
    # Chat History Display
    st.subheader("💬 Chat")
    
    # Create chat container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "system":
                    st.info(message["content"])
                else:
                    display_message(
                        message["role"], 
                        message["content"], 
                        message["timestamp"],
                        show_audio=(message["role"] == "assistant" and voice_enabled),
                        voice_handler=st.session_state.voice_handler
                    )
                st.markdown("---")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome to AI Learning Assistant!</h3>
                <p>Start a conversation by typing a message below or upload a PDF to analyze.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Input Section (Fixed at bottom)
    st.markdown("### 💭 Your Message")
    
    # Create input columns
    col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
    
    with col1:
        # Use the voice input if available, otherwise use the text input
        current_value = st.session_state.voice_input if st.session_state.voice_input else ""
        user_input = st.text_input(
            "Type your message...",
            value=current_value,
            placeholder="Ask me anything about your studies or the uploaded PDF...",
            label_visibility="collapsed",
            key="user_text_input"
        )
        
        # If there's voice input, clear it after using
        if st.session_state.voice_input:
            st.session_state.voice_input = ""
    
    with col2:
        if voice_enabled and st.session_state.voice_handler and st.session_state.voice_handler.microphone:
            if st.button("🎤 Voice", help="Click to use voice input", key="voice_button"):
                st.session_state.listening = True
                placeholder = st.empty()
                with placeholder:
                    st.info("🎤 Listening... Speak now!")
                
                # Get voice input
                speech_text = st.session_state.voice_handler.listen_for_speech()
                
                placeholder.empty()
                
                if not speech_text.startswith("Error"):
                    st.session_state.voice_input = speech_text
                    st.success(f"🎤 Voice captured: {speech_text}")
                    st.rerun()
                else:
                    st.error(speech_text)
                
                st.session_state.listening = False
        else:
            st.button("🎤 Voice", disabled=True, help="Voice input not available")
    
    with col3:
        send_button = st.button("📤 Send", type="primary")
    
    # Handle message sending
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Get AI response
        with st.spinner("🤖 AI is thinking..."):
            if st.session_state.pdf_loaded:
                response = st.session_state.client.ask_pdf_question(user_input)
            else:
                response = st.session_state.client.simple_prompt(user_input)
            
            # Add AI response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Auto-speak if enabled
            if auto_speak and voice_enabled and st.session_state.voice_handler:
                st.session_state.voice_handler.speak_text(response)
        
        # Clear the input field
        st.session_state.voice_input = ""
        st.rerun()
    
    # Footer with helpful tips
    with st.expander("💡 Tips & Features"):
        st.markdown("""
        **🎯 How to use:**
        - Upload a PDF in the sidebar to ask questions about it
        - Use voice input by clicking the microphone button
        - Enable auto-speak to hear responses automatically
        - Use quick actions in the sidebar for common tasks
        
        **🎤 Voice Features:**
        - Click the microphone button to speak your question
        - Responses can be spoken automatically or manually
        - Audio download available for responses
        
        **📄 PDF Analysis:**
        - Upload any PDF document
        - Ask specific questions about the content
        - Request summaries, key points, or explanations
        
        **⚙️ Settings:**
        - Choose different AI models in the sidebar
        - Adjust temperature for creativity vs focus
        - Enable/disable voice features as needed
        """)
    
    # Status bar at bottom
    status_col1, status_col2, status_col3 = st.columns([1, 1, 1])
    
    with status_col1:
        if api_key:
            st.markdown("🟢 **API Connected**")
        else:
            st.markdown("🔴 **API Not Connected**")
    
    with status_col2:
        if st.session_state.pdf_loaded:
            st.markdown(f"📄 **PDF Loaded:** {st.session_state.pdf_filename}")
        else:
            st.markdown("📄 **No PDF Loaded**")
    
    with status_col3:
        if voice_enabled and st.session_state.voice_handler:
            if st.session_state.voice_handler.microphone and st.session_state.voice_handler.tts_engine:
                st.markdown("🎤 **Voice Ready**")
            else:
                st.markdown("🎤 **Voice Partial**")
        else:
            st.markdown("🎤 **Voice Disabled**")

if __name__ == "__main__":
    main()