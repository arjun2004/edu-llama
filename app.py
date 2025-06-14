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
import re
from PIL import Image
import asyncio

class ImageScraper:
    def __init__(self):
        # Firecrawl API configuration
        self.firecrawl_api_key = "fc-90a16588f1684a14a4c35396cea6d911"
        self.firecrawl_scrape_url = "https://api.firecrawl.dev/v1/scrape"
    
    def extract_search_keywords(self, prompt: str) -> str:
        """Extract relevant keywords from user prompt for image search"""
        # Remove common question words and phrases
        question_words = [
            'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which',
            'tell', 'me', 'about', 'explain', 'describe', 'define', 'definition',
            'can', 'you', 'please', 'help', 'understand', 'learning', 'study',
            'the', 'a', 'an', 'and', 'or', 'but', 'for', 'in', 'on', 'at',
            'to', 'of', 'with', 'by', 'from', 'as', 'like', 'than'
        ]
        
        # Convert to lowercase and split into words
        words = prompt.lower().split()
        
        # Remove question words and keep meaningful terms
        meaningful_words = []
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word not in question_words and len(clean_word) > 2:
                meaningful_words.append(clean_word)
        
        # If we have meaningful words, join them
        if meaningful_words:
            # Take the first 2-3 most relevant words to avoid overly specific searches
            search_terms = meaningful_words[:3]
            return ' '.join(search_terms)
        
        # Fallback: use original prompt if no meaningful words found
        return prompt
    
    def get_images_simple_scrape(self, query: str) -> List[str]:
        """Simplified approach using Firecrawl API"""
        
        headers = {
            "Authorization": f"Bearer {self.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        
        # Try Wikipedia first as it's more scraping-friendly
        wikipedia_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        
        data = {
            "url": wikipedia_url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True
        }
        
        try:
            response = requests.post(self.firecrawl_scrape_url, headers=headers, json=data)
            
            if response.status_code == 200:
                json_data = response.json()
                
                # Look for images in the correct nested structure
                images = []
                
                # Check the correct nested structure: json_data['data']
                if 'data' in json_data:
                    data_content = json_data['data']
                    
                    # Check if there's a direct images field in data
                    if isinstance(data_content, dict) and 'images' in data_content:
                        images = data_content['images']
                    
                    # Check in metadata within data
                    elif isinstance(data_content, dict) and 'metadata' in data_content and 'images' in data_content['metadata']:
                        images = data_content['metadata']['images']
                    
                    # Parse HTML content for images if available
                    elif isinstance(data_content, dict) and 'html' in data_content:
                        html_content = data_content['html']
                        # Simple regex to find image URLs
                        img_pattern = r'<img[^>]+src="([^"]+)"'
                        found_images = re.findall(img_pattern, html_content)
                        # Filter for valid URLs and convert relative URLs to absolute
                        for img in found_images:
                            if img.startswith('http'):
                                images.append(img)
                            elif img.startswith('//'):
                                images.append('https:' + img)
                            elif img.startswith('/'):
                                images.append('https://en.wikipedia.org' + img)
                    
                    # Also check markdown content for image references
                    elif isinstance(data_content, dict) and 'markdown' in data_content:
                        markdown_content = data_content['markdown']
                        # Find markdown images ![alt](url)
                        md_img_pattern = r'!\[.*?\]\((https?://[^\)]+)\)'
                        found_images = re.findall(md_img_pattern, markdown_content)
                        images.extend(found_images)
                
                # Remove duplicates and filter out small icons
                unique_images = []
                seen = set()
                for img in images:
                    if img not in seen and not any(skip in img.lower() for skip in ['icon', 'favicon', 'logo', 'thumb/1', 'thumb/2']):
                        unique_images.append(img)
                        seen.add(img)
                
                return unique_images[:3]
            else:
                st.error(f"Firecrawl API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Exception in image scraping: {str(e)}")
            return []
    
    def get_images_from_alternative_sources(self, query: str) -> List[str]:
        """Try multiple sources for images related to the query"""
        
        # Alternative image sources that are more scraping-friendly
        sources = [
            f"https://commons.wikimedia.org/wiki/Special:Search?search={query.replace(' ', '+')}&go=Go",
            f"https://pixabay.com/images/search/{query.replace(' ', '+')}/",
            f"https://unsplash.com/s/photos/{query.replace(' ', '+')}"
        ]
        
        headers = {
            "Authorization": f"Bearer {self.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        
        all_images = []
        
        for source_url in sources:
            try:
                data = {
                    "url": source_url,
                    "formats": ["extract"],
                    "extract": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "images": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of image URLs found on the page"
                                }
                            }
                        }
                    }
                }
                
                response = requests.post(self.firecrawl_scrape_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    json_data = response.json()
                    
                    # Try different ways to extract images
                    if 'extract' in json_data and 'images' in json_data['extract']:
                        all_images.extend(json_data['extract']['images'])
                    elif 'images' in json_data:
                        all_images.extend(json_data['images'])
                    
                    # If we got some images, break early
                    if len(all_images) >= 3:
                        break
                        
            except Exception as e:
                st.error(f"Error scraping {source_url}: {str(e)}")
                continue
        
        return all_images[:3]  # Return top 3 images

    async def search_images(self, query: str, max_images: int = 3) -> List[Dict]:
        """Search for images related to the query using Firecrawl API"""
        try:
            # Extract meaningful keywords from the query for image search
            search_keywords = self.extract_search_keywords(query)
            st.info(f"🔍 Searching for images with keywords: '{search_keywords}' (from prompt: '{query}')")
            
            # First try the simple scrape approach
            image_urls = self.get_images_simple_scrape(search_keywords)
            
            # If that doesn't work, try alternative sources
            if not image_urls:
                st.info("Trying alternative sources...")
                image_urls = self.get_images_from_alternative_sources(search_keywords)
            
            # Process and validate images
            valid_images = []
            for idx, img_url in enumerate(image_urls[:max_images]):
                try:
                    st.info(f"Processing image {idx + 1}: {img_url}")
                    
                    # Download and validate image
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                    
                    img_response = requests.get(img_url, headers=headers, timeout=10)
                    if img_response.status_code == 200:
                        try:
                            img = Image.open(BytesIO(img_response.content))
                            # Convert to RGB if needed
                            if img.mode in ('RGBA', 'P'):
                                img = img.convert('RGB')
                            # Resize if too large
                            if max(img.size) > 800:
                                img.thumbnail((800, 800))
                            # Convert to bytes
                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            valid_images.append({
                                'url': img_url,
                                'title': f"Image related to: {search_keywords}",
                                'image_data': img_byte_arr
                            })
                            st.success(f"Successfully processed image {idx + 1}")
                        except Exception as img_error:
                            st.warning(f"Failed to process image {idx + 1}: {str(img_error)}")
                    else:
                        st.warning(f"Failed to download image {idx + 1}: HTTP {img_response.status_code}")
                except Exception as e:
                    st.warning(f"Failed to process image {idx + 1}: {str(e)}")
                    continue
            
            st.info(f"Successfully processed {len(valid_images)} images")
            return valid_images
        except Exception as e:
            st.error(f"Image search error: {str(e)}")
            return []

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
        # Initialize image scraper
        self.image_scraper = ImageScraper()
    
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
    
    async def simple_prompt(self, prompt: str) -> Dict:
        """Get a simple text response from the model"""
        try:
            # Get text response using chat completion
            text_response = self.chat_completion(
                [
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
                    }
                ],
                "meta-llama/llama-3.1-8b-instruct:free"
            )

            # Handle error or missing output
            if not text_response or "error" in text_response:
                return {
                    'text': f"Error: {text_response.get('error', 'Unknown error occurred')}",
                    'images': []
                }

            try:
                content = text_response["choices"][0]["message"]["content"].strip()
                text_content = content if content else "Sorry, I couldn't generate a response. Please try rephrasing your question."
                
                # Get images asynchronously using the original prompt
                images = await self.image_scraper.search_images(prompt)
                st.info(f"Found {len(images)} images for prompt: {prompt}")
                
                # Structure the response
                result = {
                    'text': text_content,
                    'images': images
                }
                
                st.info(f"Response structure: {result}")
                return result
                
            except (KeyError, IndexError, TypeError) as e:
                st.error(f"Error processing response: {str(e)}")
                return {
                    'text': "Error: Unexpected response format from the model.",
                    'images': []
                }
                
        except Exception as e:
            st.error(f"Error in simple_prompt: {str(e)}")
            return {
                'text': f"Error: {str(e)}",
                'images': []
            }

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

def display_message(message: Dict):
    """Display a message in the chat interface"""
    try:
        # Get message content
        content = message.get('content', '')
        role = message.get('role', 'assistant')
        
        # Create message container
        with st.chat_message(role):
            # Display text content
            if isinstance(content, str):
                st.write(content)
            elif isinstance(content, dict):
                # Handle text content
                if 'text' in content:
                    st.write(content['text'])
                
                # Handle images
                if 'images' in content:
                    st.info(f"Found {len(content['images'])} images to display")
                    for idx, img in enumerate(content['images']):
                        try:
                            if isinstance(img, dict) and 'image_data' in img:
                                st.image(img['image_data'], caption=img.get('title', f'Image {idx + 1}'))
                                st.success(f"Successfully displayed image {idx + 1}")
                            else:
                                st.warning(f"Invalid image data format for image {idx + 1}")
                        except Exception as e:
                            st.error(f"Error displaying image {idx + 1}: {str(e)}")
                else:
                    st.warning("No images found in content")
            else:
                st.warning(f"Unexpected content type: {type(content)}")
            
            # Display timestamp if available
            if 'timestamp' in message:
                st.caption(message['timestamp'])
    except Exception as e:
        st.error(f"Error displaying message: {str(e)}")
        st.error(f"Message content: {message}")

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
                display_message(message)
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
    
    with col2:
        if voice_enabled and st.session_state.voice_handler and st.session_state.voice_handler.microphone:
            if st.button("🎤 Voice", help="Click to use voice input", key="voice_button"):
                st.session_state.listening = True
                st.rerun()  # Rerun to show the listening state
        else:
            st.button("🎤 Voice", disabled=True, help="Voice input not available")
    
    # Show listening prompt if in listening state
    if st.session_state.listening:
        st.info("🎤 Listening... Speak now!")
        speech_text = st.session_state.voice_handler.listen_for_speech()
        
        if not speech_text.startswith("Error"):
            st.session_state.voice_input = speech_text
            st.success(f"🎤 Voice captured: {speech_text}")
        else:
            st.error(speech_text)
        
        st.session_state.listening = False
        st.rerun()
    
    with col3:
        send_button = st.button("📤 Send", type="primary")
    
    # Handle sending the message
    if (send_button or st.session_state.voice_input) and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Clear voice input if it was used
        if st.session_state.voice_input:
            st.session_state.voice_input = ""
        
        # Get AI response
        with st.spinner("🤖 AI is thinking..."):
            try:
                if st.session_state.pdf_loaded:
                    response = st.session_state.client.ask_pdf_question(user_input)
                else:
                    # Run async operation
                    response = asyncio.run(st.session_state.client.simple_prompt(user_input))
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Auto-speak if enabled
                if auto_speak and voice_enabled and st.session_state.voice_handler:
                    text_to_speak = response['text'] if isinstance(response, dict) else response
                    st.session_state.voice_handler.speak_text(text_to_speak)
                
                # Force a rerun to update the display
                st.rerun()
            except Exception as e:
                st.error(f"Error getting AI response: {str(e)}")
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