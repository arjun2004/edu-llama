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
import base64
import tempfile
import os
import platform
from datetime import datetime
import re
import asyncio
import re
import threading
import time
from pomodoro_timer import PomodoroTimer
import time  
from cv import ImprovedEmotionDetector, shared_engagement_state  # 👈 import detector and state

# Disengagement alert timing
DISENGAGED_SECONDS = 15        # continuous disengagement before suggesting the quiz
NOTIFY_COOLDOWN_SECONDS = 120  # minimum gap between two quiz suggestions

@st.dialog("😌 Losing focus?")
def quiz_break_dialog():
    """Pop-up suggesting a quiz break when the student seems disengaged"""
    st.write("Looks like you need a short break. Try a quick quiz to chill! 🧠")
    col1, col2 = st.columns(2)
    if col1.button("🧠 Take the quiz", type="primary", use_container_width=True):
        st.switch_page("pages/quiz.py")
    if col2.button("Keep studying", use_container_width=True):
        st.rerun()

@st.fragment(run_every=2)
def engagement_watcher():
    """Runs every 2 s on its own (even when the student is idle) and triggers the quiz pop-up"""
    if not (st.session_state.get('engagement_monitoring_enabled', False)
            and st.session_state.get('notification_enabled', True)):
        return
    if not detect_disengagement():
        return
    if time.time() - st.session_state.get('last_quiz_prompt', 0) < NOTIFY_COOLDOWN_SECONDS:
        return
    st.session_state.last_quiz_prompt = time.time()
    st.session_state.show_quiz_dialog = True
    shared_engagement_state['disengaged_duration'] = 0
    st.rerun(scope="app")

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

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "z-ai/glm-5.2:free",
    "google/gemma-4-26b-a4b-it:free",
]

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-OpenRouter-Title": "edu-llama",
            "Content-Type": "application/json"
        }
        # Store PDF content for follow-up questions
        self.pdf_content = ""
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: str,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict:
        """Send a chat completion request to OpenRouter"""
        # OpenRouter tries these in order if one is rate-limited or down
        fallbacks = [m for m in FREE_MODELS if m != model][:2]
        payload = {
            "models": [model] + fallbacks,
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
                timeout=60
            )
            if not response.ok:
                try:
                    detail = response.json()["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    detail = response.text[:300]
                return {"error": f"Request failed ({response.status_code}): {detail}"}
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode failed: {str(e)}"}

    async def simple_prompt(self, prompt: str, model: str) -> Dict:
        """Get a simple text response from the model"""
        try:
            # Get text response using chat completion
            text_response = self.chat_completion(
                messages=[
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
                model=model
            )

            # Handle error or missing output
            if not text_response or "error" in text_response:
                return {
                    'text': f"Error: {text_response.get('error', 'Unknown error occurred')}"
                }

            try:
                content = text_response["choices"][0]["message"]["content"].strip()
                text_content = content if content else "Sorry, I couldn't generate a response. Please try rephrasing your question."
                return {'text': text_content}
                
            except (KeyError, IndexError, TypeError) as e:
                st.error(f"Error processing response: {str(e)}")
                return {
                    'text': "Error: Unexpected response format from the model."
                }
                
        except Exception as e:
            st.error(f"Error in simple_prompt: {str(e)}")
            return {
                'text': f"Error: {str(e)}"
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
    
    async def summarize_pdf(self, model: str, custom_prompt: str = None) -> Dict:
        """Generate a summary of the loaded PDF"""
        if not self.pdf_content:
            return {"text": "Error: No PDF content loaded. Please upload a PDF first."}
        
        # Prepare summary prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = "Please provide a comprehensive summary of the following PDF content. Include the main topics, key points, and important details:"
        
        # Add disengagement adaptation if monitoring is enabled and disengagement detected
        if detect_disengagement():
            adaptation_instruction = (
                "The student appears disengaged. Please adapt your explanation by simplifying the language, "
                "adding visual examples, and suggesting interactive or audio aids.\n\n"
            )
            prompt = adaptation_instruction + prompt
            
            # Show notification if not already shown
            if st.session_state.get('last_notified_state') != 'DISENGAGED':
                st.toast("⚠️ Student appears disengaged. Adapting teaching strategy...", icon="⚠️")
                st.session_state.last_notified_state = 'DISENGAGED'
        else:
            if st.session_state.get('last_notified_state') == 'DISENGAGED':
                st.session_state.last_notified_state = 'ENGAGED'
        
        # Handle long content by truncating if necessary
        max_content_length = 12000
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            if custom_prompt:
                prompt = f"{prompt}\n\nPDF Content (truncated):\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
            else:
                prompt = f"{prompt}\n\n{truncated_content}\n\n[Note: Content was truncated due to length limits]"
        else:
            prompt = f"{prompt}\n\n{self.pdf_content}"
        
        return await self.simple_prompt(prompt, model=model)
    
    async def ask_pdf_question(self, question: str, model: str) -> Dict:
        """Ask a question about the loaded PDF content"""
        if not self.pdf_content:
            return {"text": "Error: No PDF content loaded. Please upload a PDF first."}
        
        # Prepare question prompt
        prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content:\n{self.pdf_content}\n\nIf the answer is not found in the PDF content, please say so clearly."
        
        # Handle long content by truncating if necessary
        max_content_length = 12000
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            prompt = f"Based on the following PDF content (truncated), please answer this question: {question}\n\nPDF Content:\n{truncated_content}\n\n[Note: Content was truncated due to length limits]\n\nIf the answer is not found in the available PDF content, please say so clearly."
        
        return await self.simple_prompt(prompt, model=model)
    
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
def detect_disengagement():
    """Check if the student is disengaged based on real-time emotion data"""
    # Only check if engagement monitoring is enabled
    if not st.session_state.get('engagement_monitoring_enabled', False):
        return False
        
    return (
        shared_engagement_state.get('last_state') == "DISENGAGED" and
        shared_engagement_state.get('disengaged_duration', 0) > DISENGAGED_SECONDS
    )


def engagement_status_panel():
    """Show the current engagement state from the webcam detector"""
    engagement_state = shared_engagement_state.get('last_state', 'UNKNOWN')
    disengaged_duration = shared_engagement_state.get('disengaged_duration', 0)
    disengaged_count = shared_engagement_state.get('disengaged_count', 0)
    current_emotion = shared_engagement_state.get('current_emotion', 'UNKNOWN')
    engagement_score = shared_engagement_state.get('engagement_score', 0.0)
    face_detected = shared_engagement_state.get('face_detected', False)

    if engagement_state == "ENGAGED":
        st.success("🟢 Student Engaged")
    elif engagement_state == "DISENGAGED":
        st.error("🔴 Student Disengaged")
    else:
        st.warning("🟡 Student Neutral")

    st.markdown(f"**Disengaged Duration:** {disengaged_duration:.1f}s")
    st.markdown(f"**Disengagement Count:** {disengaged_count}")
    st.markdown(f"**Current Emotion:** `{current_emotion}`")
    st.markdown(f"**Engagement Score:** `{engagement_score:.2f}`")
    st.markdown(f"**Face Detected:** `{face_detected}`")
    st.progress(max(0.0, min(1.0, float(engagement_score))))

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
    # Hide default Streamlit sidebar page menu
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # Quiz-break pop-up requested by engagement_watcher()
    if st.session_state.pop('show_quiz_dialog', False):
        quiz_break_dialog()

    # Initialize disengagement tracking
    if 'last_notified_state' not in st.session_state:
        st.session_state.last_notified_state = 'UNKNOWN'
    
    # Initialize engagement monitoring state
    if 'engagement_monitoring_enabled' not in st.session_state:
        st.session_state.engagement_monitoring_enabled = False
    if 'emotion_detector' not in st.session_state:
        st.session_state.emotion_detector = None
    if 'emotion_thread' not in st.session_state:
        st.session_state.emotion_thread = None
    
    # Start Emotion Detector Thread only if monitoring is enabled
    if st.session_state.engagement_monitoring_enabled:
        if st.session_state.emotion_thread is None or not st.session_state.emotion_thread.is_alive():
            # Create the detector here (not in the thread) so session state is only touched
            # from the script thread; the thread just runs the camera loop.
            with st.spinner("Loading emotion detection model..."):
                detector = ImprovedEmotionDetector()
            if detector.start_camera():
                st.session_state.emotion_detector = detector
                st.session_state.emotion_thread = threading.Thread(target=detector.run_detection, daemon=True)
                st.session_state.emotion_thread.start()
                st.info("🎥 Real-time emotion tracking initialized.")
            else:
                st.error("Could not open the webcam. Check that it's connected and not used by another app.")
                st.session_state.engagement_monitoring_enabled = False
    else:
        # Stop emotion detector if monitoring is disabled; the detection loop releases the camera itself
        if st.session_state.emotion_detector:
            st.session_state.emotion_detector.is_running = False
            st.session_state.emotion_detector = None
        st.session_state.emotion_thread = None
        shared_engagement_state.update(last_state='UNKNOWN', disengaged_duration=0)

    # Background check that opens the quiz pop-up when the student is disengaged
    engagement_watcher()

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
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key",
            placeholder="sk-or-v1-...",
            value=st.session_state.api_key
        ).strip()

        # Store API key in session state for sharing with other pages
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        if api_key:
            st.success("✅ API Key configured")
            st.info("🔗 This API key is shared with the Quiz app")
        else:
            st.warning("⚠️ API Key required")
        
        st.markdown("---")
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        selected_model = st.selectbox("AI Model", FREE_MODELS, index=0,
                                      help="If this model is busy, OpenRouter automatically falls back to the next ones in the list.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Voice Settings
        with st.expander("🎤 Voice Features"):
            voice_enabled = st.checkbox("Enable Voice", value=True)

            if voice_enabled and st.session_state.voice_handler:
                mic_status = "🟢" if st.session_state.voice_handler.microphone else "🔴"
                tts_status = "🟢" if st.session_state.voice_handler.tts_engine else "🔴"

                st.markdown(f"**Microphone:** {mic_status}")
                st.markdown(f"**Text-to-Speech:** {tts_status}")
                auto_speak = st.checkbox("Auto-speak responses", value=False)

                if st.session_state.voice_handler.is_speaking:
                    if st.button("⏹️ Stop All Speech", type="secondary"):
                        st.session_state.voice_handler.stop_speech()
                        st.success("Speech stopped!")
                        st.rerun()
            else:
                auto_speak = False

        
        st.markdown("---")
        
        # Engagement Monitoring
        with st.expander("📊 Engagement Monitoring", expanded=False):
            # Main toggle for engagement monitoring
            engagement_enabled = st.toggle(
                "🎥 Enable Engagement Monitoring", 
                value=st.session_state.engagement_monitoring_enabled,
                help="Turn on real-time emotion and engagement tracking"
            )
            
            # Update session state when toggle changes
            if engagement_enabled != st.session_state.engagement_monitoring_enabled:
                st.session_state.engagement_monitoring_enabled = engagement_enabled
                st.rerun()
            
            if engagement_enabled:
                st.success("🟢 Engagement monitoring is active")
                
                # Initialize live update state
                if 'live_update_enabled' not in st.session_state:
                    st.session_state.live_update_enabled = False
                
                # Show monitoring interface only when enabled
                realtime = st.checkbox("🔄 Live Update", value=st.session_state.live_update_enabled)
                
                # Update session state when checkbox changes
                if realtime != st.session_state.live_update_enabled:
                    st.session_state.live_update_enabled = realtime
                    st.rerun()
                
                # Live Update refreshes just this panel every 2 s without blocking the page
                if realtime:
                    st.fragment(run_every=2)(engagement_status_panel)()
                else:
                    engagement_status_panel()

                # Initialize notification state
                if 'notification_enabled' not in st.session_state:
                    st.session_state.notification_enabled = True
                
                # Disengagement notification settings
                notification_enabled = st.checkbox("Enable Disengagement Alerts", value=st.session_state.notification_enabled)
                
                # Update session state when checkbox changes
                if notification_enabled != st.session_state.notification_enabled:
                    st.session_state.notification_enabled = notification_enabled
                    st.rerun()
                
                if notification_enabled:
                    st.info(f"🔔 A quiz-break pop-up appears after {DISENGAGED_SECONDS}s of disengagement")

                    # Test notification button
                    if st.button("🧪 Test Disengagement Alert", type="secondary"):
                        quiz_break_dialog()
                else:
                    st.warning("🔕 Disengagement alerts are disabled")
            else:
                st.warning("🔴 Engagement monitoring is disabled")
                st.info("Toggle the switch above to enable real-time emotion tracking and engagement monitoring.")

        # Sidebar: Custom Navigation Section
        with st.sidebar:
            st.markdown("## 🎓 AI Learning Assistant")

            # Engagement Status (optional toggle display)
            if st.session_state.get('engagement_monitoring_enabled', False):
                st.success("📶 Engagement Monitoring: ON")
            else:
                st.warning("📴 Engagement Monitoring: OFF")

            # Navigation Section
            st.markdown("---")
            st.subheader("🧭 Navigation")

            if st.button("🧠 Interactive Quiz", type="secondary", use_container_width=True):
                st.switch_page("pages/quiz.py")  # ✅ Make sure quiz.py is in the same directory as app.py

            st.markdown("*Navigate to other apps*")

            st.markdown("---")
            st.markdown("Made with ❤️ by your AI assistant.")

        # Pomodoro Timer Section
        st.markdown("---")
        st.subheader("⏱️ Pomodoro Focus Timer")

        # Initialize and render the Pomodoro timer
        if 'timer' not in st.session_state:
            st.session_state.timer = PomodoroTimer()
        st.session_state.timer.render_sidebar()

        
        # PDF Upload Section
        st.subheader("📄 PDF Upload")
        
        uploaded_file = st.file_uploader("Choose PDF", type="pdf")
        
        if uploaded_file and api_key:
            if st.button("📂 Load PDF", type="primary"):
                if st.session_state.client is None or st.session_state.client.api_key != api_key:
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
                        summary = asyncio.run(st.session_state.client.summarize_pdf(model=selected_model))
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
    
    # Initialize client (recreate if the API key changed)
    if st.session_state.client is None or st.session_state.client.api_key != api_key:
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
                    response = asyncio.run(st.session_state.client.ask_pdf_question(user_input, model=selected_model))
                else:
                    # Run async operation
                    response = asyncio.run(st.session_state.client.simple_prompt(user_input, model=selected_model))
                
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
    status_col1, status_col2, status_col3, status_col4 = st.columns([1, 1, 1, 1])
    
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
    
    with status_col4:
        # Engagement monitoring status
        if st.session_state.engagement_monitoring_enabled:
            # Real-time engagement status when monitoring is enabled
            engagement_state = shared_engagement_state.get('last_state', 'UNKNOWN')
            if engagement_state == "ENGAGED":
                st.markdown("🟢 **Student Engaged**")
            elif engagement_state == "DISENGAGED":
                st.markdown("🔴 **Student Disengaged**")
            else:
                st.markdown("🟡 **Student Neutral**")
        else:
            st.markdown("🔴 **Monitoring Off**")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup notification system
        pass