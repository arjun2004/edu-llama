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
import re
from urllib.parse import quote, urljoin
import threading
import time
from pomodoro_timer import PomodoroTimer
import time  
from cv import ImprovedEmotionDetector, shared_engagement_state  # 👈 import detector and state

def check_disengagement_and_notify():
    """Check for disengagement and show Streamlit notification if needed"""
    try:
        # Only check if engagement monitoring is enabled
        if not st.session_state.get('engagement_monitoring_enabled', False):
            return False
            
        if detect_disengagement():
            disengaged_duration = shared_engagement_state.get('disengaged_duration', 0)
            disengaged_count = shared_engagement_state.get('disengaged_count', 0)
            if disengaged_duration > 120:
                message = f"Student has been disengaged for {disengaged_duration:.1f} seconds!"
            elif disengaged_count >= 20:
                message = f"Student has shown disengagement {disengaged_count} times!"
            else:
                message = "Student appears disengaged!"
            notification_enabled = st.session_state.get('notification_enabled', False)
            if notification_enabled:
                st.toast(f"⚠️ {message}\nIt seems like you might be getting bored — maybe try the quiz or take a short break?", icon="⚠️")
            return True
        return False
    except Exception as e:
        st.warning(f"Error checking disengagement: {e}")
        return False

class ImageScraper:
    def __init__(self):
        # Firecrawl API configuration
        self.firecrawl_api_key = "fc-90a16588f1684a14a4c35396cea6d911"
        self.firecrawl_scrape_url = "https://api.firecrawl.dev/v1/scrape"
        
        # Enhanced image sources with better success rates
        self.image_sources = [
            {
                'name': 'Wikipedia',
                'url_template': 'https://en.wikipedia.org/wiki/{}',
                'priority': 1
            },
            {
                'name': 'Wikimedia Commons',
                'url_template': 'https://commons.wikimedia.org/wiki/Category:{}',
                'priority': 2
            },
            {
                'name': 'Britannica',
                'url_template': 'https://www.britannica.com/search?query={}',
                'priority': 3
            },
            {
                'name': 'Pexels',
                'url_template': 'https://www.pexels.com/search/{}/',
                'priority': 4
            },
            {
                'name': 'Pixabay',
                'url_template': 'https://pixabay.com/images/search/{}/',
                'priority': 5
            },
            {
                'name': 'Unsplash',
                'url_template': 'https://unsplash.com/s/photos/{}',
                'priority': 6
            }
        ]
    
    def extract_search_keywords(self, prompt: str) -> List[str]:
        """Extract multiple keyword variations from user prompt for better image search"""
        # Enhanced stop words list
        question_words = {
            'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which',
            'tell', 'me', 'about', 'explain', 'describe', 'define', 'definition',
            'can', 'you', 'please', 'help', 'understand', 'learning', 'study',
            'the', 'a', 'an', 'and', 'or', 'but', 'for', 'in', 'on', 'at',
            'to', 'of', 'with', 'by', 'from', 'as', 'like', 'than', 'this',
            'that', 'these', 'those', 'will', 'would', 'could', 'should'
        }
        
        # Clean and extract meaningful words
        words = re.findall(r'\b\w+\b', prompt.lower())
        meaningful_words = [w for w in words if w not in question_words and len(w) > 2]
        
        # Generate multiple search variations
        keywords = []
        if meaningful_words:
            # Primary: First 2-3 most relevant words
            keywords.append(' '.join(meaningful_words[:3]))
            # Secondary: Individual important words
            if len(meaningful_words) > 1:
                keywords.append(' '.join(meaningful_words[:2]))
            # Tertiary: Single most important word
            keywords.append(meaningful_words[0])
        else:
            # Fallback to original prompt
            keywords.append(prompt.strip())
        
        return keywords
    
    def get_images_simple_scrape(self, query: str) -> List[str]:
        """Enhanced scraping with multiple sources and better error handling"""
        headers = {
            "Authorization": f"Bearer {self.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        
        all_images = []
        keywords = self.extract_search_keywords(query)
        
        # Try multiple sources in priority order
        for source in sorted(self.image_sources, key=lambda x: x['priority']):
            if len(all_images) >= 5:  # Collect more initially, filter later
                break
                
            for keyword in keywords[:2]:  # Try top 2 keyword variations
                try:
                    # Format URL based on source
                    if source['name'] in ['Pixabay', 'Unsplash']:
                        search_term = quote(keyword.replace(' ', '+'))
                    else:
                        search_term = keyword.replace(' ', '_')
                    
                    url = source['url_template'].format(search_term)
                    
                    data = {
                        "url": url,
                        "formats": ["markdown", "html"],
                        "onlyMainContent": True,
                        "timeout": 25000
                    }
                    
                    st.info(f"🔍 Searching {source['name']} for: {keyword}")
                    
                    response = requests.post(self.firecrawl_scrape_url, headers=headers, json=data, timeout=30)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        images = self._extract_images_from_response(json_data, source['name'])
                        
                        if images:
                            all_images.extend(images)
                            st.success(f"✅ Found {len(images)} images from {source['name']}")
                            break  # Move to next source after success
                    else:
                        st.warning(f"⚠️ {source['name']} returned status {response.status_code}")
                        
                except Exception as e:
                    st.warning(f"⚠️ Error with {source['name']}: {str(e)}")
                    continue
        
        # Remove duplicates and filter quality
        return self._filter_and_deduplicate_images(all_images)
    
    def _extract_images_from_response(self, json_data: dict, source_name: str) -> List[str]:
        """Extract images from API response based on source type"""
        images = []
        
        if 'data' not in json_data:
            return images
        
        data_content = json_data['data']
        
        if not isinstance(data_content, dict):
            return images
        
        # Check direct images field
        if 'images' in data_content:
            images.extend(data_content['images'])
        
        # Check metadata images
        if 'metadata' in data_content and 'images' in data_content['metadata']:
            images.extend(data_content['metadata']['images'])
        
        # Parse HTML content with source-specific patterns
        if 'html' in data_content:
            html_content = data_content['html']
            images.extend(self._extract_images_from_html(html_content, source_name))
        
        # Parse markdown content
        if 'markdown' in data_content:
            markdown_content = data_content['markdown']
            md_img_pattern = r'!\[.*?\]\((https?://[^\)]+)\)'
            md_images = re.findall(md_img_pattern, markdown_content)
            images.extend(md_images)
        
        return images
    
    def _extract_images_from_html(self, html_content: str, source_name: str) -> List[str]:
        """Extract images from HTML with source-specific patterns"""
        images = []
        
        # Source-specific extraction patterns
        patterns = {
            'Wikipedia': [
                r'<img[^>]+src="(//upload\.wikimedia\.org/[^"]+)"',
                r'<img[^>]+src="(https://upload\.wikimedia\.org/[^"]+)"',
                r'<img[^>]+src="(/wiki/[^"]+\.(?:jpg|jpeg|png|gif|svg))"'
            ],
            'Wikimedia Commons': [
                r'<img[^>]+src="(https://upload\.wikimedia\.org/[^"]+)"',
                r'href="(https://commons\.wikimedia\.org/wiki/File:[^"]+)"'
            ],
            'Britannica': [
                r'<img[^>]+src="(https://cdn\.britannica\.com/[^"]+)"',
                r'<img[^>]+data-src="(https://cdn\.britannica\.com/[^"]+)"'
            ],
            'Pexels': [
                r'<img[^>]+src="(https://images\.pexels\.com/photos/[^"]+)"',
                r'srcset="([^"]*https://images\.pexels\.com/photos/[^"]*)"'
            ],
            'Pixabay': [
                r'<img[^>]+src="(https://cdn\.pixabay\.com/photo/[^"]+)"',
                r'data-lazy="(https://cdn\.pixabay\.com/photo/[^"]+)"'
            ],
            'Unsplash': [
                r'<img[^>]+src="(https://images\.unsplash\.com/[^"]+)"',
                r'srcSet="([^"]*https://images\.unsplash\.com/[^"]*)"'
            ]
        }
        
        # Use source-specific patterns, fallback to generic
        source_patterns = patterns.get(source_name, [r'<img[^>]+src="([^"]+)"'])
        
        for pattern in source_patterns:
            found_images = re.findall(pattern, html_content, re.IGNORECASE)
            for img in found_images:
                # Handle protocol-relative URLs
                if img.startswith('//'):
                    images.append('https:' + img)
                elif img.startswith('/') and source_name == 'Wikipedia':
                    images.append('https://en.wikipedia.org' + img)
                else:
                    # For srcset, extract first URL
                    if ' ' in img and 'http' in img:
                        img = img.split()[0]
                    images.append(img)
        
        return images
    
    def _filter_and_deduplicate_images(self, images: List[str]) -> List[str]:
        """Filter out low-quality images and remove duplicates"""
        # Patterns to exclude
        exclude_patterns = [
            'icon', 'favicon', 'logo', 'avatar', 'thumb/1', 'thumb/2',
            'edit-icon', 'commons-logo', 'wikimedia-button', 'sprite',
            'w=50', 'w=100', 'h=50', 'h=100', 'placeholder'
        ]
        
        # Filter and deduplicate
        seen = set()
        filtered_images = []
        
        for img in images:
            # Skip if already seen
            if img in seen:
                continue
            
            # Skip if matches exclude patterns
            if any(pattern in img.lower() for pattern in exclude_patterns):
                continue
            
            # Only include valid image URLs
            if any(ext in img.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']) or \
               any(domain in img.lower() for domain in ['upload.wikimedia.org', 'images.pexels.com', 'images.unsplash.com', 'cdn.pixabay.com', 'cdn.britannica.com']):
                filtered_images.append(img)
                seen.add(img)
        
        return filtered_images[:5]  # Return top 5 quality images
    
    def get_images_from_alternative_sources(self, query: str) -> List[str]:
        """Fallback method - keeping for compatibility"""
        # This now uses the enhanced get_images_simple_scrape
        return self.get_images_simple_scrape(query)
    
    def _validate_image_download(self, img_url: str, headers: dict) -> tuple:
        """Validate and download image with enhanced error handling"""
        try:
            img_response = requests.get(img_url, headers=headers, timeout=15)
            
            if img_response.status_code != 200:
                return None, f"HTTP {img_response.status_code}"
            
            if len(img_response.content) < 1024:  # Skip very small files
                return None, "File too small"
            
            # Validate it's actually an image
            try:
                img = Image.open(BytesIO(img_response.content))
                
                # Skip very small images (likely icons)
                if min(img.size) < 150:
                    return None, f"Image too small: {img.size}"
                
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large (optimize for web display)
                max_size = 1000
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to bytes with optimization
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                return img_byte_arr, None
                
            except Exception as img_error:
                return None, f"Image processing error: {str(img_error)}"
                
        except Exception as e:
            return None, f"Download error: {str(e)}"
    
    async def search_images(self, query: str, max_images: int = 3) -> List[Dict]:
        """Enhanced image search with better error handling and quality filtering"""
        try:
            # Extract meaningful keywords from the query
            search_keywords = self.extract_search_keywords(query)
            primary_keyword = search_keywords[0] if search_keywords else query
            
            st.info(f"🔍 Searching for images with keywords: '{primary_keyword}' (from prompt: '{query}')")
            
            # Get image URLs using enhanced scraping
            image_urls = self.get_images_simple_scrape(primary_keyword)
            
            if not image_urls:
                st.warning("No images found with primary method, trying alternative approach...")
                # Could add more fallback methods here if needed
                return []
            
            st.info(f"📥 Found {len(image_urls)} potential images, validating...")
            
            # Process and validate images
            valid_images = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            for idx, img_url in enumerate(image_urls[:max_images * 2]):  # Try more than needed
                if len(valid_images) >= max_images:
                    break
                
                st.info(f"🖼️ Processing image {idx + 1}/{min(len(image_urls), max_images * 2)}: {img_url[:50]}...")
                
                img_data, error = self._validate_image_download(img_url, headers)
                
                if img_data:
                    valid_images.append({
                        'url': img_url,
                        'title': f"Image related to: {primary_keyword}",
                        'image_data': img_data,
                        'source': self._identify_image_source(img_url)
                    })
                    st.success(f"✅ Successfully processed image {len(valid_images)}")
                else:
                    st.warning(f"⚠️ Skipped image {idx + 1}: {error}")
            
            if valid_images:
                st.success(f"🎉 Successfully processed {len(valid_images)} high-quality images")
            else:
                st.error("❌ No valid images could be processed")
            
            return valid_images
            
        except Exception as e:
            st.error(f"❌ Image search error: {str(e)}")
            return []
    
    def _identify_image_source(self, url: str) -> str:
        """Identify the source of an image URL"""
        if 'wikimedia.org' in url or 'wikipedia.org' in url:
            return 'Wikipedia/Wikimedia'
        elif 'pexels.com' in url:
            return 'Pexels'
        elif 'pixabay.com' in url:
            return 'Pixabay'
        elif 'unsplash.com' in url:
            return 'Unsplash'
        elif 'britannica.com' in url:
            return 'Britannica'
        else:
            return 'Unknown'

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
    
    async def summarize_pdf(self, custom_prompt: str = None) -> Dict:
        """Generate a summary of the loaded PDF"""
        if not self.pdf_content:
            return {"text": "Error: No PDF content loaded. Please upload a PDF first.", "images": []}
        
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
        
        return await self.simple_prompt(prompt)
    
    async def ask_pdf_question(self, question: str) -> Dict:
        """Ask a question about the loaded PDF content"""
        if not self.pdf_content:
            return {"text": "Error: No PDF content loaded. Please upload a PDF first.", "images": []}
        
        # Prepare question prompt
        prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content:\n{self.pdf_content}\n\nIf the answer is not found in the PDF content, please say so clearly."
        
        # Handle long content by truncating if necessary
        max_content_length = 12000
        if len(prompt) > max_content_length:
            truncated_content = self.pdf_content[:max_content_length-500]
            prompt = f"Based on the following PDF content (truncated), please answer this question: {question}\n\nPDF Content:\n{truncated_content}\n\n[Note: Content was truncated due to length limits]\n\nIf the answer is not found in the available PDF content, please say so clearly."
        
        return await self.simple_prompt(prompt)
    
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
        
    limit_duration = shared_engagement_state.get('disengaged_duration_limit', 10)
    limit_count = shared_engagement_state.get('disengaged_count_limit', 3)
    return (
        shared_engagement_state.get('disengaged_duration', 0) > limit_duration or
        shared_engagement_state.get('disengaged_count', 0) >= limit_count
    )


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
    # Hide default Streamlit sidebar page menu
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    
    # Initialize disengagement tracking
    if 'last_disengagement_check' not in st.session_state:
        st.session_state.last_disengagement_check = 0
    if 'disengagement_notified' not in st.session_state:
        st.session_state.disengagement_notified = False
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
            def start_emotion_monitor():
                detector = ImprovedEmotionDetector()
                st.session_state.emotion_detector = detector
                if detector.start_camera():
                    detector.run_detection()  # runs in loop

            st.session_state.emotion_thread = threading.Thread(target=start_emotion_monitor, daemon=True)
            st.session_state.emotion_thread.start()
            st.info("🎥 Real-time emotion tracking initialized.")
    else:
        # Stop emotion detector if monitoring is disabled
        if st.session_state.emotion_detector:
            st.session_state.emotion_detector.stop_camera()
            st.session_state.emotion_detector = None
        if st.session_state.emotion_thread and st.session_state.emotion_thread.is_alive():
            st.session_state.emotion_thread = None
    
    # Periodic disengagement check only if monitoring is enabled
    if st.session_state.engagement_monitoring_enabled:
        current_time = time.time()
        if current_time - st.session_state.last_disengagement_check > 2:  # Check every 2 seconds
            st.session_state.last_disengagement_check = current_time
            
            # Check for disengagement and show notification
            if check_disengagement_and_notify():
                if not st.session_state.disengagement_notified:
                    st.session_state.disengagement_notified = True
                    st.toast("⚠️ Disengagement detected! Adapting teaching strategy...", icon="⚠️")
            else:
                st.session_state.disengagement_notified = False

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
        )
        
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
                
                engagement_box = st.empty()

                if realtime:
                    for _ in range(60):  # Refresh loop (max 60 cycles)
                        with engagement_box.container():
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

                        time.sleep(2)
                else:
                    with engagement_box.container():
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

                # Engagement score progress bar
                st.progress(engagement_score)
                
                # Initialize notification state
                if 'notification_enabled' not in st.session_state:
                    st.session_state.notification_enabled = False
                
                # Disengagement notification settings
                notification_enabled = st.checkbox("Enable Disengagement Alerts", value=st.session_state.notification_enabled)
                
                # Update session state when checkbox changes
                if notification_enabled != st.session_state.notification_enabled:
                    st.session_state.notification_enabled = notification_enabled
                    st.rerun()
                
                if notification_enabled:
                    st.info("🔔 Pop-up notifications will appear when disengagement is detected")
                    
                    # Test notification button
                    if st.button("🧪 Test Disengagement Alert", type="secondary"):
                        st.toast("This is a test notification for disengagement detection!", icon="⚠️")
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
                        summary = asyncio.run(st.session_state.client.summarize_pdf())
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
                    response = asyncio.run(st.session_state.client.ask_pdf_question(user_input))
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