# Dependencies Analysis for edu-llama AI Learning Assistant

## 📋 Overview

This document provides a comprehensive analysis of all dependencies required for the edu-llama AI Learning Assistant application.

## 🔧 Core Dependencies

### **Streamlit Framework**

- **Package**: `streamlit>=1.28.0,<2.0.0`
- **Purpose**: Main web framework for the application interface
- **Used in**: `app.py`, `pomodoro_timer.py`
- **Features**: Chat interface, sidebar, file uploads, real-time updates

### **HTTP Requests**

- **Package**: `requests>=2.31.0,<3.0.0`
- **Purpose**: API communication with OpenRouter and image scraping
- **Used in**: `app.py` (OpenRouterClient, ImageScraper)
- **Features**: Chat completions, image downloads, web scraping

### **PDF Processing**

- **Package**: `PyPDF2>=3.0.0,<4.0.0`
- **Purpose**: PDF text extraction and analysis
- **Used in**: `app.py` (OpenRouterClient)
- **Features**: PDF upload, text extraction, content analysis

## 🎤 Speech & Audio

### **Speech Recognition**

- **Package**: `SpeechRecognition>=3.10.0,<4.0.0`
- **Purpose**: Convert speech to text for voice input
- **Used in**: `app.py` (VoiceHandler)
- **Features**: Microphone input, Google Speech API integration

### **Text-to-Speech**

- **Package**: `pyttsx3>=2.90,<3.0.0`
- **Purpose**: Convert text responses to speech
- **Used in**: `app.py` (VoiceHandler)
- **Features**: Audio output, voice synthesis, audio file generation

### **Audio Interface**

- **Package**: `pyaudio>=0.2.11,<1.0.0`
- **Purpose**: Low-level audio interface for microphone access
- **Used in**: `SpeechRecognition` (dependency)
- **Features**: Microphone input, audio streaming

## 🖼️ Image Processing

### **Pillow (PIL)**

- **Package**: `Pillow>=10.0.0,<11.0.0`
- **Purpose**: Image processing and manipulation
- **Used in**: `app.py` (ImageScraper)
- **Features**: Image validation, resizing, format conversion

## 🎯 Computer Vision & AI

### **OpenCV**

- **Package**: `opencv-python>=4.8.0,<5.0.0`
- **Purpose**: Computer vision for emotion detection
- **Used in**: `cv.py` (ImprovedEmotionDetector)
- **Features**: Camera capture, face detection, image processing

### **NumPy**

- **Package**: `numpy>=1.24.0,<2.0.0`
- **Purpose**: Numerical computing for AI/ML operations
- **Used in**: `cv.py` (ImprovedEmotionDetector)
- **Features**: Array operations, mathematical computations

### **FER (Facial Emotion Recognition)**

- **Package**: `fer>=22.4.0,<23.0.0`
- **Purpose**: Facial emotion detection and analysis
- **Used in**: `cv.py` (ImprovedEmotionDetector)
- **Features**: Emotion classification, engagement scoring

## 📊 Data Visualization

### **Matplotlib**

- **Package**: `matplotlib>=3.7.0,<4.0.0`
- **Purpose**: Data visualization and plotting
- **Used in**: `cv.py` (EnhancedEmotionGUI)
- **Features**: Real-time emotion charts, engagement graphs

## 📚 Standard Library Dependencies

These are included with Python and don't require installation:

### **Core Python Libraries**

- `json` - JSON data handling
- `io` - Input/output operations
- `pathlib` - Path manipulation
- `threading` - Multi-threading support
- `base64` - Base64 encoding/decoding
- `tempfile` - Temporary file operations
- `os` - Operating system interface
- `platform` - Platform information
- `datetime` - Date and time handling
- `re` - Regular expressions
- `urllib` - URL handling
- `asyncio` - Asynchronous programming
- `typing` - Type hints
- `collections` - Specialized container datatypes
- `enum` - Enumerations

### **GUI Framework**

- `tkinter` - GUI toolkit (used in cv.py for emotion detection interface)

## 🚫 Removed Dependencies

### **Previously Included (No Longer Needed)**

- `crawl4ai` - Not used in current implementation
- `beautifulsoup4` - Not used in current implementation

## 📦 Installation Commands

### **For Development (Flexible Versions)**

```bash
pip install -r requirements.txt
```

### **For Production (Exact Versions)**

```bash
pip install -r requirements-exact.txt
```

### **Individual Installation**

```bash
pip install streamlit requests PyPDF2 SpeechRecognition pyttsx3 pyaudio Pillow opencv-python numpy fer matplotlib
```

## 🔍 Version Compatibility Notes

- **Streamlit 1.45.1**: Latest stable version with all required features
- **OpenCV 4.8.1.78**: Compatible with current emotion detection models
- **FER 22.4.0**: Latest version with improved accuracy
- **PyPDF2 3.0.1**: Latest version with better PDF parsing
- **NumPy 1.24.3**: Stable version compatible with OpenCV and FER

## ⚠️ System Requirements

### **Operating System**

- Windows 10/11 (tested)
- macOS 10.15+ (compatible)
- Linux Ubuntu 18.04+ (compatible)

### **Python Version**

- Python 3.8+ (recommended: 3.9 or 3.10)
- pip package manager

### **Hardware Requirements**

- Camera (for emotion detection)
- Microphone (for voice input)
- Speakers/Headphones (for voice output)
- 4GB+ RAM (recommended)
- 2GB+ free disk space

## 🛠️ Troubleshooting

### **Common Issues**

1. **pyaudio installation**: May require system audio libraries
2. **OpenCV camera access**: May need camera permissions
3. **FER model download**: Requires internet connection for first run
4. **Streamlit port conflicts**: Use `--server.port` flag to change port

### **Platform-Specific Notes**

- **Windows**: All dependencies work out of the box
- **macOS**: May need to install portaudio: `brew install portaudio`
- **Linux**: May need: `sudo apt-get install portaudio19-dev python3-pyaudio`
