# API Key Integration Documentation

## Overview
This document explains how the OpenRouter API key is now shared between `app.py` (AI Learning Assistant) and `quiz.py` (Interactive Quiz) applications.

## How It Works

### 1. Single API Key Configuration
- **Primary Configuration**: The API key is configured only in `app.py` through the sidebar
- **Shared Access**: The API key is stored in Streamlit's session state and shared with `quiz.py`
- **No Duplication**: You no longer need to enter the API key in both applications

### 2. Session State Sharing
```python
# In app.py - API key is stored in session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# In quiz.py - API key is retrieved from session state
api_key = st.session_state.get('api_key', None)
```

### 3. User Experience Flow

#### First Time Setup:
1. Open the main app (`app.py`)
2. Enter your OpenRouter API key in the sidebar
3. Navigate to the quiz app (`quiz.py`) using the navigation button
4. The quiz app will automatically use the same API key

#### Subsequent Usage:
- The API key persists across both applications during your session
- No need to re-enter the API key when switching between apps
- The API key is cleared when you refresh the browser or start a new session

## Navigation

### From Main App to Quiz:
- Use the "🧠 Go to Interactive Quiz" button in the main app
- Located in the navigation section below the header

### From Quiz to Main App:
- Use the "🏠 Go to Main App" button in the quiz app
- Appears when no API key is configured

## Error Handling

### No API Key Configured:
- Quiz app shows a clear error message
- Provides instructions to configure the API key in the main app
- Includes a direct navigation button to the main app

### Invalid API Key:
- Both apps will show appropriate error messages
- Users can update the API key in the main app sidebar

## Security Considerations

- API key is stored in Streamlit session state (client-side)
- Key is not persisted between browser sessions
- No hardcoded API keys in the code
- Users must manually enter their API key each session

## Code Changes Made

### app.py Changes:
1. Added API key storage in session state
2. Added navigation section to quiz app
3. Updated sidebar to show API key sharing status

### quiz.py Changes:
1. Removed hardcoded API key
2. Updated `get_openrouter_client()` to use session state
3. Added API key availability check in start screen
4. Added navigation back to main app
5. Added user-friendly error messages

## Testing the Integration

1. Start with `app.py`
2. Enter a valid OpenRouter API key
3. Navigate to `quiz.py` using the navigation button
4. Verify that the quiz app shows "✅ API Key Configured"
5. Try generating quiz questions to confirm the API key works

## Troubleshooting

### Quiz app shows "API Key Required":
- Go back to the main app and ensure the API key is entered
- Check that the API key is valid
- Try refreshing the page

### API calls fail in quiz:
- Verify the API key is correct in the main app
- Check your OpenRouter account for usage limits
- Ensure you have sufficient credits

## Future Enhancements

- Add API key validation before storing in session state
- Implement persistent storage for API key (with user consent)
- Add API usage statistics and monitoring
- Support for multiple API keys for different models 