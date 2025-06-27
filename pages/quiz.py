import streamlit as st
import random
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Interactive Quiz App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@dataclass
class Question:
    """Data class to represent a quiz question"""
    question_text: str
    question_type: str  # 'multiple_choice' or 'true_false'
    options: List[str]  # For multiple choice, ['True', 'False'] for true/false
    correct_answer: str
    explanation: str = ""

class OpenRouterClient:
    """OpenRouter API client for LLM integration"""
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

class QuizManager:
    """Manages quiz state and operations"""
    
    def __init__(self):
        self.initialize_session_state()
        # Initialize OpenRouter client using shared API key
        self.openrouter_client = self.get_openrouter_client()
    

    def get_openrouter_client(self):
        """Initialize OpenRouter client with API key from app.py's session state"""
        try:
            # Get API key from the shared session state (set by app.py)
            api_key = st.session_state.get('api_key', None)
            
            if not api_key:
                st.error("⚠️ Please enter your OpenRouter API key in the main app (app.py) first.")
                st.info("💡 Go to the main app and configure your API key in the sidebar.")
                return None
                
            return OpenRouterClient(api_key)
        except Exception as e:
            st.error(f"Error initializing OpenRouter client: {str(e)}")
            return None
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'quiz_started' not in st.session_state:
            st.session_state.quiz_started = False
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'questions' not in st.session_state:
            st.session_state.questions = []
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = []
        if 'quiz_completed' not in st.session_state:
            st.session_state.quiz_completed = False
        if 'score' not in st.session_state:
            st.session_state.score = 0
        if 'show_feedback' not in st.session_state:
            st.session_state.show_feedback = False
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None
        if 'selected_difficulty' not in st.session_state:
            st.session_state.selected_difficulty = None
        if 'custom_topic' not in st.session_state:
            st.session_state.custom_topic = ""
        if 'use_custom_topic' not in st.session_state:
            st.session_state.use_custom_topic = False
    
    def is_topic_appropriate(self, topic: str) -> bool:
        """Check if the topic is appropriate and educational"""
        if not self.openrouter_client:
            return True  # Allow if we can't check
            
        inappropriate_keywords = [
            'violence', 'drugs', 'adult', 'sexual', 'gambling', 'weapons',
            'hate', 'discrimination', 'illegal', 'harmful', 'dangerous',
            'suicide', 'self-harm', 'terrorism', 'explicit'
        ]
        
        topic_lower = topic.lower()
        if any(keyword in topic_lower for keyword in inappropriate_keywords):
            return False
            
        # Use LLM to check if topic is appropriate
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a content moderator for an educational quiz platform. "
                        "Determine if the given topic is appropriate for educational quizzes. "
                        "Respond with only 'APPROPRIATE' or 'INAPPROPRIATE'. "
                        "Topics should be educational, safe, and suitable for all ages. "
                        "Reject topics that involve violence, illegal activities, adult content, "
                        "self-harm, discrimination, or other harmful subjects."
                    )
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}"
                }
            ]
            
            response = self.openrouter_client.chat_completion(messages, max_tokens=10)
            
            if "error" not in response:
                content = response["choices"][0]["message"]["content"].strip().upper()
                return "APPROPRIATE" in content
        except Exception as e:
            st.warning(f"Could not verify topic appropriateness: {str(e)}")
            
        return True  # Default to allowing if check fails
    
    def generate_llm_questions(self, topic: str, difficulty: str) -> List[Question]:
        """Generate quiz questions using LLM based on topic and difficulty"""
        if not self.openrouter_client:
            st.error("LLM client not available. Using fallback questions.")
            return self.get_fallback_questions()
        
        # Check if topic is appropriate
        if not self.is_topic_appropriate(topic):
            st.error("⚠️ The topic you entered may not be appropriate for educational quizzes. Please choose a different topic.")
            return []
        
        system_prompt = """You are an expert quiz creator for educational purposes. Create exactly 10 quiz questions based on the given topic and difficulty level.

For each question, you must provide:
1. A clear, educational question
2. Question type: either "multiple_choice" (with 4 options) or "true_false" (with True/False options)
3. All answer options
4. The correct answer (must match exactly one of the options)
5. A brief explanation of why the answer is correct

Format your response as a valid JSON array with this exact structure:
[
  {
    "question_text": "Your question here?",
    "question_type": "multiple_choice",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Brief explanation of why this is correct."
  },
  {
    "question_text": "True or false question?",
    "question_type": "true_false",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "Brief explanation."
  }
]

Make sure:
- Questions are educational and appropriate
- Mix of multiple choice and true/false questions (aim for 7 multiple choice, 3 true/false)
- Difficulty matches the requested level
- All JSON syntax is correct
- Exactly 10 questions
- No duplicate questions
- Questions test actual knowledge, not trivia"""

        user_prompt = f"""Create 10 quiz questions about: {topic}
Difficulty level: {difficulty}

Please ensure the questions are:
- Educational and factual
- Appropriate for the {difficulty.lower()} level
- Well-researched and accurate
- Mix of multiple choice (4 options) and true/false questions
- Progressively challenging within the difficulty level

Topic: {topic}
Difficulty: {difficulty}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            with st.spinner(f"🤖 Generating {difficulty.lower()} questions about '{topic}'..."):
                response = self.openrouter_client.chat_completion(
                    messages, 
                    max_tokens=2500,
                    temperature=0.7
                )
            
            if "error" in response:
                st.error(f"Error generating questions: {response['error']}")
                return self.get_fallback_questions()
            
            content = response["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON from the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                questions_data = json.loads(json_content)
                
                # Convert to Question objects
                questions = []
                for q_data in questions_data:
                    if self.validate_question_data(q_data):
                        question = Question(
                            question_text=q_data["question_text"],
                            question_type=q_data["question_type"],
                            options=q_data["options"],
                            correct_answer=q_data["correct_answer"],
                            explanation=q_data.get("explanation", "")
                        )
                        questions.append(question)
                
                if len(questions) >= 5:  # At least 5 valid questions
                    st.success(f"✅ Generated {len(questions)} questions about '{topic}'!")
                    return questions[:10]  # Return up to 10 questions
                else:
                    st.warning("Generated insufficient valid questions. Using fallback questions.")
                    return self.get_fallback_questions()
            
            else:
                st.error("Could not parse LLM response. Using fallback questions.")
                return self.get_fallback_questions()
                
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}. Using fallback questions.")
            return self.get_fallback_questions()
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}. Using fallback questions.")
            return self.get_fallback_questions()
    
    def validate_question_data(self, q_data: dict) -> bool:
        """Validate that question data has all required fields"""
        required_fields = ["question_text", "question_type", "options", "correct_answer"]
        
        if not all(field in q_data for field in required_fields):
            return False
        
        if q_data["question_type"] not in ["multiple_choice", "true_false"]:
            return False
        
        if q_data["question_type"] == "multiple_choice" and len(q_data["options"]) != 4:
            return False
        
        if q_data["question_type"] == "true_false" and set(q_data["options"]) != {"True", "False"}:
            return False
        
        if q_data["correct_answer"] not in q_data["options"]:
            return False
        
        return True
    
    def get_fallback_questions(self) -> List[Question]:
        """Fallback questions when LLM generation fails"""
        return [
            Question("What is the capital of France?", "multiple_choice", ["London", "Berlin", "Paris", "Madrid"], "Paris", "Paris is the capital and largest city of France."),
            Question("Python is a programming language.", "true_false", ["True", "False"], "True", "Python is indeed a high-level programming language."),
            Question("Which planet is known as the Red Planet?", "multiple_choice", ["Venus", "Mars", "Jupiter", "Saturn"], "Mars", "Mars is called the Red Planet due to iron oxide on its surface."),
            Question("The Earth is flat.", "true_false", ["True", "False"], "False", "The Earth is an oblate spheroid, not flat."),
            Question("What does HTML stand for?", "multiple_choice", ["Hyper Text Markup Language", "Home Tool Markup Language", "Hyperlinks and Text Markup Language", "Hyperlinking Text Markup Language"], "Hyper Text Markup Language", "HTML stands for Hyper Text Markup Language."),
            Question("Water boils at 100°C at sea level.", "true_false", ["True", "False"], "True", "Water boils at 100°C (212°F) at standard atmospheric pressure."),
            Question("Which is the largest ocean on Earth?", "multiple_choice", ["Atlantic", "Indian", "Arctic", "Pacific"], "Pacific", "The Pacific Ocean is the largest and deepest ocean on Earth."),
            Question("Artificial Intelligence can learn without human intervention.", "true_false", ["True", "False"], "True", "Machine learning allows AI systems to learn from data without explicit programming."),
            Question("What is 2 + 2?", "multiple_choice", ["3", "4", "5", "6"], "4", "2 + 2 equals 4."),
            Question("The sun rises in the west.", "true_false", ["True", "False"], "False", "The sun rises in the east and sets in the west.")
        ]
    
    def generate_questions(self, topic: str, difficulty: str) -> List[Question]:
        """Generate quiz questions using LLM for any topic and difficulty"""
        # Always use LLM generation for custom topics
        return self.generate_llm_questions(topic, difficulty)
    
    def reset_quiz(self):
        """Reset all quiz-related session state"""
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.questions = []
        st.session_state.user_answers = []
        st.session_state.quiz_completed = False
        st.session_state.score = 0
        st.session_state.show_feedback = False
        st.session_state.custom_topic = ""
        st.session_state.use_custom_topic = False

def display_start_screen():
    """Display the quiz start screen with topic input and difficulty selection"""
    
    # Check if API key is available
    api_key = st.session_state.get('api_key', None)
    if not api_key:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>🧠 Interactive Quiz</h1>
            <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>Test your knowledge on any topic with AI-generated questions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.error("⚠️ **API Key Required**")
        st.info("""
        🔗 **This quiz app shares the API key with the main AI Learning Assistant.**
        
        To use this quiz:
        1. Go to the **main app** (AI Learning Assistant)
        2. Enter your **OpenRouter API key** in the sidebar
        3. Return to this quiz page
        
        The API key will be automatically shared between both apps.
        """)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Go to Main App", type="primary", use_container_width=True):
                st.switch_page("app.py")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #888; font-size: 0.9rem;'>
            💡 <b>Tip:</b> You only need to configure the API key once in the main app!<br>
            🔒 <b>Secure:</b> The API key is stored securely in your session.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>🧠 Interactive Quiz</h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>Test your knowledge on any topic with AI-generated questions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Key status
    st.success("✅ **API Key Configured** - Ready to generate questions!")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    ### 📋 How it works:
    - **Enter any topic** you want to learn about
    - **10 AI-generated questions** tailored to your chosen topic and difficulty
    - **Immediate feedback** after each answer with explanations
    - **Progress tracking** throughout the quiz
    - **Detailed results** at the end
    """)
    
    st.markdown("---")
    
    # Topic and difficulty selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✨ Enter Your Topic")
        custom_topic = st.text_input(
            "What would you like to be quizzed on?",
            placeholder="e.g., Ancient Rome, Machine Learning, Photography, Biology...",
            help="Enter any educational topic you'd like to learn about!",
            key="topic_input"
        )
        
        if custom_topic:
            st.markdown(f"**Your topic:** {custom_topic}")
    
    with col2:
        st.markdown("#### ⚡ Select Difficulty")
        difficulty = st.selectbox(
            "Choose difficulty level:",
            ["Easy", "Medium", "Hard"],
            key="difficulty_select",
            help="Select the difficulty level that matches your expertise"
        )
        
        # Difficulty descriptions
        difficulty_descriptions = {
            "Easy": "Basic concepts and fundamental knowledge",
            "Medium": "Intermediate understanding and application",
            "Hard": "Advanced concepts and expert-level knowledge"
        }
        
        st.markdown(f"**{difficulty}:** {difficulty_descriptions[difficulty]}")
    
    st.markdown("---")
    
    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if topic is provided
        can_start = custom_topic and len(custom_topic.strip()) >= 3
        
        if not can_start:
            st.warning("⚠️ Please enter a topic with at least 3 characters.")
            st.button("🚀 Start Quiz", type="primary", use_container_width=True, disabled=True)
        else:
            if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
                st.session_state.selected_category = custom_topic.strip()
                st.session_state.selected_difficulty = difficulty
                st.session_state.custom_topic = custom_topic.strip()
                st.session_state.quiz_started = True
                st.rerun()
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        🤖 <b>AI-Powered:</b> Questions are generated specifically for your chosen topic and difficulty level!<br>
        💡 <b>Tip:</b> Be specific with your topic for better, more focused questions.<br>
        🔒 <b>Safe Learning:</b> Topics are automatically checked to ensure educational and appropriate content.
    </div>
    """, unsafe_allow_html=True)

def display_question(question: Question, question_num: int):
    """Display a single question with options"""
    
    # Progress bar
    progress = question_num / 10
    st.progress(progress, text=f"Question {question_num} of 10")
    
    st.markdown("---")
    
    # Question header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Question {question_num}")
    with col2:
        if question.question_type == "multiple_choice":
            st.markdown("**Type:** Multiple Choice")
        else:
            st.markdown("**Type:** True/False")
    
    # Question text
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <h4 style='margin: 0; color: #1f77b4;'>{question.question_text}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer options
    if question.question_type == "multiple_choice":
        answer = st.radio(
            "**Select your answer:**",
            question.options,
            key=f"question_{question_num}",
            index=None
        )
    else:  # true_false
        answer = st.radio(
            "**Select your answer:**",
            question.options,
            key=f"question_{question_num}",
            index=None,
            horizontal=True
        )
    
    return answer

def display_feedback(question: Question, user_answer: str, is_correct: bool):
    """Display immediate feedback for the answered question"""
    
    st.markdown("---")
    
    # Feedback header
    if is_correct:
        st.markdown("""
        <div style='background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; text-align: center;'>
            <h3 style='margin: 0;'>✅ Correct! Well done!</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; text-align: center;'>
            <h3 style='margin: 0;'>❌ Incorrect</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background-color: #cce5ff; color: #004085; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
            <strong>Correct answer:</strong> {question.correct_answer}
        </div>
        """, unsafe_allow_html=True)
    
    # Explanation
    if question.explanation:
        st.markdown(f"""
        <div style='background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
            <strong>💡 Explanation:</strong><br>{question.explanation}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def display_results():
    """Display final quiz results"""
    score = st.session_state.score
    total_questions = len(st.session_state.questions)
    percentage = (score / total_questions) * 100
    
    # Results header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem;'>🎉 Quiz Complete!</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; background-color: #e8f4fd; padding: 2rem; border-radius: 10px;'>
            <h2 style='color: #1f77b4; margin: 0;'>{score}/{total_questions}</h2>
            <p style='margin: 0.5rem 0 0 0;'>Questions Correct</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; background-color: #f0f9ff; padding: 2rem; border-radius: 10px;'>
            <h2 style='color: #0ea5e9; margin: 0;'>{percentage:.1f}%</h2>
            <p style='margin: 0.5rem 0 0 0;'>Score Percentage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if percentage >= 80:
            grade = "Excellent! 🌟"
            color = "#059669"
            bg_color = "#ecfdf5"
        elif percentage >= 60:
            grade = "Good! 👍"
            color = "#0891b2"
            bg_color = "#f0f9ff"
        elif percentage >= 40:
            grade = "Fair 📚"
            color = "#ea580c"
            bg_color = "#fff7ed"
        else:
            grade = "Keep Learning! 💪"
            color = "#dc2626"
            bg_color = "#fef2f2"
        
        st.markdown(f"""
        <div style='text-align: center; background-color: {bg_color}; padding: 2rem; border-radius: 10px;'>
            <h2 style='color: {color}; margin: 0;'>{grade}</h2>
            <p style='margin: 0.5rem 0 0 0;'>Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance message
    topic = st.session_state.get('custom_topic', 'this topic')
    if percentage >= 80:
        st.success(f"🌟 Outstanding performance! You've mastered {topic}!")
    elif percentage >= 60:
        st.info(f"👍 Well done! You have a good understanding of {topic}.")
    elif percentage >= 40:
        st.warning(f"📚 Not bad! Consider studying more about {topic} to improve your understanding.")
    else:
        st.error(f"💪 Keep studying {topic}! Practice makes perfect.")
    
    st.markdown("---")
    
    # Review answers section
    with st.expander("📋 Review Your Answers", expanded=False):
        for i, (question, user_answer) in enumerate(zip(st.session_state.questions, st.session_state.user_answers)):
            is_correct = user_answer == question.correct_answer
            
            # Question review
            if is_correct:
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                    <strong>Question {i+1}:</strong> {question.question_text}<br>
                    <span style='color: #155724;'>✅ Your answer: {user_answer} (Correct)</span><br>
                    <small><strong>Explanation:</strong> {question.explanation}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8d7da; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                    <strong>Question {i+1}:</strong> {question.question_text}<br>
                    <span style='color: #721c24;'>❌ Your answer: {user_answer}</span><br>
                    <span style='color: #004085;'>✅ Correct answer: {question.correct_answer}</span><br>
                    <small><strong>Explanation:</strong> {question.explanation}</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Take Another Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_manager.reset_quiz()
            st.rerun()
    
    with col2:
        if st.button("📊 Retake Same Topic", type="secondary", use_container_width=True):
            # Keep the same topic and difficulty but reset quiz
            st.session_state.quiz_started = False
            st.session_state.current_question = 0
            st.session_state.questions = []
            st.session_state.user_answers = []
            st.session_state.quiz_completed = False
            st.session_state.score = 0
            st.session_state.show_feedback = False
            st.rerun()
    
    with col3:
        # Generate a simple text summary for sharing
        summary = f"I scored {score}/{total_questions} ({percentage:.1f}%) on a quiz about {topic}!"
        st.download_button(
            "📤 Download Summary",
            data=f"Quiz Results\n\nTopic: {topic}\nDifficulty: {st.session_state.get('selected_difficulty', 'Unknown')}\nScore: {score}/{total_questions} ({percentage:.1f}%)\n\nPerformance: {grade}",
            file_name=f"quiz_results_{topic.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def test_api_key_integration():
    """Test function to verify API key integration"""
    api_key = st.session_state.get('api_key', None)
    if api_key:
        st.success(f"✅ API Key found: {api_key[:10]}...")
        return True
    else:
        st.error("❌ No API key found in session state")
        return False

def main():
    """Main application logic"""
    
    # Test API key integration
    if st.session_state.get('debug_mode', False):
        test_api_key_integration()
    
    # Initialize quiz manager
    if 'quiz_manager' not in st.session_state:
        st.session_state.quiz_manager = QuizManager()
    
    quiz_manager = st.session_state.quiz_manager
    
    # Main application flow
    if not st.session_state.quiz_started:
        display_start_screen()
    
    elif st.session_state.quiz_completed:
        display_results()
    
    else:
        # Generate questions if not already done
        if not st.session_state.questions:
            with st.spinner("🎯 Preparing your personalized quiz..."):
                questions = quiz_manager.generate_questions(
                    st.session_state.selected_category,
                    st.session_state.selected_difficulty
                )
                
                if not questions:  # If LLM generation failed and returned empty
                    st.error("Failed to generate questions. Please try a different topic.")
                    if st.button("← Back to Topic Selection"):
                        quiz_manager.reset_quiz()
                        st.rerun()
                    return
                
                st.session_state.questions = questions
                st.session_state.user_answers = []
        
        # Display current question
        current_q = st.session_state.questions[st.session_state.current_question]
        question_num = st.session_state.current_question + 1
        
        # Show topic and difficulty info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Topic:** {st.session_state.selected_category}")
        with col2:
            st.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
        
        if not st.session_state.show_feedback:
            # Display question and get answer
            user_answer = display_question(current_q, question_num)
            
            # Submit answer button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if user_answer:
                    if st.button("Submit Answer", type="primary", use_container_width=True):
                        # Store the answer
                        st.session_state.user_answers.append(user_answer)
                        
                        # Check if correct
                        is_correct = user_answer == current_q.correct_answer
                        if is_correct:
                            st.session_state.score += 1
                        
                        st.session_state.show_feedback = True
                        st.rerun()
                else:
                    st.button("Submit Answer", type="primary", use_container_width=True, disabled=True)
        
        else:
            # Show feedback
            user_answer = st.session_state.user_answers[-1]
            is_correct = user_answer == current_q.correct_answer
            display_feedback(current_q, user_answer, is_correct)
            
            # Next question or finish button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.session_state.current_question < len(st.session_state.questions) - 1:
                    if st.button("Next Question →", type="primary", use_container_width=True):
                        st.session_state.current_question += 1
                        st.session_state.show_feedback = False
                        st.rerun()
                else:
                    if st.button("🏁 Finish Quiz", type="primary", use_container_width=True):
                        st.session_state.quiz_completed = True
                        st.rerun()
        
        # Sidebar with progress
        with st.sidebar:
            st.markdown("### 📊 Progress")
            progress = (st.session_state.current_question + 1) / len(st.session_state.questions)
            st.progress(progress)
            
            st.markdown(f"**Question:** {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
            st.markdown(f"**Score:** {st.session_state.score}/{len(st.session_state.user_answers)}")
            
            if st.session_state.user_answers:
                accuracy = (st.session_state.score / len(st.session_state.user_answers)) * 100
                st.markdown(f"**Accuracy:** {accuracy:.1f}%")
            
            st.markdown("---")
            
            if st.button("🔄 Restart Quiz", use_container_width=True):
                quiz_manager.reset_quiz()
                st.rerun()


if __name__ == "__main__":
    main()