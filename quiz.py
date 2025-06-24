import streamlit as st
import random
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
import time

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
    """Client for OpenRouter API to generate quiz questions"""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_questions(self, category: str, difficulty: str, num_questions: int = 10) -> List[Question]:
        """Generate quiz questions using OpenRouter API"""
        
        prompt = f"""Generate exactly {num_questions} quiz questions about {category} at {difficulty} difficulty level.

Please create a mix of multiple choice (with 4 options) and true/false questions.

For each question, provide:
1. The question text
2. Question type ('multiple_choice' or 'true_false')
3. Options (4 choices for multiple choice, ['True', 'False'] for true/false)
4. The correct answer (must match exactly one of the options)
5. A brief explanation of why the answer is correct

Return the questions in this exact JSON format:
{{
  "questions": [
    {{
      "question_text": "Your question here?",
      "question_type": "multiple_choice",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Brief explanation here."
    }},
    {{
      "question_text": "Your true/false question here.",
      "question_type": "true_false",
      "options": ["True", "False"],
      "correct_answer": "True",
      "explanation": "Brief explanation here."
    }}
  ]
}}

Make sure:
- Questions are appropriate for {difficulty} difficulty
- Mix of question types (roughly 60% multiple choice, 40% true/false)
- Clear, unambiguous questions
- Correct answers that exactly match one of the provided options
- Educational explanations

Category: {category}
Difficulty: {difficulty}
Number of questions: {num_questions}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse the JSON response
            questions_data = json.loads(content)
            questions = []
            
            for q_data in questions_data['questions']:
                question = Question(
                    question_text=q_data['question_text'],
                    question_type=q_data['question_type'],
                    options=q_data['options'],
                    correct_answer=q_data['correct_answer'],
                    explanation=q_data.get('explanation', '')
                )
                questions.append(question)
            
            return questions
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return self._get_fallback_questions(category, difficulty)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse API response: {str(e)}")
            return self._get_fallback_questions(category, difficulty)
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return self._get_fallback_questions(category, difficulty)
    
    def _get_fallback_questions(self, category: str, difficulty: str) -> List[Question]:
        """Fallback questions if API fails"""
        return [
            Question(
                "This is a fallback question. Please check your API configuration.",
                "true_false",
                ["True", "False"],
                "True",
                "This appears when the API is not properly configured."
            )
        ]

class QuizManager:
    """Manages quiz state and operations"""
    
    def __init__(self):
        self.initialize_session_state()
    
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
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        if 'api_configured' not in st.session_state:
            st.session_state.api_configured = False
    
    def reset_quiz(self):
        """Reset all quiz-related session state"""
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.questions = []
        st.session_state.user_answers = []
        st.session_state.quiz_completed = False
        st.session_state.score = 0
        st.session_state.show_feedback = False

def display_api_setup():
    """Display API configuration screen"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>🧠 Interactive Quiz</h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>Powered by AI-Generated Questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔑 API Configuration")
    st.markdown("""
    This quiz uses OpenRouter API to generate dynamic questions. You'll need an API key to get started.
    
    **How to get your API key:**
    1. Visit [OpenRouter.ai](https://openrouter.ai/)
    2. Sign up for an account
    3. Go to your dashboard and create an API key
    4. Enter your API key below
    """)
    
    # API key input
    api_key = st.text_input(
        "Enter your OpenRouter API Key:",
        type="password",
        placeholder="sk-or-...",
        help="Your API key will be stored securely for this session only"
    )
    
    # Model selection
    model_options = [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o",
        "openai/gpt-3.5-turbo",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-pro"
    ]
    
    selected_model = st.selectbox(
        "Select AI Model:",
        model_options,
        help="Different models may produce different question styles and quality"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Configure API", type="primary", use_container_width=True):
            if api_key.strip():
                st.session_state.api_key = api_key.strip()
                st.session_state.selected_model = selected_model
                st.session_state.api_configured = True
                st.success("✅ API configured successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter a valid API key")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background-color: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #0ea5e9;'>
        <h4 style='margin-top: 0; color: #0c4a6e;'>🔒 Privacy & Security</h4>
        <ul style='margin-bottom: 0;'>
            <li>Your API key is only stored in your browser session</li>
            <li>API key is never logged or permanently stored</li>
            <li>All communication is encrypted (HTTPS)</li>
            <li>Questions are generated fresh for each quiz</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_start_screen():
    """Display the quiz start screen with category and difficulty selection"""
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>🧠 Interactive Quiz</h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>AI-Generated Questions Tailored for You!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    ### 📋 How it works:
    - **AI-Generated Questions** - Fresh questions created just for you
    - **10 questions** per quiz with mixed question types
    - **Immediate feedback** after each answer
    - **Progress tracking** throughout the quiz
    - **Detailed explanations** for every answer
    """)
    
    st.markdown("---")
    
    # Selection columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Select Category")
        category_options = [
            "Programming", "Web Development", "Data Science", "Machine Learning",
            "Artificial Intelligence", "Cybersecurity", "Cloud Computing",
            "Software Engineering", "Database Management", "DevOps",
            "Mathematics", "Physics", "Chemistry", "Biology",
            "History", "Geography", "Literature", "Philosophy",
            "Business", "Economics", "Marketing", "Finance",
            "Health & Medicine", "Psychology", "General Knowledge"
        ]
        
        category = st.selectbox(
            "Choose your topic:",
            category_options,
            key="category_select",
            help="Select the topic you want to be quizzed on"
        )
    
    with col2:
        st.markdown("#### ⚡ Select Difficulty")
        difficulty = st.selectbox(
            "Choose difficulty level:",
            ["Beginner", "Intermediate", "Advanced", "Expert"],
            key="difficulty_select",
            help="Select the difficulty level that matches your expertise"
        )
    
    # Custom topic option
    st.markdown("---")
    st.markdown("#### 🎯 Custom Topic (Optional)")
    custom_topic = st.text_input(
        "Or enter a specific topic:",
        placeholder="e.g., React Hooks, Quantum Computing, Ancient Rome...",
        help="Be specific for better questions!"
    )
    
    st.markdown("---")
    
    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
            final_category = custom_topic.strip() if custom_topic.strip() else category
            st.session_state.selected_category = final_category
            st.session_state.selected_difficulty = difficulty
            st.session_state.quiz_started = True
            st.rerun()
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        💡 <b>Tip:</b> Questions are generated fresh each time using AI. Expect variety and creativity!
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
    if percentage >= 80:
        st.success("🌟 Outstanding performance! You've mastered this topic!")
    elif percentage >= 60:
        st.info("👍 Well done! You have a good understanding of the material.")
    elif percentage >= 40:
        st.warning("📚 Not bad! Consider reviewing the material to improve your understanding.")
    else:
        st.error("💪 Keep studying! Practice makes perfect.")
    
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
                    <span style='color: #155724;'>✅ Your answer: {user_answer}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8d7da; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                    <strong>Question {i+1}:</strong> {question.question_text}<br>
                    <span style='color: #721c24;'>❌ Your answer: {user_answer}</span><br>
                    <span style='color: #004085;'>✓ Correct answer: {question.correct_answer}</span>
                </div>
                """, unsafe_allow_html=True)
            
            if question.explanation:
                st.markdown(f"**💡 Explanation:** {question.explanation}")
            
            if i < len(st.session_state.questions) - 1:
                st.markdown("---")

def main():
    """Main application function"""
    
    # Initialize quiz manager
    quiz_manager = QuizManager()
    
    # Check if API is configured
    if not st.session_state.api_configured:
        display_api_setup()
        return
    
    # Initialize OpenRouter client
    openrouter_client = OpenRouterClient(
        st.session_state.api_key,
        st.session_state.get('selected_model', 'anthropic/claude-3.5-sonnet')
    )
    
    # Main quiz logic
    if not st.session_state.quiz_started:
        # Show start screen
        display_start_screen()
    
    elif st.session_state.quiz_completed:
        # Show results
        display_results()
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Take Another Quiz", use_container_width=True):
                quiz_manager.reset_quiz()
                st.rerun()
        
        with col2:
            if st.button("🎯 New Topic", use_container_width=True):
                quiz_manager.reset_quiz()
                st.rerun()
        
        with col3:
            if st.button("📤 Share Results", use_container_width=True):
                score = st.session_state.score
                total = len(st.session_state.questions)
                percentage = (score / total) * 100
                category = st.session_state.selected_category
                difficulty = st.session_state.selected_difficulty
                
                share_text = f"I just completed a {difficulty} {category} quiz and scored {score}/{total} ({percentage:.1f}%)! 🧠✨ #AIQuiz"
                st.success(f"Share this: {share_text}")
    
    else:
        # Quiz in session
        
        # Generate questions if not already done
        if not st.session_state.questions:
            with st.spinner("🤖 AI is generating your personalized quiz questions... This may take a moment."):
                try:
                    st.session_state.questions = openrouter_client.generate_questions(
                        st.session_state.selected_category,
                        st.session_state.selected_difficulty,
                        10
                    )
                    if not st.session_state.questions or len(st.session_state.questions) == 0:
                        st.error("Failed to generate questions. Please check your API key and try again.")
                        st.session_state.quiz_started = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    st.session_state.quiz_started = False
                    st.rerun()
        
        current_q_index = st.session_state.current_question
        current_question = st.session_state.questions[current_q_index]
        
        # Quiz header
        st.markdown(f"""
        <div style='text-align: center; background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: #1f77b4; margin: 0;'>{st.session_state.selected_category} Quiz</h2>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Difficulty: {st.session_state.selected_difficulty} | AI-Generated Questions</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.show_feedback:
            # Display question
            user_answer = display_question(current_question, current_q_index + 1)
            
            if user_answer is not None:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("✅ Submit Answer", type="primary", use_container_width=True):
                        # Process answer
                        st.session_state.user_answers.append(user_answer)
                        
                        is_correct = user_answer == current_question.correct_answer
                        if is_correct:
                            st.session_state.score += 1
                        
                        st.session_state.show_feedback = True
                        st.rerun()
            else:
                st.info("👆 Please select an answer before submitting.")
        
        else:
            # Show feedback
            user_answer = st.session_state.user_answers[-1]
            is_correct = user_answer == current_question.correct_answer
            
            # Display question and user's answer
            st.markdown(f"### Question {current_q_index + 1}")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                <h4 style='margin: 0; color: #1f77b4;'>{current_question.question_text}</h4>
                <p style='margin: 1rem 0 0 0; color: #666;'><strong>Your answer:</strong> {user_answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_feedback(current_question, user_answer, is_correct)
            
            # Navigation
            if current_q_index < len(st.session_state.questions) - 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("➡️ Next Question", type="primary", use_container_width=True):
                        st.session_state.current_question += 1
                        st.session_state.show_feedback = False
                        st.rerun()
            else:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🎯 View Final Results", type="primary", use_container_width=True):
                        st.session_state.quiz_completed = True
                        st.rerun()
    
    # Sidebar with quiz info
    with st.sidebar:
        st.markdown("### 🤖 AI Quiz Info")
        
        if st.session_state.api_configured:
            st.success("✅ API Connected")
            st.markdown(f"**Model:** {st.session_state.get('selected_model', 'claude-3.5-sonnet').split('/')[-1]}")
        
        if st.session_state.quiz_started and not st.session_state.quiz_completed:
            # Current progress
            current_q = st.session_state.current_question + 1
            progress = current_q / 10
            st.progress(progress)
            st.write(f"Question {current_q} of 10")
            
            # Current score
            if st.session_state.user_answers:
                current_score = st.session_state.score
                questions_answered = len(st.session_state.user_answers)
                st.metric("Current Score", f"{current_score}/{questions_answered}")
            
            st.markdown("---")
            st.markdown(f"**Topic:** {st.session_state.selected_category}")
            st.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
        
        elif st.session_state.quiz_completed:
            # Final results summary
            score = st.session_state.score
            total = len(st.session_state.questions)
            percentage = (score / total) * 100
            
            st.metric("Final Score", f"{score}/{total}")
            st.metric("Percentage", f"{percentage:.1f}%")
            
            st.markdown("---")
            st.markdown(f"**Topic:** {st.session_state.selected_category}")
            st.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
        
        else:
            # Welcome message
            st.markdown("""
            ### Welcome! 👋
            
            Experience AI-powered quizzes! 
            
            **Features:**
            - 🤖 AI-generated questions
            - 🎯 Any topic you want
            - ⚡ Multiple difficulty levels
            - 💡 Instant explanations
            - 📊 Detailed results
            - 🔄 Always fresh content
            
            Choose your topic and let AI create the perfect quiz for you!
            """)
        
        st.markdown("---")
        
        # Reset and settings buttons
        if st.button("🔄 Reset Quiz", help="Start over with a new quiz"):
            quiz_manager.reset_quiz()
            st.rerun()
        
        if st.button("⚙️ Change API Settings", help="Reconfigure API settings"):
            st.session_state.api_configured = False
            quiz_manager.reset_quiz()
            st.rerun()

# Run the application
if __name__ == "__main__":
    main()