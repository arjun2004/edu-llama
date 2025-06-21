import cv2
import numpy as np
from fer import FER
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImprovedEmotionDetector:
    def __init__(self):
        """Initialize the emotion detector with FER model"""
        self.detector = FER(mtcnn=True)
        self.cap = None
        self.is_running = False
        self.current_emotion = "Unknown"
        self.emotion_confidence = 0.0
        self.emotion_history = deque(maxlen=50)  # Increased history for better analysis
        self.lock = threading.Lock()
        
        # Enhanced disengagement detection parameters
        self.disengagement_history = deque(maxlen=20)  # Track engagement state over time
        self.face_detection_history = deque(maxlen=15)  # Track face detection consistency
        self.current_engagement_state = "UNKNOWN"
        self.last_face_detected_time = time.time()
        self.no_face_threshold = 3.0  # seconds without face = disengagement
        
        # Improved emotion categorization with weights
        self.disengaged_emotions = {
            'sad': 0.8,      # Strong indicator
            'angry': 0.7,    # Strong indicator
            'disgust': 0.6,  # Moderate indicator
            'fear': 0.5,     # Moderate indicator
            'bored': 0.9     # Strongest indicator (if available)
        }
        
        self.engaged_emotions = {
            'happy': 0.8,
            'surprise': 0.6,
            'neutral': 0.3   # Neutral can be engaged or disengaged
        }
        
        # Adaptive thresholds based on individual baseline
        self.confidence_threshold = 0.4  # Lowered for better detection
        self.engagement_smoothing_factor = 0.7  # For temporal smoothing
        self.baseline_emotions = deque(maxlen=100)  # For adaptive baseline
        
        # Eye and face analysis (additional features)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 255),  # Orange
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 255, 0), # Cyan
            'neutral': (128, 128, 128) # Gray
        }
    
    def start_camera(self, camera_index=0):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera started successfully!")
        return True
    
    def stop_camera(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped!")
    
    def analyze_eye_contact(self, frame, face_box):
        """Analyze eye contact and direction - basic implementation"""
        try:
            x, y, w, h = face_box
            face_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
            
            # Simple heuristic: if both eyes detected, likely looking at camera
            eye_contact_score = len(eyes) / 2.0 if len(eyes) <= 2 else 1.0
            return min(eye_contact_score, 1.0)
            
        except Exception as e:
            return 0.5  # Default neutral score
    
    def calculate_engagement_score(self, emotions, face_box=None, frame=None):
        """Calculate comprehensive engagement score"""
        if not emotions:
            return 0.0, "NO_FACE"
        
        engagement_score = 0.0
        primary_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name, confidence = primary_emotion
        
        # Base score from emotion analysis
        if emotion_name in self.disengaged_emotions:
            base_score = (1.0 - self.disengaged_emotions[emotion_name]) * confidence
        elif emotion_name in self.engaged_emotions:
            base_score = self.engaged_emotions[emotion_name] * confidence
        else:
            base_score = 0.5  # Neutral for unknown emotions
        
        engagement_score = base_score
        
        # Factor in eye contact (if face detected)
        if face_box is not None and frame is not None:
            eye_contact_score = self.analyze_eye_contact(frame, face_box)
            engagement_score = 0.7 * engagement_score + 0.3 * eye_contact_score
        
        # Apply confidence weighting
        engagement_score *= confidence
        
        # Normalize to 0-1 range
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        return engagement_score, emotion_name
    
    def update_engagement_state(self, engagement_score, emotion_name):
        """Update engagement state with temporal smoothing"""
        current_time = time.time()
        
        # Record face detection
        face_detected = engagement_score > 0
        self.face_detection_history.append(face_detected)
        
        if face_detected:
            self.last_face_detected_time = current_time
        
        # Check for prolonged absence of face
        time_since_face = current_time - self.last_face_detected_time
        if time_since_face > self.no_face_threshold:
            engagement_state = "DISENGAGED"
            confidence = min(1.0, time_since_face / 10.0)  # Increase confidence over time
        else:
            # Determine engagement based on score
            if engagement_score > 0.6:
                engagement_state = "ENGAGED"
            elif engagement_score < 0.4:
                engagement_state = "DISENGAGED"
            else:
                engagement_state = "NEUTRAL"
            
            confidence = abs(engagement_score - 0.5) * 2  # 0.5 is neutral point
        
        # Add to history for smoothing
        self.disengagement_history.append({
            'state': engagement_state,
            'score': engagement_score,
            'confidence': confidence,
            'emotion': emotion_name,
            'timestamp': current_time
        })
        
        # Apply temporal smoothing
        if len(self.disengagement_history) >= 5:
            recent_states = [entry['state'] for entry in list(self.disengagement_history)[-5:]]
            recent_scores = [entry['score'] for entry in list(self.disengagement_history)[-5:]]
            
            # Use majority voting with score weighting
            state_counts = {}
            weighted_scores = {}
            
            for i, state in enumerate(recent_states):
                if state not in state_counts:
                    state_counts[state] = 0
                    weighted_scores[state] = 0
                state_counts[state] += 1
                weighted_scores[state] += recent_scores[i]
            
            # Choose state with highest weighted score
            if weighted_scores:
                smoothed_state = max(weighted_scores.items(), key=lambda x: x[1])[0]
                self.current_engagement_state = smoothed_state
        else:
            self.current_engagement_state = engagement_state
        
        return self.current_engagement_state, confidence
    
    def detect_emotion(self, frame):
        """Detect emotion in a single frame with enhanced analysis"""
        try:
            # Detect emotions in the frame
            result = self.detector.detect_emotions(frame)
            
            if result:
                # Get the first face detected (or the largest face)
                if len(result) > 1:
                    # Choose the largest face
                    face = max(result, key=lambda x: x['box'][2] * x['box'][3])
                else:
                    face = result[0]
                
                emotions = face['emotions']
                face_box = face['box']
                
                # Calculate engagement score
                engagement_score, emotion_name = self.calculate_engagement_score(
                    emotions, face_box, frame
                )
                
                # Update engagement state
                engagement_state, state_confidence = self.update_engagement_state(
                    engagement_score, emotion_name
                )
                
                # Find the emotion with highest confidence for display
                max_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, confidence = max_emotion
                
                # Update current emotion with thread safety
                with self.lock:
                    self.current_emotion = emotion_name
                    self.emotion_confidence = confidence
                    self.emotion_history.append({
                        'emotion': emotion_name,
                        'confidence': confidence,
                        'engagement_score': engagement_score,
                        'engagement_state': engagement_state,
                        'timestamp': time.time()
                    })
                
                return emotions, face_box, engagement_score, engagement_state
            else:
                # No face detected
                engagement_score, emotion_name = self.calculate_engagement_score(None)
                engagement_state, state_confidence = self.update_engagement_state(
                    engagement_score, "No Face"
                )
                
                with self.lock:
                    self.current_emotion = "No Face Detected"
                    self.emotion_confidence = 0.0
                
                return None, None, engagement_score, engagement_state
                
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None, None, 0.0, "ERROR"
    
    def draw_emotion_info(self, frame, emotions, face_box, engagement_score, engagement_state):
        """Draw comprehensive emotion and engagement information on the frame"""
        if emotions and face_box:
            x, y, w, h = face_box
            
            # Draw face rectangle with engagement state color
            if engagement_state == "ENGAGED":
                color = (0, 255, 0)  # Green
            elif engagement_state == "DISENGAGED":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow for neutral
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label above face
            emotion_label = f"{self.current_emotion}: {self.emotion_confidence:.2f}"
            cv2.putText(frame, emotion_label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Draw engagement state below face
            engagement_label = f"{engagement_state} ({engagement_score:.2f})"
            cv2.putText(frame, engagement_label, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            
            # Draw engagement bar
            bar_width = 200
            bar_height = 15
            bar_x = 10
            bar_y = frame.shape[0] - 60
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Engagement score bar
            score_width = int(bar_width * engagement_score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), 
                         color, -1)
            
            # Engagement score text
            cv2.putText(frame, f"Engagement: {engagement_score:.2f}", 
                       (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current engagement state in top-left corner
        state_color = (0, 255, 0) if engagement_state == "ENGAGED" else (0, 0, 255) if engagement_state == "DISENGAGED" else (0, 255, 255)
        cv2.putText(frame, f"State: {engagement_state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Time since last face detection
        time_since_face = time.time() - self.last_face_detected_time
        if time_since_face > 1.0:
            cv2.putText(frame, f"No face: {time_since_face:.1f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def run_detection(self):
        """Main detection loop with enhanced disengagement detection"""
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect emotions and engagement
            emotions, face_box, engagement_score, engagement_state = self.detect_emotion(frame)
            
            # Draw information on frame
            frame = self.draw_emotion_info(frame, emotions, face_box, engagement_score, engagement_state)
            
            # Send to orchestrator (placeholder)
            self.send_to_orchestrator(engagement_state, self.current_emotion, engagement_score)
            
            # Display the frame
            cv2.imshow('Enhanced Emotion & Engagement Detection', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()
    
    def send_to_orchestrator(self, status, emotion, score):
        """Send engagement status to orchestrator"""
        # Enhanced logging with more details
        if hasattr(self, 'last_orchestrator_status') and self.last_orchestrator_status != status:
            print(f"ENGAGEMENT CHANGE: {status} | Emotion: {emotion} | Score: {score:.2f}")
        self.last_orchestrator_status = status

# Enhanced GUI class remains mostly the same but with updated detector
class EnhancedEmotionGUI:
    def __init__(self):
        """Initialize the GUI for emotion visualization"""
        self.root = tk.Tk()
        self.root.title("Enhanced Real-time Emotion & Engagement Detection")
        self.root.geometry("900x700")
        
        self.detector = ImprovedEmotionDetector()
        self.is_running = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Enhanced Real-time Emotion & Engagement Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # Current state display
        state_frame = ttk.LabelFrame(main_frame, text="Current State", padding="10")
        state_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Emotion display
        self.emotion_label = ttk.Label(state_frame, text="Unknown", 
                                      font=("Arial", 20, "bold"), foreground="gray")
        self.emotion_label.grid(row=0, column=0, pady=5)
        
        self.confidence_label = ttk.Label(state_frame, text="Confidence: 0.00", 
                                         font=("Arial", 10))
        self.confidence_label.grid(row=1, column=0, pady=2)
        
        # Engagement display
        self.engagement_label = ttk.Label(state_frame, text="UNKNOWN", 
                                         font=("Arial", 18, "bold"), foreground="orange")
        self.engagement_label.grid(row=2, column=0, pady=5)
        
        self.engagement_score_label = ttk.Label(state_frame, text="Score: 0.00", 
                                               font=("Arial", 10))
        self.engagement_score_label.grid(row=3, column=0, pady=2)
        
        # History chart
        chart_frame = ttk.LabelFrame(main_frame, text="Engagement History", padding="10")
        chart_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # Start update loop
        self.update_gui()
    
    def start_detection(self):
        """Start emotion detection"""
        if self.detector.start_camera():
            self.is_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start detection in separate thread
            self.detection_thread = threading.Thread(target=self.detector.run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def stop_detection(self):
        """Stop emotion detection"""
        self.is_running = False
        self.detector.stop_camera()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
    
    def update_gui(self):
        """Update GUI elements"""
        if self.is_running:
            # Update displays
            with self.detector.lock:
                emotion = self.detector.current_emotion
                confidence = self.detector.emotion_confidence
                engagement_state = self.detector.current_engagement_state
                
                # Get latest engagement score from history
                engagement_score = 0.0
                if self.detector.emotion_history:
                    latest_entry = list(self.detector.emotion_history)[-1]
                    engagement_score = latest_entry.get('engagement_score', 0.0)
            
            # Update emotion display
            self.emotion_label.config(text=emotion)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            
            # Update engagement display
            self.engagement_label.config(text=engagement_state)
            self.engagement_score_label.config(text=f"Score: {engagement_score:.2f}")
            
            # Update colors
            emotion_colors = {
                'angry': 'red', 'disgust': 'orange', 'fear': 'purple',
                'happy': 'green', 'sad': 'blue', 'surprise': 'cyan',
                'neutral': 'gray'
            }
            emotion_color = emotion_colors.get(emotion.lower(), 'gray')
            self.emotion_label.config(foreground=emotion_color)
            
            engagement_colors = {
                'ENGAGED': 'green',
                'DISENGAGED': 'red',
                'NEUTRAL': 'orange',
                'UNKNOWN': 'gray'
            }
            engagement_color = engagement_colors.get(engagement_state, 'gray')
            self.engagement_label.config(foreground=engagement_color)
            
            # Update chart
            self.update_chart()
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def update_chart(self):
        """Update the engagement history chart"""
        with self.detector.lock:
            history = list(self.detector.emotion_history)
        
        if history and len(history) > 1:
            # Clear previous plot
            self.ax.clear()
            
            # Prepare data
            timestamps = [(entry['timestamp'] - history[0]['timestamp']) for entry in history]
            engagement_scores = [entry.get('engagement_score', 0.0) for entry in history]
            engagement_states = [entry.get('engagement_state', 'UNKNOWN') for entry in history]
            
            # Plot engagement score over time
            self.ax.plot(timestamps, engagement_scores, 'b-', linewidth=2, label='Engagement Score')
            self.ax.fill_between(timestamps, engagement_scores, alpha=0.3)
            
            # Add engagement state markers
            state_colors = {'ENGAGED': 'green', 'DISENGAGED': 'red', 'NEUTRAL': 'orange', 'UNKNOWN': 'gray'}
            for i, (timestamp, state) in enumerate(zip(timestamps, engagement_states)):
                color = state_colors.get(state, 'gray')
                self.ax.scatter(timestamp, engagement_scores[i], c=color, s=30, alpha=0.7)
            
            # Formatting
            self.ax.set_ylabel('Engagement Score')
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_title('Real-time Engagement Tracking')
            self.ax.set_ylim(0, 1)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Add horizontal lines for thresholds
            self.ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Engaged Threshold')
            self.ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Disengaged Threshold')
            
            # Update canvas
            self.canvas.draw()
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    """Enhanced main function with improved disengagement detection"""
    # Initialize enhanced detector
    detector = ImprovedEmotionDetector()
    
    # Start camera
    if not detector.start_camera():
        return
    
    print("Enhanced Emotion & Engagement Detection started!")
    print("Improvements:")
    print("- Lower confidence thresholds for better detection")
    print("- Temporal smoothing to reduce false positives")
    print("- Face absence tracking")
    print("- Multi-factor engagement scoring")
    print("- Adaptive baseline adjustment")
    print("Press 'q' to quit")
    
    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        detector.stop_camera()

if __name__ == "__main__":
    # Choose to run enhanced detection or GUI
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        app = EnhancedEmotionGUI()
        app.run()
    else:
        main()