import streamlit as st
import time
import base64
from datetime import datetime, timedelta
from enum import Enum

class Phase(Enum):
    WORK = "work"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"

class PomodoroTimer:
    def __init__(self):
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize all session state variables for the timer"""
        defaults = {
            'pomodoro_work_duration': 25,
            'pomodoro_short_break_duration': 5,
            'pomodoro_long_break_duration': 15,
            'pomodoro_long_break_frequency': 4,
            'pomodoro_audio_enabled': True,
            'pomodoro_current_phase': Phase.WORK,
            'pomodoro_current_time': 25 * 60,  # in seconds
            'pomodoro_is_running': False,
            'pomodoro_cycle_count': 0,
            'pomodoro_last_update': time.time(),
            'pomodoro_show_settings': False,
            'pomodoro_phase_changed': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _get_phase_config(self, phase):
        """Get configuration for a specific phase"""
        configs = {
            Phase.WORK: {
                'duration': st.session_state.pomodoro_work_duration * 60,
                'color': '#FF6B6B',
                'icon': '🍅',
                'title': 'Work Time',
                'message': 'Time to focus and be productive!'
            },
            Phase.SHORT_BREAK: {
                'duration': st.session_state.pomodoro_short_break_duration * 60,
                'color': '#4ECDC4',
                'icon': '☕',
                'title': 'Short Break',
                'message': 'Take a quick break and recharge!'
            },
            Phase.LONG_BREAK: {
                'duration': st.session_state.pomodoro_long_break_duration * 60,
                'color': '#45B7D1',
                'icon': '🌟',
                'title': 'Long Break',
                'message': 'Time for a longer rest!'
            }
        }
        return configs[phase]
    
    def _update_timer(self):
        """Update timer logic - non-blocking"""
        if st.session_state.pomodoro_is_running:
            current_time = time.time()
            elapsed = current_time - st.session_state.pomodoro_last_update
            st.session_state.pomodoro_last_update = current_time
            
            # Update remaining time
            st.session_state.pomodoro_current_time -= elapsed
            
            # Check if phase is complete
            if st.session_state.pomodoro_current_time <= 0:
                self._handle_phase_change()
    
    def _handle_phase_change(self):
        """Handle transition between phases"""
        current_phase = st.session_state.pomodoro_current_phase
        
        # Determine next phase
        if current_phase == Phase.WORK:
            st.session_state.pomodoro_cycle_count += 1
            if st.session_state.pomodoro_cycle_count % st.session_state.pomodoro_long_break_frequency == 0:
                next_phase = Phase.LONG_BREAK
            else:
                next_phase = Phase.SHORT_BREAK
        else:
            next_phase = Phase.WORK
        
        # Update phase and reset timer
        st.session_state.pomodoro_current_phase = next_phase
        phase_config = self._get_phase_config(next_phase)
        st.session_state.pomodoro_current_time = phase_config['duration']
        st.session_state.pomodoro_is_running = False
        st.session_state.pomodoro_phase_changed = True
        
        # Show notification
        self._show_phase_notification(next_phase)
    
    def _show_phase_notification(self, phase):
        """Show visual notification for phase change"""
        config = self._get_phase_config(phase)
        
        if phase == Phase.WORK:
            st.success(f"{config['icon']} {config['title']}: {config['message']}")
        elif phase == Phase.SHORT_BREAK:
            st.info(f"{config['icon']} {config['title']}: {config['message']}")
        else:
            st.warning(f"{config['icon']} {config['title']}: {config['message']}")
        
        # Play audio notification if enabled
        if st.session_state.pomodoro_audio_enabled:
            self._play_audio_notification()
    
    def _play_audio_notification(self):
        """Play audio notification using HTML5 audio"""
        # Simple beep sound encoded in base64
        audio_base64 = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmUdBjuN2/PHdSYGKnHC7t6RTwsN"
        
        st.markdown(f"""
        <audio autoplay>
            <source src="{audio_base64}" type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)
    
    def _format_time(self, seconds):
        """Format seconds into MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _render_progress(self):
        """Render progress bar and time display"""
        current_phase = st.session_state.pomodoro_current_phase
        config = self._get_phase_config(current_phase)
        
        # Calculate progress
        total_time = config['duration']
        remaining_time = max(0, st.session_state.pomodoro_current_time)
        progress = (total_time - remaining_time) / total_time if total_time > 0 else 0
        
        # Phase header with icon and color
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: {config['color']}20; border-radius: 10px; margin-bottom: 10px;">
            <h3 style="color: {config['color']}; margin: 0;">{config['icon']} {config['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Time display
        st.markdown(f"""
        <div style="text-align: center; font-size: 2.5em; font-weight: bold; color: {config['color']}; margin: 20px 0;">
            {self._format_time(remaining_time)}
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(progress)
        
        # Cycle counter
        st.markdown(f"""
        <div style="text-align: center; color: #666; margin-top: 10px;">
            🔄 Cycle: {st.session_state.pomodoro_cycle_count}
        </div>
        """, unsafe_allow_html=True)
    
    def _handle_phase_selection(self, selected_phase):
        """Handle manual phase selection from dropdown"""
        if selected_phase != st.session_state.pomodoro_current_phase:
            st.session_state.pomodoro_current_phase = selected_phase
            config = self._get_phase_config(selected_phase)
            st.session_state.pomodoro_current_time = config['duration']
            st.session_state.pomodoro_is_running = False
            st.rerun()
    
    def _render_controls(self):
        """Render control buttons and phase selector"""
        # Phase selector dropdown
        phase_options = {
            "🍅 Work Time": Phase.WORK,
            "☕ Short Break": Phase.SHORT_BREAK,
            "🌟 Long Break": Phase.LONG_BREAK
        }
        
        # Find current phase label
        current_label = None
        for label, phase in phase_options.items():
            if phase == st.session_state.pomodoro_current_phase:
                current_label = label
                break
        
        selected_label = st.selectbox(
            "Select Phase:",
            options=list(phase_options.keys()),
            index=list(phase_options.keys()).index(current_label) if current_label else 0,
            key="phase_selector"
        )
        
        # Handle phase change
        selected_phase = phase_options[selected_label]
        self._handle_phase_selection(selected_phase)
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.pomodoro_is_running:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.pomodoro_is_running = False
            else:
                if st.button("▶ Start", use_container_width=True):
                    st.session_state.pomodoro_is_running = True
                    st.session_state.pomodoro_last_update = time.time()
        
        with col2:
            if st.button("⏹ Reset", use_container_width=True):
                current_phase = st.session_state.pomodoro_current_phase
                config = self._get_phase_config(current_phase)
                st.session_state.pomodoro_current_time = config['duration']
                st.session_state.pomodoro_is_running = False
    
    def _render_settings(self):
        """Render settings panel"""
        if st.button("⚙ Settings", use_container_width=True):
            st.session_state.pomodoro_show_settings = not st.session_state.pomodoro_show_settings
        
        if st.session_state.pomodoro_show_settings:
            with st.expander("Timer Settings", expanded=True):
                # Duration settings
                st.session_state.pomodoro_work_duration = st.slider(
                    "Work Duration (minutes)", 1, 60, st.session_state.pomodoro_work_duration
                )
                
                st.session_state.pomodoro_short_break_duration = st.slider(
                    "Short Break (minutes)", 1, 30, st.session_state.pomodoro_short_break_duration
                )
                
                st.session_state.pomodoro_long_break_duration = st.slider(
                    "Long Break (minutes)", 1, 60, st.session_state.pomodoro_long_break_duration
                )
                
                st.session_state.pomodoro_long_break_frequency = st.slider(
                    "Long break every N cycles", 2, 10, st.session_state.pomodoro_long_break_frequency
                )
                
                # Audio setting
                st.session_state.pomodoro_audio_enabled = st.checkbox(
                    "🔊 Audio notifications", st.session_state.pomodoro_audio_enabled
                )
                
                # Reset all button
                if st.button("🔄 Reset All Settings"):
                    # Reset to defaults
                    st.session_state.pomodoro_work_duration = 25
                    st.session_state.pomodoro_short_break_duration = 5
                    st.session_state.pomodoro_long_break_duration = 15
                    st.session_state.pomodoro_long_break_frequency = 4
                    st.session_state.pomodoro_cycle_count = 0
                    st.session_state.pomodoro_current_phase = Phase.WORK
                    st.session_state.pomodoro_current_time = 25 * 60
                    st.session_state.pomodoro_is_running = False
                    st.rerun()
    
    def render_sidebar(self):
        """Main method to render the complete timer in sidebar"""
        # Update timer logic
        self._update_timer()
        
        # Render components
        st.markdown("## 🍅 Pomodoro Timer")
        
        self._render_progress()
        st.markdown("---")
        
        self._render_controls()
        st.markdown("---")
        
        self._render_settings()
        
        # Auto-refresh every second when running
        if st.session_state.pomodoro_is_running:
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            st.rerun()
        
        # Handle phase change notifications
        if st.session_state.pomodoro_phase_changed:
            st.session_state.pomodoro_phase_changed = False
            st.rerun()

# Example usage in your main app:
"""
# In your main Streamlit app file:

from pomodoro_timer import PomodoroTimer

# Initialize timer
if 'timer' not in st.session_state:
    st.session_state.timer = PomodoroTimer()

# In your sidebar
with st.sidebar:
    st.session_state.timer.render_sidebar()

# Your main app content continues here...
"""