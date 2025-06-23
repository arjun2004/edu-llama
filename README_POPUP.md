# Disengagement Pop-up Notification System

## Overview

The AI Learning Assistant now includes a real-time disengagement detection system with pop-up notifications to alert educators when students appear disengaged during learning sessions.

## Features

### 🎯 Real-time Disengagement Detection

- **Emotion Analysis**: Monitors facial expressions for signs of disengagement
- **Face Detection**: Tracks when students are not looking at the camera
- **Engagement Scoring**: Calculates engagement levels based on multiple factors
- **Temporal Smoothing**: Reduces false positives with intelligent filtering

### 🔔 Pop-up Notifications

- **Automatic Alerts**: Shows pop-up windows when disengagement is detected
- **Contextual Messages**: Provides specific information about disengagement patterns
- **Cooldown System**: Prevents notification spam with configurable delays
- **Educational Suggestions**: Includes tips for re-engaging students

### 📊 Engagement Monitoring Dashboard

- **Real-time Status**: Shows current engagement state in the sidebar
- **Detailed Metrics**: Displays engagement score, emotion, and face detection status
- **Historical Tracking**: Monitors disengagement duration and frequency
- **Visual Indicators**: Progress bars and color-coded status indicators

## How It Works

### Detection Algorithm

1. **Face Detection**: Uses OpenCV to detect faces in the camera feed
2. **Emotion Analysis**: FER (Facial Emotion Recognition) analyzes expressions
3. **Engagement Scoring**: Combines emotion data with eye contact analysis
4. **State Classification**: Categorizes as ENGAGED, DISENGAGED, or NEUTRAL
5. **Temporal Analysis**: Tracks patterns over time to reduce false positives

### Notification Triggers

- **Duration-based**: Alerts when disengaged for >10 seconds
- **Frequency-based**: Alerts after 3+ disengagement events
- **Face absence**: Alerts when no face detected for >3 seconds

### Pop-up Content

```
⚠️ Disengagement Detected

[Contextual message based on detection type]

Consider:
• Simplifying explanations
• Adding visual aids
• Taking a short break
• Checking if the student needs help
```

## Configuration

### Sidebar Settings

- **Enable Disengagement Alerts**: Toggle pop-up notifications on/off
- **Test Button**: Manually trigger notifications for testing
- **Real-time Metrics**: View current engagement status and scores

### Detection Parameters

```python
# Configurable thresholds in cv.py
disengaged_duration_limit = 10  # seconds
disengaged_count_limit = 3      # events
no_face_threshold = 3.0         # seconds
notification_cooldown = 10      # seconds between alerts
```

## Usage

### Starting the System

1. Run the main application: `streamlit run app.py`
2. The emotion detection thread starts automatically
3. Camera feed begins monitoring for engagement

### Monitoring Engagement

1. **Sidebar Dashboard**: Check real-time engagement status
2. **Status Bar**: View engagement state at the bottom
3. **Pop-up Alerts**: Automatic notifications when disengagement detected

### Testing Notifications

1. Enable "Disengagement Alerts" in sidebar
2. Click "🧪 Test Disengagement Alert" button
3. Verify pop-up appears with test message

## Technical Implementation

### Key Components

#### `DisengagementNotifier` Class (`app.py`)

- Manages tkinter pop-up windows
- Handles notification cooldowns
- Thread-safe pop-up display

#### `ImprovedEmotionDetector` Class (`cv.py`)

- Real-time emotion analysis
- Engagement state calculation
- Shared state management

#### `shared_engagement_state` Dictionary

- Thread-safe state sharing between components
- Real-time metrics storage
- Configuration parameters

### Thread Safety

- Emotion detection runs in separate thread
- Pop-up notifications use dedicated threads
- Shared state protected with proper synchronization

### Error Handling

- Graceful degradation if camera unavailable
- Fallback mechanisms for detection failures
- Robust pop-up system with error recovery

## Testing

### Manual Testing

```bash
# Test pop-up system independently
python test_popup.py
```

### Integration Testing

1. Start the main application
2. Enable notifications in sidebar
3. Use test button to verify functionality
4. Monitor real-time engagement metrics

## Troubleshooting

### Common Issues

#### Pop-ups Not Appearing

- Check if notifications are enabled in sidebar
- Verify tkinter is available on your system
- Check console for error messages

#### False Positives

- Adjust detection thresholds in `cv.py`
- Increase temporal smoothing parameters
- Check camera positioning and lighting

#### Performance Issues

- Reduce camera resolution if needed
- Adjust detection frequency
- Monitor system resources

### Debug Information

- Console logs show detection events
- Sidebar displays real-time metrics
- Test button provides immediate feedback

## Future Enhancements

### Planned Features

- **Customizable Thresholds**: User-adjustable detection sensitivity
- **Notification History**: Log of all disengagement events
- **Advanced Analytics**: Detailed engagement reports
- **Integration APIs**: Connect with other educational tools

### Potential Improvements

- **Machine Learning**: Adaptive threshold adjustment
- **Multi-person Detection**: Support for multiple students
- **Mobile Support**: Smartphone camera integration
- **Cloud Analytics**: Remote monitoring capabilities

## Dependencies

### Required Packages

```
streamlit
opencv-python
fer
tkinter (usually included with Python)
numpy
matplotlib
```

### Optional Enhancements

```
pyttsx3 (for voice feedback)
speech_recognition (for voice input)
```

## Support

For issues or questions about the disengagement notification system:

1. Check the troubleshooting section
2. Review console logs for error messages
3. Test with the provided test script
4. Verify all dependencies are installed

---

**Note**: This system is designed for educational use and should be used responsibly with appropriate privacy considerations for students.
