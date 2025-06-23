#!/usr/bin/env python3
"""
Test script for disengagement popup notifications
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time

class TestNotifier:
    def __init__(self):
        self.root = None
        self.last_notification_time = 0
        self.notification_cooldown = 5  # seconds between notifications
        self.is_initialized = False
    
    def initialize_tkinter(self):
        """Initialize tkinter root window (hidden)"""
        if not self.is_initialized:
            try:
                self.root = tk.Tk()
                self.root.withdraw()  # Hide the main window
                self.is_initialized = True
                print("✅ Tkinter initialized successfully")
            except Exception as e:
                print(f"❌ Could not initialize notification system: {e}")
    
    def show_disengagement_popup(self, message="Student appears disengaged!"):
        """Show a pop-up notification for disengagement"""
        current_time = time.time()
        
        # Check cooldown to avoid spam
        if current_time - self.last_notification_time < self.notification_cooldown:
            print(f"⏳ Cooldown active, skipping notification")
            return
        
        try:
            self.initialize_tkinter()
            
            if self.root:
                # Show pop-up using root.after to ensure main thread execution
                def show_popup():
                    try:
                        result = messagebox.showwarning(
                            "⚠️ Disengagement Detected",
                            f"{message}\n\nConsider:\n• Simplifying explanations\n• Adding visual aids\n• Taking a short break\n• Checking if the student needs help"
                        )
                        print(f"✅ Popup shown successfully, result: {result}")
                    except Exception as e:
                        print(f"❌ Popup error: {e}")
                
                # Schedule popup in main thread
                self.root.after(100, show_popup)
                
                self.last_notification_time = current_time
                print(f"📢 Notification triggered: {message}")
                
        except Exception as e:
            print(f"❌ Could not show popup: {e}")
    
    def cleanup(self):
        """Clean up tkinter resources"""
        if self.root:
            try:
                self.root.destroy()
                print("✅ Tkinter resources cleaned up")
            except:
                print("⚠️ Error during cleanup")

def main():
    """Test the notification system"""
    print("🧪 Testing Disengagement Notification System")
    print("=" * 50)
    
    notifier = TestNotifier()
    
    try:
        # Initialize tkinter
        notifier.initialize_tkinter()
        
        # Test 1: Basic notification
        print("\n1️⃣ Testing basic notification...")
        notifier.show_disengagement_popup("This is a test notification!")
        
        # Process tkinter events
        if notifier.root:
            notifier.root.update()
        
        time.sleep(2)
        
        # Test 2: Notification with custom message
        print("\n2️⃣ Testing custom message...")
        notifier.show_disengagement_popup("Student has been disengaged for 15.5 seconds!")
        
        # Process tkinter events
        if notifier.root:
            notifier.root.update()
        
        time.sleep(2)
        
        # Test 3: Test cooldown (should be skipped)
        print("\n3️⃣ Testing cooldown (should be skipped)...")
        notifier.show_disengagement_popup("This should be skipped due to cooldown!")
        
        # Process tkinter events
        if notifier.root:
            notifier.root.update()
        
        time.sleep(2)
        
        # Test 4: Test after cooldown
        print("\n4️⃣ Testing after cooldown...")
        time.sleep(6)  # Wait for cooldown to expire
        notifier.show_disengagement_popup("This should work after cooldown!")
        
        # Process tkinter events
        if notifier.root:
            notifier.root.update()
        
        print("\n✅ All tests completed!")
        print("Check for popup windows that should have appeared.")
        
        # Keep window open briefly to see results
        if notifier.root:
            notifier.root.after(3000, notifier.root.destroy)  # Close after 3 seconds
            notifier.root.mainloop()
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        notifier.cleanup()

if __name__ == "__main__":
    main() 