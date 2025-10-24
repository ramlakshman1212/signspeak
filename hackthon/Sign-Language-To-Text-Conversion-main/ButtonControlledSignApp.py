#!/usr/bin/env python3
"""
Button Controlled Sign Language to Text Application
Camera only starts when button is pressed
"""

import numpy as np
import cv2
import os
import sys
import time
import operator
from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

import enchant
import pyttsx3

class ButtonControlledSignApp:
    def __init__(self):
        self.hs = enchant.Dict("en_US")
        
        # Initialize text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            self.voice_enabled = True
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.voice_enabled = False
        
        # Camera and video processing variables
        self.vs = None
        self.current_image = None
        self.current_image2 = None
        self.camera_active = False
        
        # Simple gesture recognition without ML models
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Gesture recognition parameters
        self.gesture_history = []
        self.history_length = 5
        
        print("Button Controlled Sign Language App initialized!")

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Button Controlled Sign Language To Text")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1000x800")

        # Main title
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Button Controlled Sign Language To Text", font=("Courier", 25, "bold"))

        # Camera control section
        self.camera_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.camera_frame.place(x=20, y=50, width=600, height=400)
        
        self.camera_label = tk.Label(self.camera_frame, text="Camera Control", font=("Courier", 16, "bold"))
        self.camera_label.place(x=10, y=10)
        
        # Camera status
        self.camera_status = tk.Label(self.camera_frame, text="Camera: OFF", font=("Courier", 12), fg="red")
        self.camera_status.place(x=10, y=40)
        
        # Start/Stop camera buttons
        self.start_camera_btn = tk.Button(self.camera_frame, text="Start Camera", 
                                        command=self.start_camera, font=("Courier", 14), 
                                        bg="lightgreen", width=15)
        self.start_camera_btn.place(x=10, y=70)
        
        self.stop_camera_btn = tk.Button(self.camera_frame, text="Stop Camera", 
                                       command=self.stop_camera, font=("Courier", 14), 
                                       bg="lightcoral", width=15, state=tk.DISABLED)
        self.stop_camera_btn.place(x=200, y=70)
        
        # Camera display areas
        self.panel = tk.Label(self.camera_frame, bg="black", text="Camera Feed\n(Click Start Camera)", 
                            font=("Courier", 12), fg="white")
        self.panel.place(x=10, y=110, width=280, height=200)
        
        self.panel2 = tk.Label(self.camera_frame, bg="black", text="Processed Image\n(Click Start Camera)", 
                              font=("Courier", 12), fg="white")
        self.panel2.place(x=300, y=110, width=280, height=200)

        # Text output section
        self.text_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.text_frame.place(x=20, y=470, width=600, height=300)
        
        self.text_label = tk.Label(self.text_frame, text="Text Output", font=("Courier", 16, "bold"))
        self.text_label.place(x=10, y=10)

        self.panel3 = tk.Label(self.text_frame)
        self.panel3.place(x=200, y=50)

        self.T1 = tk.Label(self.text_frame)
        self.T1.place(x=10, y=50)
        self.T1.config(text="Character:", font=("Courier", 20, "bold"))

        self.panel4 = tk.Label(self.text_frame)
        self.panel4.place(x=200, y=90)

        self.T2 = tk.Label(self.text_frame)
        self.T2.place(x=10, y=90)
        self.T2.config(text="Word:", font=("Courier", 20, "bold"))

        self.panel5 = tk.Label(self.text_frame)
        self.panel5.place(x=200, y=130)

        self.T3 = tk.Label(self.text_frame)
        self.T3.place(x=10, y=130)
        self.T3.config(text="Sentence:", font=("Courier", 20, "bold"))

        # Control buttons section
        self.control_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.control_frame.place(x=650, y=50, width=320, height=400)
        
        self.control_label = tk.Label(self.control_frame, text="Controls", font=("Courier", 16, "bold"))
        self.control_label.place(x=10, y=10)
        
        # Voice control
        self.voice_btn = tk.Button(self.control_frame, text="Speak Text", command=self.speak_text, 
                                  font=("Courier", 12), bg="lightblue", width=20)
        self.voice_btn.place(x=10, y=50)
        
        # Clear text
        self.clear_btn = tk.Button(self.control_frame, text="Clear All Text", command=self.clear_text, 
                                 font=("Courier", 12), bg="lightcoral", width=20)
        self.clear_btn.place(x=10, y=90)
        
        # Manual input
        self.manual_label = tk.Label(self.control_frame, text="Manual Input:", font=("Courier", 12, "bold"))
        self.manual_label.place(x=10, y=130)
        
        self.manual_input = tk.Entry(self.control_frame, font=("Courier", 12), width=25)
        self.manual_input.place(x=10, y=155)
        self.manual_input.bind('<Return>', self.add_manual_text)
        
        self.manual_btn = tk.Button(self.control_frame, text="Add Text", command=self.add_manual_text, 
                                  font=("Courier", 10), bg="lightgreen", width=15)
        self.manual_btn.place(x=10, y=185)
        
        # Gesture recognition info
        self.info_label = tk.Label(self.control_frame, text="Gesture Recognition: OFF", 
                                 font=("Courier", 10), fg="red")
        self.info_label.place(x=10, y=220)
        
        # Confidence threshold
        self.confidence_label = tk.Label(self.control_frame, text="Confidence: --", 
                                       font=("Courier", 10), fg="blue")
        self.confidence_label.place(x=10, y=250)
        
        # Settings
        self.settings_label = tk.Label(self.control_frame, text="Settings:", font=("Courier", 12, "bold"))
        self.settings_label.place(x=10, y=280)
        
        # Confidence threshold slider
        self.confidence_var = tk.DoubleVar(value=0.7)
        self.confidence_scale = tk.Scale(self.control_frame, from_=0.5, to=0.9, resolution=0.1,
                                       orient=tk.HORIZONTAL, variable=self.confidence_var,
                                       command=self.update_confidence_threshold,
                                       label="Confidence Threshold", length=250)
        self.confidence_scale.place(x=10, y=310)

        # Suggestions section
        self.suggestions_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.suggestions_frame.place(x=20, y=780, width=600, height=100)
        
        self.T4 = tk.Label(self.suggestions_frame)
        self.T4.place(x=10, y=10)
        self.T4.config(text="Suggestions:", fg="red", font=("Courier", 16, "bold"))

        self.bt1 = tk.Button(self.suggestions_frame, command=self.action1, height=1, width=15)
        self.bt1.place(x=10, y=40)

        self.bt2 = tk.Button(self.suggestions_frame, command=self.action2, height=1, width=15)
        self.bt2.place(x=200, y=40)

        self.bt3 = tk.Button(self.suggestions_frame, command=self.action3, height=1, width=15)
        self.bt3.place(x=390, y=40)

        # Initialize text variables
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.confidence_threshold = 0.7

    def start_camera(self):
        """Start the camera when button is pressed"""
        print("Starting camera...")
        try:
            self.vs = cv2.VideoCapture(0)
            if not self.vs.isOpened():
                print("Error: Could not open camera!")
                self.camera_status.config(text="Camera: ERROR", fg="red")
                return
            
            self.camera_active = True
            self.camera_status.config(text="Camera: ON", fg="green")
            self.start_camera_btn.config(state=tk.DISABLED)
            self.stop_camera_btn.config(state=tk.NORMAL)
            self.info_label.config(text="Gesture Recognition: ON", fg="green")
            
            # Start video loop
            self.video_loop()
            print("Camera started successfully!")
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.camera_status.config(text="Camera: ERROR", fg="red")

    def stop_camera(self):
        """Stop the camera when button is pressed"""
        print("Stopping camera...")
        self.camera_active = False
        
        if self.vs:
            self.vs.release()
            self.vs = None
        
        self.camera_status.config(text="Camera: OFF", fg="red")
        self.start_camera_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.DISABLED)
        self.info_label.config(text="Gesture Recognition: OFF", fg="red")
        
        # Clear camera displays
        self.panel.config(image="", text="Camera Feed\n(Click Start Camera)", fg="white")
        self.panel2.config(image="", text="Processed Image\n(Click Start Camera)", fg="white")
        
        print("Camera stopped!")

    def video_loop(self):
        """Video processing loop - only runs when camera is active"""
        if not self.camera_active or not self.vs:
            return
            
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)
            
            # Simple hand detection using center region
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 255, 0), 2)
            cv2.putText(cv2image, "Hand Detection Area", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk, text="")

            # Extract ROI for processing
            roi_image = cv2image[y1:y2, x1:x2]

            # Simple gesture recognition
            confidence = self.simple_gesture_recognition(roi_image)

            # Show processed image
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            self.current_image2 = Image.fromarray(gray)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk, text="")

            # Update displays
            self.panel3.config(text=self.current_symbol, font=("Courier", 20))
            self.panel4.config(text=self.word, font=("Courier", 20))
            self.panel5.config(text=self.str, font=("Courier", 20))

            # Update confidence display
            if confidence is not None:
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            else:
                self.confidence_label.config(text="Confidence: --")

            # Update suggestions
            predicts = self.hs.suggest(self.word)
            
            if len(predicts) > 1:
                self.bt1.config(text=predicts[0], font=("Courier", 12))
            else:
                self.bt1.config(text="")

            if len(predicts) > 2:
                self.bt2.config(text=predicts[1], font=("Courier", 12))
            else:
                self.bt2.config(text="")

            if len(predicts) > 3:
                self.bt3.config(text=predicts[2], font=("Courier", 12))
            else:
                self.bt3.config(text="")

        # Continue loop if camera is still active
        if self.camera_active:
            self.root.after(5, self.video_loop)

    def simple_gesture_recognition(self, roi_image):
        """Simple gesture recognition based on hand shape analysis"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Apply threshold
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        confidence = 0.0
        
        if len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Simple gesture classification based on area and shape
            if area > 5000:  # Large area - might be open hand
                gesture = "A"  # Open hand gesture
                confidence = 0.8
            elif area > 3000:  # Medium area
                gesture = "B"  # Closed fist
                confidence = 0.7
            elif area > 1000:  # Small area
                gesture = "C"  # Pointing gesture
                confidence = 0.6
            else:
                gesture = "blank"
                confidence = 0.5
        else:
            gesture = "blank"
            confidence = 0.5
        
        # Add to gesture history
        self.gesture_history.append((gesture, confidence))
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        # Use most frequent gesture in recent history
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for g, c in self.gesture_history:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            most_frequent = max(gesture_counts.items(), key=lambda x: x[1])
            if most_frequent[1] >= 2:  # Need at least 2 occurrences
                self.current_symbol = most_frequent[0]
                self.update_character_counters()
                return confidence
        
        return confidence

    def update_character_counters(self):
        """Update character counters"""
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
            return

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 30:  # Reduced threshold for faster response
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 15:  # Reduced threshold
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 50:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def speak_text(self):
        """Convert current text to speech"""
        if self.voice_enabled and self.str.strip():
            try:
                self.tts_engine.say(self.str)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def clear_text(self):
        """Clear all text"""
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        
        # Reset counters
        self.ct['blank'] = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Reset gesture history
        self.gesture_history = []
    
    def add_manual_text(self, event=None):
        """Add text manually"""
        manual_text = self.manual_input.get().strip()
        if manual_text:
            if self.str:
                self.str += " " + manual_text
            else:
                self.str = manual_text
            self.manual_input.delete(0, tk.END)
            print(f"Added manual text: {manual_text}")

    def update_confidence_threshold(self, value):
        """Update confidence threshold from slider"""
        self.confidence_threshold = float(value)
        print(f"Confidence threshold updated to: {self.confidence_threshold}")

    def action1(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]

    def action2(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def destructor(self):
        print("Closing Application...")
        self.camera_active = False
        if self.vs:
            self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Button Controlled Sign Language App...")
    app = ButtonControlledSignApp()
    app.root.mainloop()
