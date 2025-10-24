#!/usr/bin/env python3
"""
Advanced Gesture Recognition Sign Language to Text Application
Improved recognition for all letters A-Z
"""

import numpy as np
import cv2
import os
import sys
import time
import operator
from string import ascii_uppercase
import math

import tkinter as tk
from PIL import Image, ImageTk

import enchant
import pyttsx3

class AdvancedGestureRecognitionApp:
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
        
        # Character counters
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Advanced gesture recognition parameters
        self.gesture_history = []
        self.history_length = 8
        self.confidence_threshold = 0.6
        
        # Hand landmark detection parameters
        self.hand_landmarks = None
        self.previous_landmarks = None
        
        print("Advanced Gesture Recognition App initialized!")

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Advanced Gesture Recognition Sign Language To Text")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1200x900")

        # Main title
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Advanced Gesture Recognition Sign Language To Text", font=("Courier", 20, "bold"))

        # Camera control section
        self.camera_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.camera_frame.place(x=20, y=50, width=700, height=400)
        
        self.camera_label = tk.Label(self.camera_frame, text="Advanced Camera Control", font=("Courier", 16, "bold"))
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
        self.panel.place(x=10, y=110, width=330, height=250)
        
        self.panel2 = tk.Label(self.camera_frame, bg="black", text="Processed Image\n(Click Start Camera)", 
                              font=("Courier", 12), fg="white")
        self.panel2.place(x=350, y=110, width=330, height=250)

        # Text output section
        self.text_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.text_frame.place(x=20, y=470, width=700, height=200)
        
        self.text_label = tk.Label(self.text_frame, text="Text Output", font=("Courier", 16, "bold"))
        self.text_label.place(x=10, y=10)

        self.panel3 = tk.Label(self.text_frame)
        self.panel3.place(x=200, y=50)

        self.T1 = tk.Label(self.text_frame)
        self.T1.place(x=10, y=50)
        self.T1.config(text="Character:", font=("Courier", 16, "bold"))

        self.panel4 = tk.Label(self.text_frame)
        self.panel4.place(x=200, y=80)

        self.T2 = tk.Label(self.text_frame)
        self.T2.place(x=10, y=80)
        self.T2.config(text="Word:", font=("Courier", 16, "bold"))

        self.panel5 = tk.Label(self.text_frame)
        self.panel5.place(x=200, y=110)

        self.T3 = tk.Label(self.text_frame)
        self.T3.place(x=10, y=110)
        self.T3.config(text="Sentence:", font=("Courier", 16, "bold"))

        # Control buttons section
        self.control_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.control_frame.place(x=750, y=50, width=400, height=400)
        
        self.control_label = tk.Label(self.control_frame, text="Advanced Controls", font=("Courier", 16, "bold"))
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
        
        # Confidence display
        self.confidence_label = tk.Label(self.control_frame, text="Confidence: --", 
                                       font=("Courier", 10), fg="blue")
        self.confidence_label.place(x=10, y=250)
        
        # Current gesture display
        self.gesture_label = tk.Label(self.control_frame, text="Current Gesture: --", 
                                    font=("Courier", 10), fg="green")
        self.gesture_label.place(x=10, y=280)
        
        # Settings
        self.settings_label = tk.Label(self.control_frame, text="Recognition Settings:", font=("Courier", 12, "bold"))
        self.settings_label.place(x=10, y=310)
        
        # Confidence threshold slider
        self.confidence_var = tk.DoubleVar(value=0.6)
        self.confidence_scale = tk.Scale(self.control_frame, from_=0.3, to=0.9, resolution=0.1,
                                       orient=tk.HORIZONTAL, variable=self.confidence_var,
                                       command=self.update_confidence_threshold,
                                       label="Confidence Threshold", length=350)
        self.confidence_scale.place(x=10, y=340)

        # Suggestions section
        self.suggestions_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        self.suggestions_frame.place(x=20, y=680, width=700, height=100)
        
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
            
            # Hand detection region
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

            # Advanced gesture recognition
            confidence, gesture = self.advanced_gesture_recognition(roi_image)

            # Show processed image
            processed_image = self.preprocess_image(roi_image)
            self.current_image2 = Image.fromarray(processed_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk, text="")

            # Update displays
            self.panel3.config(text=self.current_symbol, font=("Courier", 20))
            self.panel4.config(text=self.word, font=("Courier", 20))
            self.panel5.config(text=self.str, font=("Courier", 20))

            # Update confidence and gesture displays
            if confidence is not None:
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            else:
                self.confidence_label.config(text="Confidence: --")
            
            if gesture:
                self.gesture_label.config(text=f"Current Gesture: {gesture}")
            else:
                self.gesture_label.config(text="Current Gesture: --")

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

    def preprocess_image(self, roi_image):
        """Enhanced image preprocessing"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh

    def advanced_gesture_recognition(self, roi_image):
        """Advanced gesture recognition using multiple features"""
        processed_image = self.preprocess_image(roi_image)
        
        # Extract multiple features
        features = self.extract_hand_features(processed_image)
        
        if features is None:
            return None, None
        
        # Classify gesture based on features
        gesture, confidence = self.classify_gesture(features)
        
        # Add to gesture history for temporal smoothing
        self.gesture_history.append((gesture, confidence))
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        # Apply temporal smoothing
        if len(self.gesture_history) >= 5:
            smoothed_gesture = self.apply_temporal_smoothing()
            if smoothed_gesture and smoothed_gesture != 'blank':
                self.current_symbol = smoothed_gesture
                self.update_character_counters()
                return confidence, smoothed_gesture
        
        return confidence, gesture

    def extract_hand_features(self, processed_image):
        """Extract comprehensive hand features"""
        try:
            # Find contours
            contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return None
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < 1000:  # Too small to be a hand
                return None
            
            # Calculate various features
            features = {}
            
            # Basic shape features
            features['area'] = area
            features['perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['extent'] = area / (w * h) if w * h > 0 else 0
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / hull_area if hull_area > 0 else 0
            
            # Convexity defects
            hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
            if len(hull_indices) > 3:
                defects = cv2.convexityDefects(largest_contour, hull_indices)
                if defects is not None:
                    features['num_defects'] = len(defects)
                else:
                    features['num_defects'] = 0
            else:
                features['num_defects'] = 0
            
            # Moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                features['centroid_x'] = moments['m10'] / moments['m00']
                features['centroid_y'] = moments['m01'] / moments['m00']
            else:
                features['centroid_x'] = 0
                features['centroid_y'] = 0
            
            # Hu moments for shape description
            hu_moments = cv2.HuMoments(moments).flatten()
            for i, hu in enumerate(hu_moments):
                features[f'hu_moment_{i}'] = hu
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def classify_gesture(self, features):
        """Classify gesture based on extracted features"""
        try:
            # Rule-based classification for different letters
            area = features['area']
            num_defects = features['num_defects']
            aspect_ratio = features['aspect_ratio']
            solidity = features['solidity']
            extent = features['extent']
            
            # Classification rules based on hand shape characteristics
            if area < 2000:
                return 'blank', 0.5
            
            # Letters with no finger separation (closed fist)
            if num_defects <= 1 and solidity > 0.85:
                if aspect_ratio > 1.2:
                    return 'I', 0.8
                elif aspect_ratio < 0.8:
                    return 'O', 0.8
                else:
                    return 'A', 0.7
            
            # Letters with 1-2 finger separations
            elif num_defects <= 3:
                if aspect_ratio > 1.3:
                    return 'L', 0.8
                elif aspect_ratio < 0.7:
                    return 'U', 0.8
                else:
                    return 'B', 0.7
            
            # Letters with 3-4 finger separations
            elif num_defects <= 5:
                if aspect_ratio > 1.1:
                    return 'V', 0.8
                elif aspect_ratio < 0.9:
                    return 'W', 0.8
                else:
                    return 'C', 0.7
            
            # Letters with many finger separations
            elif num_defects > 5:
                if aspect_ratio > 1.0:
                    return 'Y', 0.8
                else:
                    return 'X', 0.7
            
            # Default classification based on area and shape
            if area > 8000:
                return 'A', 0.6
            elif area > 6000:
                return 'B', 0.6
            elif area > 4000:
                return 'C', 0.6
            elif area > 2000:
                return 'D', 0.6
            else:
                return 'E', 0.6
                
        except Exception as e:
            print(f"Error classifying gesture: {e}")
            return 'blank', 0.0

    def apply_temporal_smoothing(self):
        """Apply temporal smoothing to reduce flickering"""
        if len(self.gesture_history) < 5:
            return None
        
        # Count occurrences of each gesture
        gesture_counts = {}
        for gesture, confidence in self.gesture_history:
            if confidence > self.confidence_threshold:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        if not gesture_counts:
            return None
        
        # Find the most frequent gesture
        most_frequent = max(gesture_counts.items(), key=lambda x: x[1])
        
        # Only return if it appears enough times
        if most_frequent[1] >= 3:
            return most_frequent[0]
        
        return None

    def update_character_counters(self):
        """Update character counters with enhanced logic"""
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
            return

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 25:  # Reduced threshold for faster response
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 10:  # Reduced threshold
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
    print("Starting Advanced Gesture Recognition App...")
    app = AdvancedGestureRecognitionApp()
    app.root.mainloop()
