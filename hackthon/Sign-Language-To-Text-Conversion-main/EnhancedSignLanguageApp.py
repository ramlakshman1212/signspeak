# Enhanced Real-Time Sign Language to Text and Voice Recognition App
# Features: MediaPipe hand landmarks, mic-style button, spell correction, TTS

import numpy as np
import cv2
import os
import sys
import time
import operator
import threading
from string import ascii_uppercase

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Try to import MediaPipe with error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"MediaPipe import failed: {e}")
    print("Falling back to OpenCV-based hand detection")
    MEDIAPIPE_AVAILABLE = False
    mp = None

import tensorflow as tf
try:
    from tensorflow.keras.models import model_from_json
except ImportError:
    from keras.models import model_from_json

# Text processing and spell correction
import enchant
from textblob import TextBlob
import pyttsx3

# For better spell correction
try:
    import symspellpy
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    print("SymSpell not available. Using basic spell correction.")

class EnhancedSignLanguageApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Sign Language Recognition App")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.setup_mediapipe()
        self.setup_models()
        self.setup_tts()
        self.setup_spell_correction()
        self.setup_camera()
        self.setup_gui()
        
        # Detection state
        self.is_detecting = False
        self.detected_letters = []
        self.prediction_history = []
        self.confidence_threshold = 0.8
        self.history_length = 5
        self.stable_count_threshold = 3
        
        # Start camera loop
        self.video_loop()
        
    def setup_mediapipe(self):
        """Initialize MediaPipe for hand detection and landmarks"""
        if not MEDIAPIPE_AVAILABLE:
            print("WARNING: MediaPipe not available, using OpenCV fallback")
            self.mp_hands = None
            self.hands = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            return
            
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,  # Focus on single hand
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("SUCCESS: MediaPipe initialized successfully!")
        except Exception as e:
            print(f"ERROR: MediaPipe setup failed: {e}")
            self.mp_hands = None
            self.hands = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
        
    def setup_models(self):
        """Load the trained models"""
        try:
            # Load main model
            with open("Models/model_new.json", "r") as json_file:
                model_json = json_file.read()
            self.loaded_model = model_from_json(model_json)
            self.loaded_model.load_weights("Models/model_new.h5")
            
            # Load specialized models
            with open("Models/model-bw_dru.json", "r") as json_file:
                model_json = json_file.read()
            self.loaded_model_dru = model_from_json(model_json)
            self.loaded_model_dru.load_weights("Models/model-bw_dru.h5")
            
            with open("Models/model-bw_tkdi.json", "r") as json_file:
                model_json = json_file.read()
            self.loaded_model_tkdi = model_from_json(model_json)
            self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi.h5")
            
            with open("Models/model-bw_smn.json", "r") as json_file:
                model_json = json_file.read()
            self.loaded_model_smn = model_from_json(model_json)
            self.loaded_model_smn.load_weights("Models/model-bw_smn.h5")
            
            print("SUCCESS: All models loaded successfully!")
            
        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
            self.show_error("Model loading failed. Please check if model files exist.")
            
    def setup_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
            
            self.voice_enabled = True
            print("SUCCESS: TTS engine initialized!")
            
        except Exception as e:
            print(f"ERROR: TTS initialization failed: {e}")
            self.voice_enabled = False
            
    def setup_spell_correction(self):
        """Initialize spell correction systems"""
        try:
            # Basic enchant dictionary
            self.enchant_dict = enchant.Dict("en_US")
            
            # Advanced SymSpell if available
            if SYMSPELL_AVAILABLE:
                self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = "frequency_dictionary_en_82_765.txt"
                if os.path.exists(dictionary_path):
                    self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                    print("SUCCESS: SymSpell dictionary loaded!")
                else:
                    print("WARNING: SymSpell dictionary not found, using basic correction")
                    
            print("SUCCESS: Spell correction initialized!")
            
        except Exception as e:
            print(f"ERROR: Spell correction setup failed: {e}")
            
    def setup_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Could not open camera!")
            return
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("SUCCESS: Camera initialized!")
        
    def setup_gui(self):
        """Create the main GUI interface"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎯 Enhanced Sign Language Recognition", 
            font=("Arial", 20, "bold"),
            bg='#2c3e50', 
            fg='#ecf0f1'
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel - Camera feed
        left_panel = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Camera label
        self.camera_label = tk.Label(left_panel, bg='#34495e')
        self.camera_label.pack(pady=10)
        
        # Hand landmarks visualization
        self.landmarks_label = tk.Label(left_panel, bg='#34495e')
        self.landmarks_label.pack(pady=5)
        
        # Right panel - Controls and output
        right_panel = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        
        # Detection status
        status_frame = tk.Frame(right_panel, bg='#34495e')
        status_frame.pack(fill='x', padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame, 
            text="🔴 Ready to Detect", 
            font=("Arial", 14, "bold"),
            bg='#34495e', 
            fg='#e74c3c'
        )
        self.status_label.pack()
        
        # Mic-style button
        self.mic_button = tk.Button(
            right_panel,
            text="🎤 Hold to Detect",
            font=("Arial", 16, "bold"),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=3,
            width=15,
            height=3
        )
        self.mic_button.pack(pady=20)
        
        # Bind mouse events for press and hold
        self.mic_button.bind('<Button-1>', self.start_detection)
        self.mic_button.bind('<ButtonRelease-1>', self.stop_detection)
        
        # Current letter display
        letter_frame = tk.Frame(right_panel, bg='#34495e')
        letter_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(letter_frame, text="Current Letter:", font=("Arial", 12, "bold"), 
                bg='#34495e', fg='#ecf0f1').pack()
        
        self.current_letter_label = tk.Label(
            letter_frame, 
            text="--", 
            font=("Arial", 24, "bold"),
            bg='#34495e', 
            fg='#f39c12'
        )
        self.current_letter_label.pack()
        
        # Confidence display
        self.confidence_label = tk.Label(
            letter_frame, 
            text="Confidence: --", 
            font=("Arial", 10),
            bg='#34495e', 
            fg='#95a5a6'
        )
        self.confidence_label.pack()
        
        # Detected letters
        letters_frame = tk.Frame(right_panel, bg='#34495e')
        letters_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(letters_frame, text="Detected Letters:", font=("Arial", 12, "bold"), 
                bg='#34495e', fg='#ecf0f1').pack()
        
        self.letters_text = tk.Text(
            letters_frame, 
            height=3, 
            width=20,
            font=("Courier", 12),
            bg='#2c3e50',
            fg='#ecf0f1',
            state='disabled'
        )
        self.letters_text.pack()
        
        # Corrected text
        corrected_frame = tk.Frame(right_panel, bg='#34495e')
        corrected_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(corrected_frame, text="Corrected Text:", font=("Arial", 12, "bold"), 
                bg='#34495e', fg='#ecf0f1').pack()
        
        self.corrected_text_label = tk.Label(
            corrected_frame, 
            text="", 
            font=("Arial", 14),
            bg='#34495e', 
            fg='#27ae60',
            wraplength=200
        )
        self.corrected_text_label.pack()
        
        # Control buttons
        controls_frame = tk.Frame(right_panel, bg='#34495e')
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Clear button
        self.clear_button = tk.Button(
            controls_frame,
            text="🗑️ Clear",
            font=("Arial", 12),
            bg='#e74c3c',
            fg='white',
            command=self.clear_all
        )
        self.clear_button.pack(fill='x', pady=5)
        
        # Speak button
        self.speak_button = tk.Button(
            controls_frame,
            text="🔊 Speak",
            font=("Arial", 12),
            bg='#9b59b6',
            fg='white',
            command=self.speak_text
        )
        self.speak_button.pack(fill='x', pady=5)
        
        # Settings frame
        settings_frame = tk.Frame(right_panel, bg='#34495e')
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(settings_frame, text="Settings:", font=("Arial", 12, "bold"), 
                bg='#34495e', fg='#ecf0f1').pack()
        
        # Confidence threshold slider
        tk.Label(settings_frame, text="Confidence Threshold:", font=("Arial", 10), 
                bg='#34495e', fg='#ecf0f1').pack()
        
        self.confidence_scale = tk.Scale(
            settings_frame,
            from_=0.5,
            to=0.95,
            resolution=0.05,
            orient='horizontal',
            command=self.update_confidence_threshold,
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.confidence_scale.set(0.8)
        self.confidence_scale.pack(fill='x')
        
    def start_detection(self, event):
        """Start sign language detection"""
        self.is_detecting = True
        self.detected_letters = []
        self.prediction_history = []
        
        self.status_label.config(text="🟢 Detecting...", fg='#27ae60')
        self.mic_button.config(text="🎤 Detecting...", bg='#e74c3c')
        
        print("Detection started!")
        
    def stop_detection(self, event):
        """Stop sign language detection and process results"""
        self.is_detecting = False
        
        self.status_label.config(text="🔴 Ready to Detect", fg='#e74c3c')
        self.mic_button.config(text="🎤 Hold to Detect", bg='#3498db')
        
        # Process detected letters
        if self.detected_letters:
            self.process_detected_letters()
        
        print("Detection stopped!")
        
    def process_detected_letters(self):
        """Process and correct detected letters"""
        if not self.detected_letters:
            return
            
        # Combine letters into text
        raw_text = ''.join(self.detected_letters)
        
        # Update letters display
        self.letters_text.config(state='normal')
        self.letters_text.delete(1.0, tk.END)
        self.letters_text.insert(1.0, raw_text)
        self.letters_text.config(state='disabled')
        
        # Apply spell correction
        corrected_text = self.correct_spelling(raw_text)
        
        # Update corrected text display
        self.corrected_text_label.config(text=corrected_text)
        
        # Auto-speak the corrected text
        if self.voice_enabled and corrected_text.strip():
            threading.Thread(target=self.speak_text_thread, args=(corrected_text,)).start()
        
        print(f"Raw: {raw_text}")
        print(f"Corrected: {corrected_text}")
        
    def correct_spelling(self, text):
        """Apply advanced spell correction"""
        if not text.strip():
            return ""
            
        # Method 1: SymSpell (if available)
        if SYMSPELL_AVAILABLE and hasattr(self, 'sym_spell'):
            try:
                suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
                if suggestions:
                    return suggestions[0].term
            except:
                pass
        
        # Method 2: TextBlob correction
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            if corrected != text:  # Only use if correction was made
                return corrected
        except:
            pass
        
        # Method 3: Word-by-word enchant correction
        words = text.split()
        corrected_words = []
        
        for word in words:
            if len(word) > 2 and not self.enchant_dict.check(word):
                suggestions = self.enchant_dict.suggest(word)
                if suggestions:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
        
    def speak_text_thread(self, text):
        """Thread-safe text-to-speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
            
    def speak_text(self):
        """Manual text-to-speech"""
        corrected_text = self.corrected_text_label.cget("text")
        if corrected_text.strip():
            threading.Thread(target=self.speak_text_thread, args=(corrected_text,)).start()
            
    def clear_all(self):
        """Clear all text and reset"""
        self.detected_letters = []
        self.prediction_history = []
        
        self.letters_text.config(state='normal')
        self.letters_text.delete(1.0, tk.END)
        self.letters_text.config(state='disabled')
        
        self.corrected_text_label.config(text="")
        self.current_letter_label.config(text="--")
        self.confidence_label.config(text="Confidence: --")
        
    def update_confidence_threshold(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        
    def detect_hand_landmarks(self, frame):
        """Detect hand landmarks using MediaPipe or OpenCV fallback"""
        if MEDIAPIPE_AVAILABLE and self.hands is not None:
            return self.detect_hand_mediapipe(frame)
        else:
            return self.detect_hand_opencv(frame)
            
    def detect_hand_mediapipe(self, frame):
        """Detect hand landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return hand_landmarks, frame
        
    def detect_hand_opencv(self, frame):
        """Fallback hand detection using OpenCV"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_landmarks = None
        if contours:
            # Get largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Hand Detected (OpenCV)", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Create a simple landmark structure for compatibility
            class SimpleLandmark:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    
            class SimpleHandLandmarks:
                def __init__(self, bbox):
                    x, y, w, h = bbox
                    # Create 21 fake landmarks based on bounding box
                    self.landmark = []
                    for i in range(21):
                        # Distribute landmarks across the bounding box
                        lx = (x + (i % 7) * w / 6) / frame.shape[1]
                        ly = (y + (i // 7) * h / 2) / frame.shape[0]
                        self.landmark.append(SimpleLandmark(lx, ly))
            
            hand_landmarks = SimpleHandLandmarks((x, y, w, h))
        
        return hand_landmarks, frame
        
    def extract_hand_roi(self, frame, hand_landmarks):
        """Extract hand region of interest"""
        if not hand_landmarks:
            return None
            
        h, w, _ = frame.shape
        
        # Get bounding box of hand
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        # Add padding
        padding = 50
        x1 = max(0, int(min(x_coords) - padding))
        y1 = max(0, int(min(y_coords) - padding))
        x2 = min(w, int(max(x_coords) + padding))
        y2 = min(h, int(max(y_coords) + padding))
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        return roi, (x1, y1, x2, y2)
        
    def preprocess_image(self, image):
        """Enhanced image preprocessing"""
        if image is None or image.size == 0:
            return None
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to model input size
        gray = cv2.resize(gray, (128, 128))
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
        
    def predict_letter(self, image):
        """Predict letter from preprocessed image"""
        if image is None:
            return None, 0.0
            
        # Resize for model
        image = cv2.resize(image, (128, 128))
        
        # Get predictions from all models
        result = self.loaded_model.predict(image.reshape(1, 128, 128, 1), verbose=0)
        result_dru = self.loaded_model_dru.predict(image.reshape(1, 128, 128, 1), verbose=0)
        result_tkdi = self.loaded_model_tkdi.predict(image.reshape(1, 128, 128, 1), verbose=0)
        result_smn = self.loaded_model_smn.predict(image.reshape(1, 128, 128, 1), verbose=0)
        
        # Build prediction dictionary
        prediction = {'blank': result[0][0]}
        
        idx = 1
        for letter in ascii_uppercase:
            prediction[letter] = result[0][idx]
            idx += 1
            
        # Get top prediction
        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_letter, top_confidence = prediction_sorted[0]
        
        # Apply specialized models for similar letters
        final_letter = self.apply_specialized_models(
            top_letter, result_dru, result_tkdi, result_smn
        )
        
        return final_letter, top_confidence
        
    def apply_specialized_models(self, letter, result_dru, result_tkdi, result_smn):
        """Apply specialized models for similar letters"""
        
        # DRU Model
        if letter in ['D', 'R', 'U']:
            dru_prediction = {
                'D': result_dru[0][0],
                'R': result_dru[0][1],
                'U': result_dru[0][2]
            }
            dru_sorted = sorted(dru_prediction.items(), key=operator.itemgetter(1), reverse=True)
            return dru_sorted[0][0]
            
        # TKDI Model
        if letter in ['D', 'I', 'K', 'T']:
            tkdi_prediction = {
                'D': result_tkdi[0][0],
                'I': result_tkdi[0][1],
                'K': result_tkdi[0][2],
                'T': result_tkdi[0][3]
            }
            tkdi_sorted = sorted(tkdi_prediction.items(), key=operator.itemgetter(1), reverse=True)
            return tkdi_sorted[0][0]
            
        # SMN Model
        if letter in ['M', 'N', 'S']:
            smn_prediction = {
                'M': result_smn[0][0],
                'N': result_smn[0][1],
                'S': result_smn[0][2]
            }
            smn_sorted = sorted(smn_prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            if smn_sorted[0][0] == 'S':
                return 'S'
            else:
                return letter
                
        return letter
        
    def apply_temporal_smoothing(self, letter, confidence):
        """Apply temporal smoothing to reduce flickering"""
        self.prediction_history.append((letter, confidence))
        
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
            
        if len(self.prediction_history) < self.stable_count_threshold:
            return None
            
        # Count occurrences
        letter_counts = {}
        for l, c in self.prediction_history:
            letter_counts[l] = letter_counts.get(l, 0) + 1
            
        # Find most frequent letter
        most_frequent = max(letter_counts.items(), key=lambda x: x[1])
        
        # Only return if it appears enough times and has good confidence
        if most_frequent[1] >= self.stable_count_threshold:
            return most_frequent[0]
            
        return None
        
    def video_loop(self):
        """Main video processing loop"""
        ret, frame = self.cap.read()
        
        if not ret:
            print("ERROR: Failed to read from camera")
            return
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Detect hand landmarks
        hand_landmarks, frame = self.detect_hand_landmarks(frame)
        
        # Only process if detecting
        if self.is_detecting and hand_landmarks:
            # Extract hand ROI
            roi, bbox = self.extract_hand_roi(frame, hand_landmarks)
            
            if roi is not None:
                # Preprocess image
                processed_image = self.preprocess_image(roi)
                
                if processed_image is not None:
                    # Predict letter
                    letter, confidence = self.predict_letter(processed_image)
                    
                    # Apply confidence threshold
                    if confidence >= self.confidence_threshold and letter != 'blank':
                        # Apply temporal smoothing
                        smoothed_letter = self.apply_temporal_smoothing(letter, confidence)
                        
                        if smoothed_letter and smoothed_letter not in ['blank', '--']:
                            # Add to detected letters if not already there
                            if not self.detected_letters or self.detected_letters[-1] != smoothed_letter:
                                self.detected_letters.append(smoothed_letter)
                                
                            # Update display
                            self.current_letter_label.config(text=smoothed_letter)
                            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{smoothed_letter} ({confidence:.2f})", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)
        
        # Update camera display
        self.camera_label.config(image=frame_tk)
        self.camera_label.image = frame_tk
        
        # Schedule next frame
        self.root.after(33, self.video_loop)  # ~30 FPS
        
    def show_error(self, message):
        """Show error message"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        error_window.geometry("300x100")
        
        tk.Label(error_window, text=message, font=("Arial", 12), 
                fg='red').pack(expand=True)
        
        tk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=10)
        
    def run(self):
        """Start the application"""
        print("Starting Enhanced Sign Language Recognition App...")
        self.root.mainloop()
        
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnhancedSignLanguageApp()
    app.run()
