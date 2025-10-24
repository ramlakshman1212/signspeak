# Importing Libraries

import numpy as np

import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

import enchant
import pyttsx3

try:
    from tensorflow.keras.models import model_from_json
except ImportError:
    from keras.models import model_from_json

# Try to import MediaPipe for better hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Using basic hand detection.")

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):

        self.hs = enchant.Dict("en_US")
        
        # Initialize text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            self.voice_enabled = True
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.voice_enabled = False
        
        # Initialize MediaPipe for hand detection if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            print("Error: Could not open camera!")
            return
        self.current_image = None
        self.current_image2 = None
        
        # Load main model
        try:
            self.json_file = open("Models/model_new.json", "r")
            self.model_json = self.json_file.read()
            self.json_file.close()
            self.loaded_model = model_from_json(self.model_json)
            self.loaded_model.load_weights("Models/model_new.h5")
        except FileNotFoundError:
            print("Error: model_new.json or model_new.h5 not found!")
            return
        
        # Load DRU model
        try:
            self.json_file_dru = open("Models/model-bw_dru.json" , "r")
            self.model_json_dru = self.json_file_dru.read()
            self.json_file_dru.close()
            self.loaded_model_dru = model_from_json(self.model_json_dru)
            self.loaded_model_dru.load_weights("Models/model-bw_dru.h5")
        except FileNotFoundError:
            print("Error: model-bw_dru.json or model-bw_dru.h5 not found!")
            return
            
        # Load TKDI model
        try:
            self.json_file_tkdi = open("Models/model-bw_tkdi.json" , "r")
            self.model_json_tkdi = self.json_file_tkdi.read()
            self.json_file_tkdi.close()
            self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
            self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi.h5")
        except FileNotFoundError:
            print("Error: model-bw_tkdi.json or model-bw_tkdi.h5 not found!")
            return
            
        # Load SMN model
        try:
            self.json_file_smn = open("Models/model-bw_smn.json" , "r")
            self.model_json_smn = self.json_file_smn.read()
            self.json_file_smn.close()
            self.loaded_model_smn = model_from_json(self.model_json_smn)
            self.loaded_model_smn.load_weights("Models/model-bw_smn.h5")
        except FileNotFoundError:
            print("Error: model-bw_smn.json or model-bw_smn.h5 not found!")
            return

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        # Enhanced accuracy tracking
        self.prediction_history = []
        self.confidence_threshold = 0.7  # Minimum confidence for prediction
        self.history_length = 5  # Number of recent predictions to consider
        self.stable_count_threshold = 3  # How many consistent predictions needed
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 580, height = 580)
        
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 400, y = 65, width = 275, height = 275)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 690)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 745)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 745)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 745)


        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        
        # Add voice control button
        self.voice_btn = tk.Button(self.root, text="🔊 Speak", command=self.speak_text, 
                                  font=("Courier", 15), bg="lightblue")
        self.voice_btn.place(x=750, y=540)
        
        # Add clear button
        self.clear_btn = tk.Button(self.root, text="🗑️ Clear", command=self.clear_text, 
                                 font=("Courier", 15), bg="lightcoral")
        self.clear_btn.place(x=750, y=580)
        
        # Add confidence display
        self.confidence_label = tk.Label(self.root, text="Confidence: --", 
                                       font=("Courier", 12), fg="blue")
        self.confidence_label.place(x=750, y=620)
        
        # Add accuracy settings
        self.accuracy_label = tk.Label(self.root, text="Accuracy Settings:", 
                                     font=("Courier", 12, "bold"))
        self.accuracy_label.place(x=750, y=650)
        
        # Confidence threshold slider
        self.confidence_var = tk.DoubleVar(value=0.7)
        self.confidence_scale = tk.Scale(self.root, from_=0.5, to=0.9, resolution=0.1,
                                       orient=tk.HORIZONTAL, variable=self.confidence_var,
                                       command=self.update_confidence_threshold,
                                       label="Confidence Threshold")
        self.confidence_scale.place(x=750, y=680)
        
        self.video_loop()


    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)
            
            # Enhanced hand detection and ROI
            hand_roi = self.detect_hand_roi(cv2image)
            
            if hand_roi is not None:
                x1, y1, x2, y2 = hand_roi
                cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 255, 0), 2)
                cv2.putText(cv2image, "Hand Detected", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Fallback to center region if no hand detected
                x1 = int(0.5 * frame.shape[1])
                y1 = 10
                x2 = frame.shape[1] - 10
                y2 = int(0.5 * frame.shape[1])
                cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
                cv2.putText(cv2image, "No Hand Detected", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            # Extract ROI for processing
            roi_image = cv2image[y1 : y2, x1 : x2]

            # Enhanced preprocessing for better hand recognition
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            # Enhanced blur and thresholding
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            
            # Use Otsu's method for better thresholding
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            self.predict(thresh)

            self.current_image2 = Image.fromarray(thresh)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            self.panel3.config(text = self.current_symbol, font = ("Courier", 30))

            self.panel4.config(text = self.word, font = ("Courier", 30))

            self.panel5.config(text = self.str,font = ("Courier", 30))

            predicts = self.hs.suggest(self.word)
            
            if(len(predicts) > 1):

                self.bt1.config(text = predicts[0], font = ("Courier", 20))

            else:

                self.bt1.config(text = "")

            if(len(predicts) > 2):

                self.bt2.config(text = predicts[1], font = ("Courier", 20))

            else:

                self.bt2.config(text = "")

            if(len(predicts) > 3):

                self.bt3.config(text = predicts[2], font = ("Courier", 20))

            else:

                self.bt3.config(text = "")


        self.root.after(5, self.video_loop)

    def detect_hand_roi(self, frame):
        """Detect hand region using MediaPipe or OpenCV"""
        if MEDIAPIPE_AVAILABLE:
            return self.detect_hand_mediapipe(frame)
        else:
            return self.detect_hand_opencv(frame)
    
    def detect_hand_mediapipe(self, frame):
        """Use MediaPipe for precise hand detection"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                # Add padding around hand
                padding = 50
                x1 = max(0, int(min(x_coords) - padding))
                y1 = max(0, int(min(y_coords) - padding))
                x2 = min(w, int(max(x_coords) + padding))
                y2 = min(h, int(max(y_coords) + padding))
                
                # Ensure minimum size
                min_size = 100
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    x1 = max(0, center_x - min_size // 2)
                    y1 = max(0, center_y - min_size // 2)
                    x2 = min(w, center_x + min_size // 2)
                    y2 = min(h, center_y + min_size // 2)
                
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"MediaPipe hand detection error: {e}")
        
        return None
    
    def detect_hand_opencv(self, frame):
        """Fallback hand detection using OpenCV"""
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (likely the hand)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding and ensure minimum size
                padding = 30
                min_size = 100
                
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                # Ensure minimum size
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    x1 = max(0, center_x - min_size // 2)
                    y1 = max(0, center_y - min_size // 2)
                    x2 = min(frame.shape[1], center_x + min_size // 2)
                    y2 = min(frame.shape[0], center_y + min_size // 2)
                
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"OpenCV hand detection error: {e}")
        
        return None

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
        
        # Reset prediction history
        self.prediction_history = []
    
    def update_confidence_threshold(self, value):
        """Update confidence threshold from slider"""
        self.confidence_threshold = float(value)
        print(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def update_confidence_display(self, confidence):
        """Update confidence display"""
        if confidence is not None:
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        else:
            self.confidence_label.config(text="Confidence: --")

    def predict(self, test_image):
        """Enhanced prediction with confidence filtering and temporal smoothing"""
        
        # Enhanced preprocessing for better accuracy
        test_image = self.enhance_image_for_prediction(test_image)
        test_image = cv2.resize(test_image, (128, 128))

        # Get predictions from all models
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        # Build prediction dictionary
        prediction = {}
        prediction['blank'] = result[0][0]
        
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1 - Get initial prediction
        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_prediction = prediction_sorted[0]
        top_confidence = top_prediction[1]
        
        # Apply confidence threshold
        if top_confidence < self.confidence_threshold:
            # If confidence is too low, don't update prediction
            return
        
        initial_symbol = top_prediction[0]

        # LAYER 2 - Apply specialized models for similar characters
        final_symbol = self.apply_specialized_models(initial_symbol, result_dru, result_tkdi, result_smn)
        
        # Add to prediction history for temporal smoothing
        self.prediction_history.append((final_symbol, top_confidence))
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        # Apply temporal smoothing
        smoothed_symbol = self.apply_temporal_smoothing()
        
        if smoothed_symbol:
            self.current_symbol = smoothed_symbol
            self.update_character_counters()
        
        # Update confidence display
        self.update_confidence_display(top_confidence)

    def enhance_image_for_prediction(self, image):
        """Enhanced image preprocessing for better accuracy"""
        # Apply additional noise reduction
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Apply bilateral filter for edge-preserving smoothing
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        return image

    def apply_specialized_models(self, initial_symbol, result_dru, result_tkdi, result_smn):
        """Apply specialized models for similar characters"""
        
        # DRU Model
        if initial_symbol in ['D', 'R', 'U']:
            dru_prediction = {}
            dru_prediction['D'] = result_dru[0][0]
            dru_prediction['R'] = result_dru[0][1]
            dru_prediction['U'] = result_dru[0][2]
            dru_sorted = sorted(dru_prediction.items(), key=operator.itemgetter(1), reverse=True)
            return dru_sorted[0][0]

        # TKDI Model
        if initial_symbol in ['D', 'I', 'K', 'T']:
            tkdi_prediction = {}
            tkdi_prediction['D'] = result_tkdi[0][0]
            tkdi_prediction['I'] = result_tkdi[0][1]
            tkdi_prediction['K'] = result_tkdi[0][2]
            tkdi_prediction['T'] = result_tkdi[0][3]
            tkdi_sorted = sorted(tkdi_prediction.items(), key=operator.itemgetter(1), reverse=True)
            return tkdi_sorted[0][0]

        # SMN Model
        if initial_symbol in ['M', 'N', 'S']:
            smn_prediction = {}
            smn_prediction['M'] = result_smn[0][0]
            smn_prediction['N'] = result_smn[0][1]
            smn_prediction['S'] = result_smn[0][2]
            smn_sorted = sorted(smn_prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            # Special logic for S
            if smn_sorted[0][0] == 'S':
                return 'S'
            else:
                return initial_symbol
        
        return initial_symbol

    def apply_temporal_smoothing(self):
        """Apply temporal smoothing to reduce flickering"""
        if len(self.prediction_history) < self.stable_count_threshold:
            return None
        
        # Count occurrences of each symbol in recent history
        symbol_counts = {}
        for symbol, confidence in self.prediction_history:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Find the most frequent symbol
        most_frequent = max(symbol_counts.items(), key=lambda x: x[1])
        
        # Only return if it appears enough times
        if most_frequent[1] >= self.stable_count_threshold:
            return most_frequent[0]
        
        return None

    def update_character_counters(self):
        """Update character counters with enhanced logic"""
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
            return

        self.ct[self.current_symbol] += 1

        # Enhanced threshold logic
        if self.ct[self.current_symbol] > 60:
            # Check if this prediction is significantly more confident than others
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                # Increased threshold for more stability
                if tmp <= 25:  # Increased from 20 to 25
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
                    
                    # Auto-speak when a word is completed
                    if self.voice_enabled and len(self.str.strip()) > 0:
                        try:
                            self.tts_engine.say(self.str)
                            self.tts_engine.runAndWait()
                        except Exception as e:
                            print(f"Auto-TTS error: {e}")
            else:
                if len(self.str) > 50:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 0):

            self.word = ""

            self.str += " "

            self.str += predicts[0]

    def action2(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 1):
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 2):
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 3):
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 4):
            self.word = ""
            self.str += " "
            self.str += predicts[4]
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()