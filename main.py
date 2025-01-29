import cv2
import mediapipe as mp
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading
from datetime import datetime

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Number of landmarks * 3 (x, y, z coordinates)
        self.feature_size = 21 * 3
        
        # Initialize state variables
        self.recording = False
        self.debug_mode = False
        self.model_active = False
        self.current_samples = []
        self.data = self.load_data()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
        # Create UI
        self.create_ui()
        
        # Start camera thread
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def create_ui(self):
        """Create the user interface"""
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition")
        
        # Main frame
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Gesture selection
        ttk.Label(frame, text="Select Gesture:").grid(row=0, column=0, pady=5)
        self.gesture_var = tk.StringVar()
        gesture_combo = ttk.Combobox(frame, textvariable=self.gesture_var, 
                                   values=[chr(i) for i in range(65, 91)])  # A-Z
        gesture_combo.grid(row=0, column=1, pady=5)
        
        # Sample counter
        self.sample_label = ttk.Label(frame, text="Samples: 0")
        self.sample_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Control buttons
        self.record_btn = ttk.Button(frame, text="Start Recording", 
                                   command=self.toggle_recording)
        self.record_btn.grid(row=2, column=0, pady=5)
        
        ttk.Button(frame, text="Save Samples", 
                  command=self.save_samples).grid(row=2, column=1, pady=5)
        
        ttk.Button(frame, text="Clear Samples", 
                  command=self.clear_samples).grid(row=3, column=0, pady=5)
        
        ttk.Button(frame, text="Train Model", 
                  command=self.train_model).grid(row=3, column=1, pady=5)
        
        # Model controls
        self.model_btn = ttk.Button(frame, text="Start Recognition", 
                                  command=self.toggle_model)
        self.model_btn.grid(row=4, column=0, pady=5)
        
        self.debug_btn = ttk.Button(frame, text="Enable Debug", 
                                  command=self.toggle_debug)
        self.debug_btn.grid(row=4, column=1, pady=5)
        
        # Data display
        self.data_list = tk.Listbox(frame, height=10, width=40)
        self.data_list.grid(row=5, column=0, columnspan=2, pady=5)
        
        ttk.Button(frame, text="Delete Selected", 
                  command=self.delete_selected).grid(row=6, column=0, pady=5)
        
        ttk.Button(frame, text="Quit", 
                  command=self.quit_app).grid(row=6, column=1, pady=5)
        
        self.update_data_list()
    
    def camera_loop(self):
        """Main camera processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features if recording or model is active
                    features = self.extract_features(hand_landmarks)
                    
                    if self.recording:
                        self.current_samples.append(features)
                        self.root.after(0, self.update_sample_count)
                    
                    if self.model_active and hasattr(self, 'model'):
                        prediction = self.model.predict(np.array([features]))
                        predicted_class = np.argmax(prediction[0])
                        confidence = float(np.max(prediction[0]))
                        gesture = self.label_encoder.inverse_transform([predicted_class])[0]
                        
                        # Display prediction
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if self.debug_mode:
                            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2)
            
            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                self.quit_app()
                break
    
    def extract_features(self, landmarks):
        """Extract features from hand landmarks"""
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Ensure consistent feature size
        if len(features) != self.feature_size:
            print(f"Warning: Expected {self.feature_size} features, got {len(features)}")
            # Pad or truncate to ensure consistent size
            if len(features) < self.feature_size:
                features.extend([0] * (self.feature_size - len(features)))
            else:
                features = features[:self.feature_size]
        
        return np.array(features, dtype=np.float32)
    
    def toggle_recording(self):
        """Toggle recording state"""
        self.recording = not self.recording
        self.record_btn.config(
            text="Stop Recording" if self.recording else "Start Recording"
        )
    
    def save_samples(self):
        """Save current samples"""
        if not self.current_samples or not self.gesture_var.get():
            messagebox.showwarning("Warning", "No samples to save or no gesture selected!")
            return
        
        # Validate samples before saving
        valid_samples = []
        for sample in self.current_samples:
            if len(sample) == self.feature_size:
                valid_samples.append(sample.tolist())
        
        if not valid_samples:
            messagebox.showwarning("Warning", "No valid samples to save!")
            return
        
        gesture = self.gesture_var.get()
        if gesture not in self.data:
            self.data[gesture] = []
        
        self.data[gesture].extend(valid_samples)
        self.save_data()
        self.clear_samples()
        self.update_data_list()
    
    def clear_samples(self):
        """Clear current samples"""
        self.current_samples = []
        self.update_sample_count()
    
    def update_sample_count(self):
        """Update sample counter in UI"""
        self.sample_label.config(text=f"Samples: {len(self.current_samples)}")
    
    def train_model(self):
        """Train the model on collected data"""
        if not self.data:
            messagebox.showwarning("Warning", "No training data available!")
            return
        
        try:
            # Prepare data
            X, y = [], []
            for gesture, samples in self.data.items():
                for sample in samples:
                    # Verify sample size
                    if len(sample) == self.feature_size:
                        X.append(sample)
                        y.append(gesture)
            
            if not X:
                messagebox.showwarning("Warning", "No valid samples found!")
                return
            
            # Convert to numpy arrays
            X = np.array(X, dtype=np.float32)
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
            print(f"Training with {len(X)} samples, {len(np.unique(y))} classes")
            print(f"Input shape: {X.shape}")
            
            # Create and train model
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(self.feature_size,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
            ])
            
            self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
            
            # Train the model
            history = self.model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
            
            # Save the model and label encoder
            self.model.save('gesture_model.h5')
            import pickle
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            messagebox.showinfo("Success", 
                              f"Model training completed!\n"
                              f"Final accuracy: {history.history['accuracy'][-1]:.2f}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            print(f"Training error: {str(e)}")
    
    def toggle_model(self):
        """Toggle model recognition"""
        if not hasattr(self, 'model'):
            try:
                self.model = tf.keras.models.load_model('gesture_model.h5')
                with open('label_encoder.pkl', 'rb') as f:
                    import pickle
                    self.label_encoder = pickle.load(f)
            except:
                messagebox.showerror("Error", "No trained model found!")
                return
        
        self.model_active = not self.model_active
        self.model_btn.config(
            text="Stop Recognition" if self.model_active else "Start Recognition"
        )
    
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        self.debug_btn.config(
            text="Disable Debug" if self.debug_mode else "Enable Debug"
        )
    
    def load_data(self):
        """Load saved gesture data"""
        try:
            if os.path.exists('gesture_data.json'):
                with open('gesture_data.json', 'r') as f:
                    loaded_data = json.load(f)
                
                # Verify loaded data
                verified_data = {}
                for gesture, samples in loaded_data.items():
                    valid_samples = [s for s in samples if len(s) == self.feature_size]
                    if valid_samples:
                        verified_data[gesture] = valid_samples
                return verified_data
        except Exception as e:
            print(f"Error loading data: {e}")
        return {}
    
    def save_data(self):
        """Save gesture data"""
        try:
            # Verify data before saving
            verified_data = {}
            for gesture, samples in self.data.items():
                valid_samples = [s for s in samples if len(s) == self.feature_size]
                if valid_samples:
                    verified_data[gesture] = valid_samples
            
            with open('gesture_data.json', 'w') as f:
                json.dump(verified_data, f)
            
            self.data = verified_data
        except Exception as e:
            print(f"Error saving data: {e}")
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def update_data_list(self):
        """Update data display in UI"""
        self.data_list.delete(0, tk.END)
        for gesture, samples in self.data.items():
            self.data_list.insert(tk.END, f"{gesture}: {len(samples)} samples")
    
    def delete_selected(self):
        """Delete selected gesture data"""
        selection = self.data_list.curselection()
        if selection:
            gesture = self.data_list.get(selection[0]).split(':')[0]
            if gesture in self.data:
                del self.data[gesture]
                self.save_data()
                self.update_data_list()
    
    def quit_app(self):
        """Clean up and quit application"""
        self.running = False
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = HandGestureRecognizer()
    app.run()
