import cv2
import mediapipe as mp
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading
from datetime import datetime
import time
from PIL import Image, ImageTk
import pickle

# --- Constants ---
DATA_FILE = 'gesture_data_normalized.json'
MODEL_FILE = 'gesture_model_normalized.h5'
ENCODER_FILE = 'label_encoder_normalized.pkl'
# Feature size: 20 landmarks (excluding wrist) * 3 coordinates (x, y, z relative to wrist)
FEATURE_SIZE = 20 * 3
GESTURE_OPTIONS = [chr(i) for i in range(65, 91)] # A-Z


class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, # Process only one hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.feature_size = FEATURE_SIZE

        # Initialize state variables
        self.recording = False
        self.debug_mode = False
        self.model_active = False
        self.model_trained = False
        self.current_samples = []
        self.data = {} # Loaded later
        self.model = None
        self.label_encoder = None

        # Initialize UI elements to None initially
        self.root = None
        self.video_label = None
        self.record_btn = None
        self.model_btn = None
        self.train_btn = None
        self.debug_btn = None
        self.status_var = None
        self.sample_label = None
        self.data_list = None
        self.gesture_var = None

        # Camera setup will happen after UI is created

        # Create UI (sets up self.root)
        self.create_ui()

        # Now initialize camera and load data
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please check connection.")
            self.root.destroy() # Close UI if camera fails
            return

        self.data = self.load_data() # Load data after UI exists for status updates
        self.update_data_list()
        self.check_model_exists() # Check if model exists on startup

        # Start camera thread
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def set_status(self, text):
        """Update the status bar safely from any thread."""
        if self.root and self.status_var:
            self.root.after(0, lambda: self.status_var.set(text))

    def create_ui(self):
        """Create the user interface"""
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition v2.0")
        self.root.geometry("900x700") # Adjusted size for video

        # --- Main Panes ---
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Pane (Video)
        video_frame = ttk.Frame(main_pane, width=640, height=480)
        video_frame.pack_propagate(False) # Prevent shrinking
        self.video_label = ttk.Label(video_frame, text="Initializing Camera...")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        main_pane.add(video_frame, stretch="always")

        # Right Pane (Controls)
        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame)

        # --- Control Sections ---
        # 1. Data Collection
        collect_frame = ttk.LabelFrame(controls_frame, text="Data Collection", padding="10")
        collect_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        ttk.Label(collect_frame, text="Gesture:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.gesture_var = tk.StringVar()
        gesture_combo = ttk.Combobox(collect_frame, textvariable=self.gesture_var,
                                   values=GESTURE_OPTIONS, state="readonly", width=10)
        gesture_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        if GESTURE_OPTIONS:
            gesture_combo.current(0) # Select 'A' by default

        self.sample_label = ttk.Label(collect_frame, text="Samples: 0")
        self.sample_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.record_btn = ttk.Button(collect_frame, text="Start Recording",
                                   command=self.toggle_recording)
        self.record_btn.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        btn_frame1 = ttk.Frame(collect_frame)
        btn_frame1.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(btn_frame1, text="Save Samples", width=12,
                  command=self.save_samples).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(btn_frame1, text="Clear Current", width=12,
                  command=self.clear_samples).pack(side=tk.LEFT, padx=2, pady=2)
        collect_frame.columnconfigure(1, weight=1)


        # 2. Model Training & Recognition
        model_frame = ttk.LabelFrame(controls_frame, text="Model", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.train_btn = ttk.Button(model_frame, text="Train Model",
                                  command=self.start_training_thread)
        self.train_btn.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.model_btn = ttk.Button(model_frame, text="Start Recognition",
                                  command=self.toggle_model)
        self.model_btn.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.model_btn.config(state=tk.DISABLED) # Disabled until model is trained/loaded

        self.debug_btn = ttk.Button(model_frame, text="Enable Debug",
                                  command=self.toggle_debug)
        self.debug_btn.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)


        # 3. Data Management
        data_frame = ttk.LabelFrame(controls_frame, text="Saved Data", padding="10")
        data_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.data_list = tk.Listbox(data_frame, height=8, width=35)
        self.data_list.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_list.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.data_list['yscrollcommand'] = scrollbar.set

        ttk.Button(data_frame, text="Delete Selected",
                  command=self.delete_selected).grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)


        # --- Status Bar ---
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.set_status("Ready.")

        # --- Quit Button ---
        ttk.Button(controls_frame, text="Quit",
                  command=self.quit_app).grid(row=3, column=0, pady=10)

        # Allow controls column to resize
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.rowconfigure(2, weight=1) # Allow data list to expand


    def check_model_exists(self):
        """Check if model files exist and update UI"""
        if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
            self.model_trained = True
            self.model_btn.config(state=tk.NORMAL)
            self.set_status("Existing model found. Ready for recognition.")
            # Optionally load model here if desired on startup
            # self.load_model_files()
        else:
            self.model_trained = False
            self.model_btn.config(state=tk.DISABLED)
            self.set_status("No trained model found. Collect data and train first.")


    def camera_loop(self):
        """Main camera processing loop"""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.set_status("Error reading frame from camera.")
                time.sleep(0.5)
                continue

            frame = cv2.flip(frame, 1) # Flip horizontally for intuitive view
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            gesture_text = ""
            confidence_text = ""

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                # Currently set to max_num_hands=1, so only one hand
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extract features
                features = self.extract_features(hand_landmarks)

                if features is not None:
                    if self.recording:
                        self.current_samples.append(features)
                        # Update count using after - schedule UI update from main thread
                        self.root.after(0, self.update_sample_count)

                    if self.model_active and self.model and self.label_encoder:
                        try:
                            prediction = self.model.predict(np.array([features]), verbose=0) # verbose=0 suppresses output
                            predicted_class_idx = np.argmax(prediction[0])
                            confidence = float(np.max(prediction[0]))
                            # Only display if confidence is reasonably high
                            if confidence > 0.6: # Confidence threshold
                                gesture = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                                gesture_text = f"Gesture: {gesture}"
                                confidence_text = f"Confidence: {confidence:.2f}"
                            else:
                                gesture_text = "Gesture: ?"
                                confidence_text = f"Confidence: {confidence:.2f}"

                        except Exception as e:
                            print(f"Prediction error: {e}")
                            gesture_text = "Prediction Error"


            # --- Display information on frame ---
            # Recording indicator
            if self.recording:
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1) # Red circle top-left
                cv2.putText(frame, "REC", (55, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Recognition results
            if self.model_active:
                 # Draw background rectangle for text
                bg_y_start = 10
                bg_height = 40 if not self.debug_mode or not confidence_text else 70
                cv2.rectangle(frame, (0, bg_y_start), (280, bg_y_start + bg_height), (0, 0, 0), -1) # Black background

                cv2.putText(frame, gesture_text, (10, bg_y_start + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if self.debug_mode and confidence_text:
                    cv2.putText(frame, confidence_text, (10, bg_y_start + 55),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Convert frame for Tkinter
            try:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Resize image to fit video_label if needed, maintaining aspect ratio
                # Let's assume video_label size is fixed for now (e.g., 640x480)
                # Or get size dynamically: label_width = self.video_label.winfo_width()
                # label_height = self.video_label.winfo_height()
                # Simple resize for now, replace with aspect-preserving later if needed
                img = img.resize((640, 480), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)

                # Update Tkinter label in the main thread
                self.root.after(0, self.update_video_label, imgtk)
            except Exception as e:
                 # Handle cases where the UI might be closing
                 if self.running:
                    print(f"Error updating video label: {e}")

            # Short sleep to yield control, cv2.waitKey is not needed here
            time.sleep(0.01)

        print("Camera loop stopped.")

    def update_video_label(self, imgtk):
        """ Safely updates the video label """
        if self.video_label:
             # Keep a reference to avoid garbage collection
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)


    def extract_features(self, landmarks):
        """Extract and normalize features from hand landmarks relative to the wrist."""
        if not landmarks or not landmarks.landmark:
            return None

        # Use landmark 0 (wrist) as the reference point
        wrist = landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

        # Extract relative coordinates for landmarks 1-20
        features = []
        for i in range(1, 21): # Landmarks 1 to 20
            landmark = landmarks.landmark[i]
            relative_x = landmark.x - wrist_x
            relative_y = landmark.y - wrist_y
            relative_z = landmark.z - wrist_z
            features.extend([relative_x, relative_y, relative_z])

        # Ensure consistent feature size (should always be 20*3=60 now)
        if len(features) != self.feature_size:
            print(f"Warning: Incorrect feature size generated. Expected {self.feature_size}, got {len(features)}")
            # Pad or truncate, though this shouldn't happen with the current logic
            if len(features) < self.feature_size:
                features.extend([0.0] * (self.feature_size - len(features)))
            else:
                features = features[:self.feature_size]

        return np.array(features, dtype=np.float32)


    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.set_status("Recording stopped.")
        else:
            selected_gesture = self.gesture_var.get()
            if not selected_gesture:
                messagebox.showwarning("Warning", "Please select a gesture before recording.")
                return
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.set_status(f"Recording gesture: {selected_gesture}")


    def save_samples(self):
        """Save current samples"""
        if not self.current_samples:
            messagebox.showwarning("Warning", "No samples recorded to save!")
            return

        gesture = self.gesture_var.get()
        if not gesture:
             messagebox.showwarning("Warning", "No gesture selected!")
             return

        # Ensure samples are valid (correct size) - though extract_features should guarantee this
        valid_samples = [s.tolist() for s in self.current_samples if len(s) == self.feature_size]

        if not valid_samples:
            messagebox.showwarning("Warning", "No valid samples recorded (check feature extraction).")
            self.clear_samples() # Clear invalid samples
            return

        count = len(valid_samples)
        if gesture not in self.data:
            self.data[gesture] = []

        self.data[gesture].extend(valid_samples)
        if self.save_data(): # Save data returns True on success
            self.set_status(f"Saved {count} samples for gesture '{gesture}'.")
            self.clear_samples()
            self.update_data_list()
        else:
             self.set_status(f"Failed to save samples for gesture '{gesture}'.")


    def clear_samples(self):
        """Clear current samples"""
        self.current_samples = []
        self.update_sample_count()
        self.set_status("Current recording buffer cleared.")

    def update_sample_count(self):
        """Update sample counter in UI"""
        if self.sample_label:
            self.sample_label.config(text=f"Samples: {len(self.current_samples)}")

    def start_training_thread(self):
        """Starts the training process in a separate thread."""
        if not self.data:
            messagebox.showwarning("Warning", "No training data available! Collect samples first.")
            return

        # Disable buttons during training
        self.train_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        self.model_btn.config(state=tk.DISABLED)

        self.set_status("Starting model training...")
        training_thread = threading.Thread(target=self.train_model, daemon=True)
        training_thread.start()


    def train_model(self):
        """Train the model on collected data (run in a separate thread)."""
        try:
            # Prepare data
            X, y_labels = [], []
            for gesture, samples in self.data.items():
                # Check if gesture has enough samples (optional but good practice)
                if len(samples) < 10: # Minimum samples per gesture
                    print(f"Warning: Skipping gesture '{gesture}' due to insufficient samples ({len(samples)} < 10).")
                    continue
                for sample in samples:
                    # Final validation check
                    if len(sample) == self.feature_size:
                        X.append(sample)
                        y_labels.append(gesture)

            if not X or len(set(y_labels)) < 2: # Need at least 2 classes
                self.root.after(0, lambda: messagebox.showerror("Error", "Training failed: Insufficient valid data or less than 2 gesture classes with enough samples."))
                self.set_status("Training failed: Insufficient data.")
                self._reenable_buttons()
                return

            # Convert to numpy arrays
            X = np.array(X, dtype=np.float32)
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y_labels)
            num_classes = len(np.unique(y))

            print(f"\n--- Starting Training ---")
            print(f"Total samples: {len(X)}")
            print(f"Number of classes: {num_classes}")
            print(f"Feature size (input shape): {self.feature_size}")
            print(f"Class distribution: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
            print(f"Classes encoded as: {list(self.label_encoder.classes_)}")

            # Create and train model
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.feature_size,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.4), # Slightly increased dropout
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

            self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

            # Train the model - consider adding callbacks like EarlyStopping
            print("\nTraining model...")
            history = self.model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=1) # Increased epochs, verbose=1 shows progress
            print("Training complete.")

            # Save the model and label encoder
            self.model.save(MODEL_FILE)
            with open(ENCODER_FILE, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Model saved to {MODEL_FILE}")
            print(f"Label encoder saved to {ENCODER_FILE}")

            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            success_msg = (f"Training completed!\n"
                           f"Final Training Accuracy: {final_acc:.3f}\n"
                           f"Final Validation Accuracy: {final_val_acc:.3f}")

            self.model_trained = True
            self.root.after(0, lambda: messagebox.showinfo("Success", success_msg))
            self.set_status(f"Model trained. Validation Accuracy: {final_val_acc:.3f}")

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"ERROR during training: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.set_status("Training failed.")
            self.model = None # Reset model if training failed
            self.label_encoder = None
            self.model_trained = False
        finally:
             # Re-enable buttons in the main thread
            self._reenable_buttons()

    def _reenable_buttons(self):
        """ Safely re-enable buttons from the main thread """
        def update_ui():
            self.train_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            if self.model_trained:
                self.model_btn.config(state=tk.NORMAL)
            else:
                self.model_btn.config(state=tk.DISABLED)
        self.root.after(0, update_ui)


    def load_model_files(self):
        """Loads the model and label encoder."""
        try:
            self.model = tf.keras.models.load_model(MODEL_FILE)
            with open(ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.model_trained = True
            self.set_status("Model and encoder loaded successfully.")
            return True
        except Exception as e:
            errmsg = f"Error loading model/encoder: {e}\nTrain the model first."
            print(f"ERROR: {errmsg}")
            # Don't show messagebox here, let toggle_model handle it if user initiated
            # messagebox.showerror("Error", errmsg)
            self.model = None
            self.label_encoder = None
            self.model_trained = False
            self.set_status("Failed to load model.")
            return False


    def toggle_model(self):
        """Toggle model recognition state."""
        if self.model_active:
            self.model_active = False
            self.model_btn.config(text="Start Recognition")
            self.set_status("Recognition stopped.")
        else:
            if not self.model_trained or not self.model:
                self.set_status("Loading model...")
                if not self.load_model_files():
                    messagebox.showerror("Error", "Could not load model files. Please train the model.")
                    self.set_status("Model load failed. Train first.")
                    return # Stop if loading failed

            # Check if label encoder is loaded (should be if model loaded)
            if not self.label_encoder:
                 messagebox.showerror("Error", "Label encoder not found or loaded.")
                 self.set_status("Label encoder missing.")
                 return

            self.model_active = True
            self.model_btn.config(text="Stop Recognition")
            self.set_status("Recognition active.")


    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        self.debug_btn.config(
            text="Disable Debug" if self.debug_mode else "Enable Debug"
        )
        self.set_status(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}.")


    def load_data(self):
        """Load saved gesture data"""
        self.set_status(f"Loading data from {DATA_FILE}...")
        loaded_data = {}
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    raw_data = json.load(f)

                # Verify loaded data structure and feature size
                verified_data = {}
                total_samples = 0
                skipped_samples = 0
                for gesture, samples in raw_data.items():
                    if isinstance(samples, list):
                        # Filter samples based on the current FEATURE_SIZE
                        valid_samples = [s for s in samples if isinstance(s, list) and len(s) == self.feature_size]
                        if valid_samples:
                            verified_data[gesture] = valid_samples
                            total_samples += len(valid_samples)
                        skipped_count = len(samples) - len(valid_samples)
                        if skipped_count > 0:
                            print(f"Warning: Skipped {skipped_count} samples for gesture '{gesture}' due to incorrect feature size.")
                            skipped_samples += skipped_count
                    else:
                         print(f"Warning: Invalid data format for gesture '{gesture}'. Skipping.")


                loaded_data = verified_data
                status_msg = f"Loaded {total_samples} samples."
                if skipped_samples > 0:
                    status_msg += f" Skipped {skipped_samples} due to feature size mismatch."
                self.set_status(status_msg)
            else:
                 self.set_status("Data file not found. Record new samples.")

        except json.JSONDecodeError:
            self.set_status(f"Error: Could not decode JSON from {DATA_FILE}.")
            messagebox.showerror("Error", f"Failed to read data file {DATA_FILE}. It might be corrupted.")
        except Exception as e:
            self.set_status(f"Error loading data: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred while loading data:\n{e}")

        return loaded_data


    def save_data(self):
        """Save gesture data"""
        self.set_status(f"Saving data to {DATA_FILE}...")
        try:
            # Verify data before saving (ensure lists of lists with correct feature size)
            verified_data = {}
            total_samples = 0
            for gesture, samples in self.data.items():
                valid_samples = [s for s in samples if isinstance(s, list) and len(s) == self.feature_size]
                if valid_samples:
                    verified_data[gesture] = valid_samples
                    total_samples += len(valid_samples)
                elif gesture in self.data: # Only warn if the gesture existed but now has no valid samples
                     print(f"Warning: No valid samples with size {self.feature_size} found for gesture '{gesture}'. It will not be saved.")


            with open(DATA_FILE, 'w') as f:
                json.dump(verified_data, f, indent=2) # Use indent for readability

            self.data = verified_data # Update internal data to only contain verified items
            self.set_status(f"Data saved successfully ({total_samples} samples).")
            return True
        except Exception as e:
            error_msg = f"Error saving data: {e}"
            print(f"ERROR: {error_msg}")
            messagebox.showerror("Error", f"Failed to save data to {DATA_FILE}:\n{e}")
            self.set_status("Error saving data.")
            return False


    def update_data_list(self):
        """Update data display in UI"""
        if not self.root or not self.data_list: return # UI not ready
        try:
            self.data_list.delete(0, tk.END)
            sorted_gestures = sorted(self.data.keys())
            for gesture in sorted_gestures:
                samples = self.data[gesture]
                self.data_list.insert(tk.END, f"{gesture}: {len(samples)} samples")
        except Exception as e:
            print(f"Error updating data list: {e}")


    def delete_selected(self):
        """Delete selected gesture data"""
        selection = self.data_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No gesture selected in the list.")
            return

        selected_item = self.data_list.get(selection[0])
        gesture = selected_item.split(':')[0]

        if gesture in self.data:
            confirm = messagebox.askyesno("Confirm Deletion",
                                         f"Are you sure you want to delete all samples for gesture '{gesture}'?")
            if confirm:
                del self.data[gesture]
                if self.save_data():
                    self.set_status(f"Deleted samples for gesture '{gesture}'.")
                    self.update_data_list()
                    # If model was trained, it's now potentially outdated
                    self.model_trained = False # Mark model as needing retraining
                    self.model_active = False
                    self.model_btn.config(state=tk.DISABLED, text="Start Recognition")
                    self.set_status(f"Deleted '{gesture}'. Model needs retraining.")
                else:
                     self.set_status(f"Failed to save data after deleting '{gesture}'.")


    def quit_app(self):
        """Clean up and quit application"""
        if not self.running: # Avoid double execution
             return
        print("Quit signal received. Cleaning up...")
        self.running = False # Signal threads to stop

        # Stop camera capture first
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            print("Camera released.")

        # Wait for camera thread to finish
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            print("Waiting for camera thread to join...")
            self.camera_thread.join(timeout=2.0) # Wait max 2 seconds
            if self.camera_thread.is_alive():
                print("Warning: Camera thread did not terminate gracefully.")

        # Destroy OpenCV windows (though we embedded video, this is good practice)
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

        # Destroy Tkinter window
        if self.root:
            self.root.destroy()
            print("Tkinter window destroyed.")
        print("Application quit.")


    def run(self):
        """Start the application"""
        if self.root: # Check if UI was created successfully
            self.root.mainloop()
        else:
             print("UI initialization failed. Cannot run application.")

if __name__ == "__main__":
    try:
        app = HandGestureRecognizer()
        app.run()
    except Exception as e:
         print(f"\n--- An unhandled error occurred ---")
         import traceback
         traceback.print_exc()
         messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nPlease check the console for details.")
