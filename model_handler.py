# model_handler.py
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import traceback
import tkinter.messagebox as messagebox

import config # Use config for paths and settings

class ModelHandler:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_size = config.FEATURE_SIZE
        self.is_loaded = False
        self.is_trained_files_exist = False # Tracks file existence
        self._check_files_exist() # Check on init

    def _check_files_exist(self):
        """Checks if model and encoder files exist."""
        model_found = os.path.exists(config.MODEL_FILE)
        encoder_found = os.path.exists(config.ENCODER_FILE)
        self.is_trained_files_exist = model_found and encoder_found
        if not self.is_trained_files_exist:
            self.is_loaded = False # Cannot be loaded if files don't exist

    def load_model(self):
        """Loads the TF model and LabelEncoder."""
        if self.is_loaded:
            # print("Model already loaded.") # Reduce noise
            return True, "Model already loaded."

        self._check_files_exist() # Re-check files before loading
        if not self.is_trained_files_exist:
            return False, "Model files not found. Train first."

        print(f"Loading model from {config.MODEL_FILE}...")
        try:
            # Load the model
            self.model = tf.keras.models.load_model(config.MODEL_FILE)
            print(f"Loading encoder from {config.ENCODER_FILE}...")
            # Load the encoder
            with open(config.ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)

            # --- Crucial Validation ---
            # 1. Check model's expected input shape
            model_input_shape = self.model.input_shape
            # Input shape for functional/sequential models might be (None, feature_size)
            # We care about the last dimension
            loaded_model_feature_size = None
            if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) >= 2:
                 loaded_model_feature_size = model_input_shape[-1]

            if loaded_model_feature_size is None or loaded_model_feature_size != self.feature_size:
                 raise ValueError(f"Model input size mismatch! Expected {self.feature_size}, loaded model expects {loaded_model_feature_size}.")

            # 2. Check model's output shape against encoder classes
            num_classes_encoder = 0
            if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                 num_classes_encoder = len(self.label_encoder.classes_)

            model_output_shape = self.model.output_shape
            num_classes_model = 0
            if isinstance(model_output_shape, (list, tuple)) and len(model_output_shape) >= 2:
                 num_classes_model = model_output_shape[-1]

            if num_classes_encoder <= 0 or num_classes_model <= 0 or num_classes_model != num_classes_encoder:
                 raise ValueError(f"Model output/encoder class mismatch! Model outputs {num_classes_model}, Encoder has {num_classes_encoder} classes.")
            # --- End Validation ---

            self.is_loaded = True
            print("Model and encoder loaded successfully.")
            return True, "Model loaded successfully."

        except FileNotFoundError as e:
            err = f"Load Error: {e}. Files missing."
            print(err); self.reset_model_state(); return False, err
        except ValueError as e: # Specific handling for validation errors
            err = f"Model Load Error: {e}"
            print(err); traceback.print_exc(); messagebox.showerror("Model Load Error", f"{err}\nPlease ensure data/model files match the current configuration or retrain."); self.reset_model_state(); return False, err
        except Exception as e: # Catch other potential errors (like corrupted file)
            err = f"Load Error: {e}"
            print(f"ERROR: {err}"); traceback.print_exc(); messagebox.showerror("Load Error", f"{err}\nCheck file compatibility and TensorFlow/Keras version."); self.reset_model_state(); return False, err

    def reset_model_state(self):
        """Resets model, encoder, and flags."""
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        # self._check_files_exist() # Update file status after reset if needed

    def train_model(self, gesture_data, status_callback=None, progress_callback=None):
        """Trains the model using the provided gesture data."""
        print("Starting model training process...")
        training_success = False
        message = "Training not started."
        temp_model = None # Define outside try
        temp_label_encoder = None # Define outside try
        try:
            X, y_labels = [], []
            counts = {}
            valid_gestures_count = 0

            for gesture, samples in gesture_data.items():
                valid_samples = [s for s in samples if isinstance(s, list) and len(s) == self.feature_size]
                counts[gesture] = len(valid_samples)
                if counts[gesture] >= config.MIN_SAMPLES_PER_CLASS_TRAIN:
                    X.extend(valid_samples)
                    y_labels.extend([gesture] * counts[gesture])
                    valid_gestures_count += 1
                elif counts[gesture] > 0:
                    print(f"Info: Skip '{gesture}' ({counts[gesture]} < {config.MIN_SAMPLES_PER_CLASS_TRAIN} samples).")

            if not X or valid_gestures_count < 2:
                message = f"Train Fail: Need >= {config.MIN_SAMPLES_PER_CLASS_TRAIN} valid samples (size {self.feature_size}) for >= 2 gestures."
                print(f"Counts: { {g:c for g,c in counts.items() if c > 0} }");
                return False, message

            # Label Encoding
            X_np = np.array(X, dtype=np.float32)
            temp_label_encoder = LabelEncoder()
            y_encoded = temp_label_encoder.fit_transform(y_labels)
            num_classes = len(temp_label_encoder.classes_)
            print(f"\n--- Training Configuration ---"); print(f"Gestures: {list(temp_label_encoder.classes_)}")
            print(f"Samples: {len(X_np)}, Classes: {num_classes}, Features: {self.feature_size}")

            # Data Imbalance Check
            valid_gestures = list(temp_label_encoder.classes_)
            valid_counts = [counts.get(g, 0) for g in valid_gestures]
            print(f"Counts for trained classes: {dict(zip(valid_gestures, valid_counts))}")
            if len(valid_counts) >= 2:
                min_c, max_c = min(valid_counts), max(valid_counts)
                if min_c >= config.MIN_SAMPLES_PER_CLASS_TRAIN and max_c / min_c > config.DATA_IMBALANCE_WARN_RATIO:
                     ratio = max_c / min_c; imbalance_warn = f"Train Warn: Data Imbalance (Ratio {ratio:.1f})"
                     print(f"WARNING: {imbalance_warn}")
                     if status_callback: status_callback(imbalance_warn)

            # Model Definition
            temp_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.feature_size,)),
                tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            temp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print(f"\nTraining for {config.N_EPOCHS} epochs...")

            # Epoch Callback
            class EpochCallback(tf.keras.callbacks.Callback):
                def __init__(self, total_epochs, status_cb, progress_cb):
                    super().__init__(); self.total_epochs = total_epochs; self.status_cb = status_cb; self.progress_cb = progress_cb
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}; acc = logs.get('accuracy', 0); val_acc = logs.get('val_accuracy', 0)
                    status_msg = f"Epoch {epoch+1}/{self.total_epochs} | Acc: {acc:.3f}" + (f" | Val: {val_acc:.3f}" if val_acc is not None else "") # Check val_acc is not None
                    if self.status_cb: self.status_cb(status_msg)
                    if self.progress_cb: self.progress_cb(epoch + 1, self.total_epochs)
                    print(status_msg)

            # Fit Model
            history = temp_model.fit(X_np, y_encoded, epochs=config.N_EPOCHS, batch_size=32,
                                     validation_split=0.2, verbose=0,
                                     callbacks=[EpochCallback(config.N_EPOCHS, status_callback, progress_callback)])
            print("Training complete.")

            # Save Model and Encoder
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            temp_model.save(config.MODEL_FILE)
            with open(config.ENCODER_FILE, 'wb') as f: pickle.dump(temp_label_encoder, f)
            print(f"Saved: {config.MODEL_FILE}, {config.ENCODER_FILE}")

            # --- Update Handler State ONLY on full success ---
            self.model = temp_model
            self.label_encoder = temp_label_encoder
            self.is_loaded = True
            self.is_trained_files_exist = True

            final_acc = history.history['accuracy'][-1]; final_val_acc = history.history.get('val_accuracy', [0])[-1] # Use get for val_accuracy
            message = f"Train OK! Gestures: {','.join(self.label_encoder.classes_)}\nAcc:{final_acc:.3f} | Val:{final_val_acc:.3f}"
            training_success = True

        except Exception as e:
            message = f"Training Failed: {e}"
            print("\n--- ERROR during training ---"); traceback.print_exc()
            self.reset_model_state()
            self._check_files_exist()

        return training_success, message

    def predict(self, features):
        """Predicts gesture from features. Returns (gesture_name, confidence) or (None, 0.0)."""
        if not self.is_loaded or self.model is None or self.label_encoder is None:
            return None, 0.0
        if features is None or features.shape != (self.feature_size,):
            return None, 0.0

        try:
            prediction = self.model.predict(np.array([features]), verbose=0)
            if prediction is None or len(prediction) == 0: return None, 0.0 # Handle empty prediction

            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            # Check if index is valid for the encoder
            if predicted_class_idx >= len(self.label_encoder.classes_):
                print(f"Warn: Predicted index {predicted_class_idx} out of bounds for encoder classes {len(self.label_encoder.classes_)}")
                return None, confidence # Return None for gesture, but confidence might be valid

            # Inverse transform only if confidence is high enough
            if confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD:
                gesture_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                return gesture_name, confidence
            else:
                return None, confidence # Gesture is None, return low confidence

        except ValueError as ve: # Handle potential issues with inverse_transform if classes mismatch
            print(f"Prediction Error (ValueError): {ve}")
            return None, 0.0
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, 0.0