import cv2
import mediapipe as mp
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, PanedWindow
# Suppress TensorFlow INFO/WARNING messages (must be before tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading
from datetime import datetime
import time
from PIL import Image, ImageTk, ImageOps
import pickle
import traceback

# --- Constants ---
DATA_FILE = 'gesture_data_normalized.json'
MODEL_FILE = 'gesture_model_normalized.h5'
ENCODER_FILE = 'label_encoder_normalized.pkl'
FEATURE_SIZE = 20 * 3 # 20 landmarks * 3 coordinates (relative to wrist)
GESTURE_OPTIONS = [chr(i) for i in range(65, 91)] # A-Z
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 900

# --- Default size, position, and ratio based on user preference ---
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 1000
DEFAULT_POS_X = 550
DEFAULT_POS_Y = 300
INITIAL_VIDEO_PANE_RATIO = 0.75
# --- End Default Layout Settings ---

PREDICTION_CONFIDENCE_THRESHOLD = 0.65
PLACEHOLDER_BG_COLOR = (64, 64, 64) # Dark gray background when camera off

class HandGestureRecognizer:
    def __init__(self):
        # Initialize state variables first
        self.recording = False
        self.model_active = False
        self.model_trained = False # Tracks if model *files* exist
        self.model_loaded = False  # Tracks if model is actually loaded in memory
        self.current_samples = []
        self.data = {}
        self.model = None
        self.label_encoder = None
        self.root = None
        self.cap = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        self.camera_thread = None
        self.running = False
        self.last_frame_size = (0, 0)
        self.camera_width = 640 # Default, updated on init
        self.camera_height = 480 # Default, updated on init
        self.camera_aspect_ratio = 640 / 480 # Default
        self.placeholder_bg = None # Placeholder background image

        # Tkinter Variable Placeholders (initialized in create_ui)
        self.show_camera_feed = None
        self.show_wireframe = None
        self.debug_mode = None
        self.gesture_var = None
        self.gesture_combo = None # Keep ref to combobox

        self._init_ui_elements_to_none()

        try:
            print("Initializing MediaPipe...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.feature_size = FEATURE_SIZE
            print("MediaPipe initialized.")

            # Create UI (sets up self.root, Tkinter Vars, and applies default geometry)
            print("Creating UI...")
            self.create_ui()
            print("UI created and initial geometry set.")

            # Initialize Camera (gets resolution)
            self.init_camera()

            # Load data and check for model files *after* UI exists
            print("Loading data...")
            self.data = self.load_data()
            self.update_data_list()
            print("Checking for existing model...")
            self.check_model_files_exist() # Checks files, doesn't load model

            # Set initial sash position based on desired layout ratio (using root.after)
            self.set_initial_sash_position()

            # Start camera thread
            print("Starting camera thread...")
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            # Handle window closing
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            print("Initialization complete. Starting main loop.")

        except Exception as e:
            print("\n--- Error during Initialization ---")
            traceback.print_exc()
            try:
                 messagebox.showerror("Initialization Error", f"Failed to initialize application:\n{e}\n\nCheck camera, dependencies, and console.")
            except tk.TclError: # If UI failed even earlier
                 print("UI Error: Could not display messagebox.")
            self.running = False
            if self.root:
                try: self.root.destroy()
                except: pass
            # No run() call if initialization fails

    def _init_ui_elements_to_none(self):
        """ Initialize all UI variable placeholders to None """
        self.root = None
        self.video_label = None
        self.record_btn = None
        self.model_btn = None
        self.train_btn = None
        self.debug_check = None
        self.camera_check = None
        self.wireframe_check = None
        self.status_var = None
        self.sample_label = None
        self.data_list = None
        self.gesture_var = None
        self.gesture_combo = None
        self.help_btn = None
        self.quit_btn = None
        self.main_pane = None
        self.save_button_ref = None
        self.clear_button_ref = None
        self.delete_button_ref = None


    def init_camera(self):
        """Initializes the camera and gets its resolution."""
        self.set_status("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            self.cap = None
            raise ConnectionError("Could not open camera. Please check connection.")

        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.camera_width > 0 and self.camera_height > 0:
            self.camera_aspect_ratio = self.camera_width / self.camera_height
            print(f"Camera initialized: {self.camera_width}x{self.camera_height} (AR: {self.camera_aspect_ratio:.2f})")
            self.placeholder_bg = np.full((self.camera_height, self.camera_width, 3), PLACEHOLDER_BG_COLOR, dtype=np.uint8)
        else:
            self.camera_width = 640; self.camera_height = 480; self.camera_aspect_ratio = 640 / 480
            self.placeholder_bg = np.full((self.camera_height, self.camera_width, 3), PLACEHOLDER_BG_COLOR, dtype=np.uint8)
            print("Warn: Cam res failed, using 640x480.")

        ret, _ = self.cap.read()
        if not ret:
            self.cap.release(); self.cap = None; raise ConnectionError("Could not read initial frame.")
        self.set_status("Camera initialized.")


    def set_status(self, text):
        """Update the status bar safely."""
        if self.root and self.status_var:
            try: self.root.after(0, lambda: self.status_var.set(text))
            except tk.TclError as e:
                 if "application has been destroyed" not in str(e): print(f"Status update err: {e}")


    def create_ui(self):
        """Create the user interface and set initial geometry."""
        self.root = tk.Tk(); self.root.title("Hand Gesture Recognition v2.3") # Version bump
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)

        self.main_pane = PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, background="#f0f0f0")
        self.main_pane.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Left Pane (Video)
        video_outer_frame = ttk.Frame(self.main_pane, style='Video.TFrame')
        video_outer_frame.grid_rowconfigure(0, weight=1); video_outer_frame.grid_columnconfigure(0, weight=1)
        style = ttk.Style(self.root); style.configure('Video.TFrame', background='black')
        self.video_label = ttk.Label(video_outer_frame, anchor=tk.CENTER, background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew"); self.main_pane.add(video_outer_frame) # Removed stretch

        # Right Pane (Controls)
        controls_scrollable_frame = ttk.Frame(self.main_pane)
        controls_scrollable_frame.grid_rowconfigure(0, weight=1); controls_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.main_pane.add(controls_scrollable_frame)
        controls_frame = ttk.Frame(controls_scrollable_frame, padding="10"); controls_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame.grid_columnconfigure(0, weight=1); controls_frame.grid_rowconfigure(3, weight=1); controls_frame.grid_rowconfigure(4, weight=0)

        # Data Collection (with Nav Buttons)
        collect_frame = ttk.LabelFrame(controls_frame, text="Data Collection", padding="10")
        collect_frame.grid(row=0, column=0, sticky="new", pady=(0, 10))
        collect_frame.grid_columnconfigure(0, weight=0); collect_frame.grid_columnconfigure(1, weight=0)
        collect_frame.grid_columnconfigure(2, weight=1); collect_frame.grid_columnconfigure(3, weight=0) # Configure columns

        ttk.Label(collect_frame, text="Gesture:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        prev_btn = ttk.Button(collect_frame, text="<", width=2, command=self.select_previous_gesture)
        prev_btn.grid(row=0, column=1, sticky=tk.E, padx=(0, 2))
        self.gesture_var = tk.StringVar();
        self.gesture_combo = ttk.Combobox(collect_frame, textvariable=self.gesture_var, values=GESTURE_OPTIONS, state="readonly", width=8)
        self.gesture_combo.grid(row=0, column=2, sticky="ew", pady=2)
        if GESTURE_OPTIONS: self.gesture_combo.current(0) # Default 'A'
        next_btn = ttk.Button(collect_frame, text=">", width=2, command=self.select_next_gesture)
        next_btn.grid(row=0, column=3, sticky=tk.W, padx=(2, 0))

        self.sample_label = ttk.Label(collect_frame, text="Samples: 0");
        self.sample_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5,2)) # Span 4
        self.record_btn = ttk.Button(collect_frame, text="⏺ Start Recording", command=self.toggle_recording);
        self.record_btn.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(5, 5)) # Span 4
        btn_frame1 = ttk.Frame(collect_frame); btn_frame1.grid(row=3, column=0, columnspan=4, sticky="ew"); # Span 4
        btn_frame1.grid_columnconfigure(0, weight=1); btn_frame1.grid_columnconfigure(1, weight=1)
        self.save_button_ref = ttk.Button(btn_frame1, text="💾 Save Samples", width=13, command=self.save_samples); self.save_button_ref.grid(row=0, column=0, sticky='ew', padx=(0,2), pady=2)
        self.clear_button_ref = ttk.Button(btn_frame1, text="🗑 Clear Current", width=13, command=self.clear_samples); self.clear_button_ref.grid(row=0, column=1, sticky='ew', padx=(2,0), pady=2)

        # Model Training & Recognition
        model_frame = ttk.LabelFrame(controls_frame, text="Model", padding="10")
        model_frame.grid(row=1, column=0, sticky="new", pady=5); model_frame.grid_columnconfigure(0, weight=1)
        self.train_btn = ttk.Button(model_frame, text="🧠 Train Model", command=self.start_training_thread); self.train_btn.grid(row=0, column=0, sticky="ew", pady=3)
        self.model_btn = ttk.Button(model_frame, text="▶️ Start Recognition", command=self.toggle_model); self.model_btn.grid(row=1, column=0, sticky="ew", pady=3); self.model_btn.config(state=tk.DISABLED)

        # Initialize Tkinter Variables
        self.show_camera_feed = tk.BooleanVar(value=True)
        self.show_wireframe = tk.BooleanVar(value=True)
        self.debug_mode = tk.BooleanVar(value=False)

        # View Options
        view_frame = ttk.LabelFrame(controls_frame, text="View Options", padding="10")
        view_frame.grid(row=2, column=0, sticky="new", pady=5); view_frame.grid_columnconfigure(0, weight=1)
        self.camera_check = ttk.Checkbutton(view_frame, text="Show Camera Feed", variable=self.show_camera_feed, command=self._on_view_toggle_changed); self.camera_check.grid(row=0, column=0, sticky="w", pady=2)
        self.wireframe_check = ttk.Checkbutton(view_frame, text="Show Wireframe", variable=self.show_wireframe, command=self._on_view_toggle_changed); self.wireframe_check.grid(row=1, column=0, sticky="w", pady=2)
        self.debug_check = ttk.Checkbutton(view_frame, text="Show Debug Info", variable=self.debug_mode, command=self._on_view_toggle_changed); self.debug_check.grid(row=2, column=0, sticky="w", pady=2)

        # Data Management
        data_frame = ttk.LabelFrame(controls_frame, text="Saved Data", padding="10")
        data_frame.grid(row=3, column=0, sticky="nsew", pady=5); data_frame.grid_rowconfigure(0, weight=1); data_frame.grid_columnconfigure(0, weight=1)
        self.data_list = tk.Listbox(data_frame, height=6); self.data_list.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_list.yview); scrollbar.grid(row=0, column=1, sticky="ns")
        self.data_list['yscrollcommand'] = scrollbar.set
        self.delete_button_ref = ttk.Button(data_frame, text="❌ Delete Selected", command=self.delete_selected); self.delete_button_ref.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Application Controls
        app_ctrl_frame = ttk.Frame(controls_frame, padding=(0, 5, 0, 0))
        app_ctrl_frame.grid(row=4, column=0, sticky="sew", pady=(10, 0))
        app_ctrl_frame.grid_columnconfigure(0, weight=1); app_ctrl_frame.grid_columnconfigure(1, weight=1)
        self.help_btn = ttk.Button(app_ctrl_frame, text="❓ Help", width=10, command=self.show_help); self.help_btn.grid(row=0, column=0, sticky="sw", padx=(0, 5))
        self.quit_btn = ttk.Button(app_ctrl_frame, text="Quit", width=10, command=self.quit_app); self.quit_btn.grid(row=0, column=1, sticky="se", padx=(5, 0))

        # Status Bar
        self.status_var = tk.StringVar(); self.status_var.set("Initializing...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5,2))
        status_bar.grid(row=1, column=0, sticky="ew")

        # Set Final Window Geometry
        try:
            self.root.update_idletasks()
            w=DEFAULT_WINDOW_WIDTH; h=DEFAULT_WINDOW_HEIGHT; x=DEFAULT_POS_X; y=DEFAULT_POS_Y
            try: sw=self.root.winfo_screenwidth(); sh=self.root.winfo_screenheight(); x=max(0,min(x,sw-w)); y=max(0,min(y,sh-h))
            except tk.TclError: pass
            self.root.geometry(f'{w}x{h}+{x}+{y}'); print(f"Set initial geometry: {w}x{h}+{x}+{y}")
        except Exception as e: print(f"Warn: Win pos err ({e})"); self.root.geometry(f'{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}')

        self._on_view_toggle_changed(); self.set_status("Ready.")


    def set_initial_sash_position(self):
        """Sets the initial position of the PanedWindow sash using root.after()."""
        if not self.root or not self.main_pane: return
        try:
            target_video_width = int(DEFAULT_WINDOW_WIDTH * INITIAL_VIDEO_PANE_RATIO)
            min_controls_width = 250
            max_allowed_video_width = DEFAULT_WINDOW_WIDTH - min_controls_width
            final_sash_pos = max(100, min(target_video_width, max_allowed_video_width))
            print(f"Debug: Scheduling sash placement at x = {final_sash_pos}px.")
            self.root.after(50, lambda pos=final_sash_pos: self._place_sash(pos))
        except Exception as e: print(f"Error calculating sash pos: {e}"); traceback.print_exc()


    def _place_sash(self, position):
        """Helper function to actually place the sash, called via root.after."""
        if not self.root or not self.main_pane: return
        try:
            print(f"Debug: _place_sash executing with position={position}")
            if len(self.main_pane.panes()) >= 1:
                 self.main_pane.sash_place(0, position, 0)
                 print(f"Initial sash placed via 'after' at x = {position}px.")
                 self.root.update_idletasks()
                 sash_verify = self.main_pane.sash_coord(0)
                 print(f"Debug: Sash verify after 'after' place: {sash_verify}")
                 if sash_verify and abs(sash_verify[0] - position) > 15:
                      print(f"WARN: Sash verify FAILED! Expected {position}, got {sash_verify[0]}.")
                 else: print(f"Debug: Sash verify OK.")
            else: print("Error: Panes not ready in _place_sash.")
        except tk.TclError as e: print(f"Warn: Sash TclError in _place_sash ({e}).")
        except Exception as e: print(f"Error in _place_sash: {e}"); traceback.print_exc()


    def _on_view_toggle_changed(self):
        """Handles logic when view option checkboxes are toggled."""
        if self.wireframe_check:
            is_cam_on = self.show_camera_feed.get()
            self.wireframe_check.config(state=tk.NORMAL if is_cam_on else tk.DISABLED)
            if not is_cam_on and not self.show_wireframe.get():
                self.show_wireframe.set(True)


    def select_previous_gesture(self):
        """Selects the previous gesture in the list, wrapping around."""
        if not self.gesture_var or not GESTURE_OPTIONS: return
        try:
            current_gesture = self.gesture_var.get()
            current_index = GESTURE_OPTIONS.index(current_gesture)
            previous_index = (current_index - 1 + len(GESTURE_OPTIONS)) % len(GESTURE_OPTIONS)
            self.gesture_var.set(GESTURE_OPTIONS[previous_index])
        except ValueError:
            print(f"Warn: Current gesture '{current_gesture}' not in options.")
            if GESTURE_OPTIONS: self.gesture_var.set(GESTURE_OPTIONS[0])
        except Exception as e: print(f"Error selecting previous gesture: {e}")


    def select_next_gesture(self):
        """Selects the next gesture in the list, wrapping around."""
        if not self.gesture_var or not GESTURE_OPTIONS: return
        try:
            current_gesture = self.gesture_var.get()
            current_index = GESTURE_OPTIONS.index(current_gesture)
            next_index = (current_index + 1) % len(GESTURE_OPTIONS)
            self.gesture_var.set(GESTURE_OPTIONS[next_index])
        except ValueError:
            print(f"Warn: Current gesture '{current_gesture}' not in options.")
            if GESTURE_OPTIONS: self.gesture_var.set(GESTURE_OPTIONS[0])
        except Exception as e: print(f"Error selecting next gesture: {e}")


    def show_help(self):
        """Displays a help message box."""
        help_text = """
Hand Gesture Recognition App - Usage Guide

1. Collect Data:
   - Select a gesture (A-Z) using the dropdown or < > buttons.
   - Click 'Start Recording'.
   - Make the gesture clearly in front of the camera. Move your hand slightly to capture variations.
   - Click 'Stop Recording'. The number of 'Samples' collected is shown.
   - Click 'Save Samples' to store the data for the selected gesture.
   - Click 'Clear Current' to discard the recording buffer without saving.
   - Repeat for all gestures you want to recognize (at least 2). Aim for 50+ samples per gesture.

2. Train Model:
   - Once you have sufficient data for multiple gestures, click 'Train Model'.
   - Training progress will be shown in the console and status bar. This may take some time.
   - A message will confirm when training is complete.

3. Recognize Gestures:
   - After training, click 'Start Recognition'.
   - Perform the trained gestures in front of the camera.
   - The predicted gesture will be displayed on the video feed if the confidence is high enough.

4. View Options:
   - 'Show Camera Feed': Toggles the live video on/off. If off, only wireframe (if detected) is shown on a gray background.
   - 'Show Wireframe': Toggles the hand landmark drawing on/off. (Forced ON if camera feed is OFF).
   - 'Show Debug Info': Toggles the display of prediction confidence scores.

5. Manage Data:
   - The 'Saved Data' list shows gestures and sample counts.
   - Select a gesture in the list and click 'Delete Selected' to remove its data (requires confirmation).
   - Deleting data requires retraining the model.

6. Quit:
   - Click 'Quit' or close the window ('X') to exit.

Tips:
- Ensure good lighting and a relatively plain background.
- Keep only your hand in the frame when recording/recognizing.
- More data generally leads to better accuracy.
"""
        messagebox.showinfo("How to Use", help_text)


    def check_model_files_exist(self):
        """Checks if model files exist and updates UI state."""
        model_found=os.path.exists(MODEL_FILE); enc_found=os.path.exists(ENCODER_FILE)
        self.model_trained = model_found and enc_found
        if self.model_trained:
            if self.model_btn: self.model_btn.config(state=tk.NORMAL)
            self.set_status("Model files found. Ready.")
        else:
            self.model_loaded=False
            if self.model_btn: self.model_btn.config(state=tk.DISABLED)
            missing=[f for f,found in [(MODEL_FILE,model_found),(ENCODER_FILE,enc_found)] if not found]
            if missing: self.set_status(f"Missing:{','.join(map(os.path.basename,missing))}. Train.")
            else: self.set_status("Train model.")


    def camera_loop(self):
        """Main camera processing loop."""
        last_time = time.time()
        while self.running:
            if not self.cap or not self.cap.isOpened(): time.sleep(0.5); continue

            # --- Step 1: ALWAYS read the frame from the camera ---
            ret, frame = self.cap.read()
            if not ret: self.set_status("Cam Read Err."); time.sleep(0.5); continue
            frame = cv2.flip(frame, 1) # Flip horizontally

            # --- Step 2: ALWAYS process the REAL frame with MediaPipe ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = None; gesture_text = ""; conf_text = ""; landmarks = None # Reset results for each frame
            try:
                 results = self.hands.process(rgb_frame)
            except Exception as e:
                 print(f"MP Err: {e}"); time.sleep(0.1); continue

            # --- Step 3: Extract features/predict IF landmarks were found ---
            if results and results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0] # Store landmarks if found
                features = self.extract_features(landmarks)
                if features is not None:
                    # Recording logic
                    if self.recording:
                        self.current_samples.append(features)
                        # Schedule UI update safely
                        if self.root: self.root.after(0, self.update_sample_count)

                    # Prediction logic
                    if self.model_active and self.model_loaded:
                        try:
                            pred = self.model.predict(np.array([features]), verbose=0)
                            idx=np.argmax(pred[0]); conf=float(np.max(pred[0]))
                            if conf >= PREDICTION_CONFIDENCE_THRESHOLD:
                                gesture = self.label_encoder.inverse_transform([idx])[0]
                                gesture_text = f"Gesture: {gesture}"
                                if self.debug_mode.get(): conf_text = f"Conf: {conf:.2f}"
                            elif self.debug_mode.get(): gesture_text="?"; conf_text=f"Conf: {conf:.2f}"
                        except Exception as e: print(f"Pred Err: {e}"); gesture_text = "Err"

            # --- Step 4: Choose the frame to DISPLAY based on the checkbox ---
            if self.show_camera_feed.get():
                display_frame = frame.copy() # Show the live camera feed
            else:
                # Show the placeholder background
                if self.placeholder_bg is None: # Safety check
                    self.placeholder_bg = np.full((self.camera_height, self.camera_width, 3), PLACEHOLDER_BG_COLOR, dtype=np.uint8)
                display_frame = self.placeholder_bg.copy()

            # --- Step 5: Draw wireframe on the DISPLAY frame IF enabled AND landmarks were found ---
            if self.show_wireframe.get() and landmarks: # Check if landmarks were detected in Step 3
                 self.mp_draw.draw_landmarks(display_frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

            # --- Step 6: Draw overlay text on the DISPLAY frame ---
            self.draw_overlay(display_frame, gesture_text, conf_text)

            # --- Step 7: Update the UI label with the chosen DISPLAY frame ---
            try:
                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.update_video_label_with_resize(img_rgb)
            except Exception as e:
                 if self.running and self.root: print(f"Vid Update Err: {e}"); time.sleep(0.1)

            # Limit FPS
            elapsed=time.time()-last_time; time.sleep(max(0,(1/30)-elapsed)); last_time=time.time()
        print("Camera loop stopped.")


    def draw_overlay(self, frame, gesture_text, confidence_text):
        """Draws overlays on the frame."""
        alpha=0.6; thick=2; pad=5; font=cv2.FONT_HERSHEY_SIMPLEX; scale_g=0.8; scale_c=0.7
        if self.recording: cv2.circle(frame,(30,30),15,(0,0,255),-1); cv2.putText(frame,"REC",(55,38),font,0.7,(255,255,255),thick)
        if self.model_active and gesture_text:
            y=30; (gw,gh),_=cv2.getTextSize(gesture_text,font,scale_g,thick); (cw,ch),_=(0,0),0
            if self.debug_mode.get() and confidence_text: (cw,ch),_=cv2.getTextSize(confidence_text,font,scale_c,thick)
            maxw=max(gw,cw); bh=gh+(ch+pad if self.debug_mode.get() and confidence_text else 0)+pad*2
            x1,y1=10,y-pad; x2,y2=x1+maxw+pad*2,y1+bh
            try:
                sub=frame[y1:y2,x1:x2];
                if sub.shape[0]>0 and sub.shape[1]>0:
                    rect=np.zeros(sub.shape,dtype=np.uint8); res=cv2.addWeighted(sub,1-alpha,rect,alpha,0)
                    frame[y1:y2,x1:x2]=res; cv2.putText(frame,gesture_text,(x1+pad,y1+gh+pad),font,scale_g,(0,255,0),thick)
                    if self.debug_mode.get() and confidence_text: cv2.putText(frame,confidence_text,(x1+pad,y1+gh+ch+pad*2),font,scale_c,(200,200,200),thick)
            except Exception as e: print(f"Warn: Overlay Err: {e}")


    def update_video_label_with_resize(self, img_rgb):
        """ Updates video label, resizing image."""
        if not self.root or not self.video_label or not hasattr(self.video_label,'winfo_width'): return
        try:
            w=self.video_label.winfo_width(); h=self.video_label.winfo_height()
            if w<=1 or h<=1: return
            img=Image.fromarray(img_rgb); img_res=ImageOps.pad(img,(w,h),color='black')
            imgtk=ImageTk.PhotoImage(image=img_res)
            def _upd():
                if self.video_label: self.video_label.imgtk=imgtk; self.video_label.config(image=imgtk)
            if self.root: self.root.after(0,_upd)
        except tk.TclError: pass
        except Exception as e: print(f"PhotoImage err:{e}")


    def extract_features(self, landmarks):
        """Extract and normalize features."""
        if not landmarks or not landmarks.landmark: return None
        try:
            w=landmarks.landmark[0]; wx,wy,wz=w.x,w.y,w.z; f=[]
            for i in range(1,21): lm=landmarks.landmark[i]; f.extend([lm.x-wx,lm.y-wy,lm.z-wz])
            if len(f)!=self.feature_size: f.extend([0.0]*(self.feature_size-len(f))); f=f[:self.feature_size]
            return np.array(f,dtype=np.float32)
        except IndexError: print("Warn: Landmark IdxErr."); return None
        except Exception as e: print(f"Feature Err:{e}"); return None


    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording: self.recording=False; self.set_status("Rec stopped."); btn_txt="⏺ Start Rec"
        else:
            g=self.gesture_var.get();
            if not g: messagebox.showwarning("Input","Select gesture."); return
            self.recording=True; self.set_status(f"Rec: {g}"); btn_txt="⏹ Stop Rec"
        if self.record_btn: self.record_btn.config(text=btn_txt)


    def save_samples(self):
        """Save current samples and clear buffer without confirmation.""" # Docstring updated
        if not self.current_samples: messagebox.showwarning("No Data", "No samples!"); return
        g=self.gesture_var.get();
        if not g: messagebox.showwarning("Input", "Select gesture!"); return
        valid=[s.tolist() for s in self.current_samples if isinstance(s,np.ndarray) and s.shape==(self.feature_size,)]
        if not valid: messagebox.showwarning("Error", "No valid samples."); self.clear_samples(); return # Keep call here if samples invalid
        cnt=len(valid);
        if g not in self.data: self.data[g]=[]
        self.data[g].extend(valid)

        if self.save_data(): # If saving the file worked
            self.set_status(f"Saved {cnt} for '{g}'. Buffer cleared.")
            # --- FIX: Directly clear list and update UI ---
            self.current_samples = []
            self.update_sample_count()
            # --- END FIX ---
            self.update_data_list() # Update the saved data list display
        else:
            # Save failed, maybe don't clear buffer in this case? Or notify user.
            self.set_status(f"Save failed for '{g}'. Buffer not cleared.")



    def clear_samples(self):
        """Clear current samples ONLY if confirmed by user."""
        if not self.current_samples:
            # messagebox.showinfo("Info", "Recording buffer is already empty.") # Optional
            return # Nothing to clear

        num_samples_to_clear = len(self.current_samples) # Store count before dialog

        confirm = messagebox.askyesno("Confirm Clear",
                                      f"Discard the current {num_samples_to_clear} recorded samples?\nThis cannot be undone.")

        # Add a debug print to see exactly what the messagebox returns
        print(f"Debug: messagebox.askyesno returned: {confirm} (Type: {type(confirm)})")

        # Explicitly check for True. False (No) and None (often 'X' or Esc) go to else.
        if confirm is True:
            print("Debug: User confirmed clear.")
            self.current_samples = []
            self.update_sample_count()
            self.set_status("Recording buffer cleared.")
        elif confirm is None:
            # Handle the case when the user closes the window without selecting 'Yes' or 'No'
            print("Debug: User closed the message box without confirming.")
            self.set_status("Clear operation cancelled.")
        elif confirm is False:
            # Handles False (No) and potentially None ('X' or Esc)
            print(f"Debug: User cancelled clear (confirm value was {confirm}).")
            self.set_status("Clear operation cancelled.")




    def update_sample_count(self):
        """Update sample counter in UI"""
        if self.sample_label:
            try: self.sample_label.config(text=f"Samples: {len(self.current_samples)}")
            except tk.TclError: pass


    def start_training_thread(self):
        """Starts the training process"""
        if not self.data: messagebox.showwarning("No Data", "No training data!"); return
        if messagebox.askyesno("Confirm", "Overwrite model?"):
            self._set_controls_state(tk.DISABLED); self.set_status("Starting training...")
            threading.Thread(target=self.train_model, daemon=True).start()


    def _set_controls_state(self, state):
        """ Helper to enable/disable controls """
        # Include button references stored during UI creation
        widgets = [self.train_btn, self.record_btn, self.model_btn, self.help_btn,
                   self.quit_btn, self.camera_check, self.wireframe_check, self.debug_check,
                   self.data_list, self.save_button_ref, self.clear_button_ref, self.delete_button_ref,
                   self.gesture_combo] # Also disable/enable combobox
        # Find prev/next buttons if they exist (might need to store refs too)
        # Safer way: iterate through children of collect_frame? Or store refs.
        # Let's assume we don't disable prev/next for now. Add if needed.

        for w in widgets:
            if w: 
                try: w.config(state=state)
                except tk.TclError: pass


    def train_model(self):
        """Train the model (run in thread)."""
        try:
            X,y_labels=[],[]; min_s=10; v_g_c=0; counts={}
            for g,s in self.data.items():
                valid=[i for i in s if isinstance(i,list) and len(i)==self.feature_size]
                counts[g]=len(valid)
                if counts[g]>=min_s: X.extend(valid); y_labels.extend([g]*counts[g]); v_g_c+=1
                elif counts[g]>0: print(f"Info: Skip '{g}' ({counts[g]}<{min_s}).")
            if not X or v_g_c<2:
                err=f"Train Fail: Need >= {min_s} valid samples for >= 2 gestures."
                print(f"Counts:{ {g:c for g,c in counts.items() if c>0} }");
                if self.root: self.root.after(0,lambda: messagebox.showerror("Error",err)); self.set_status(err); self._reenable_buttons(); return

            X=np.array(X,dtype=np.float32); self.label_encoder=LabelEncoder(); y=self.label_encoder.fit_transform(y_labels)
            num_cls=len(self.label_encoder.classes_); print(f"\n--- Train ---"); print(f"Gest:{list(self.label_encoder.classes_)}")
            print(f"Samples:{len(X)}, Cls:{num_cls}, Feats:{self.feature_size}")

            self.model=tf.keras.Sequential([tf.keras.layers.Input(shape=(self.feature_size,)),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dropout(0.4),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(0.3),tf.keras.layers.Dense(num_cls,activation='softmax')])
            self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

            n_epochs=30; print("\nTraining...");
            class EpochCB(tf.keras.callbacks.Callback):
                def __init__(self,o,t): self.o=o; self.t=t
                def on_epoch_end(self,ep,logs=None):
                    l=logs or {}; a=l.get('accuracy'); v=l.get('val_accuracy'); m=f"Ep {ep+1}/{self.t}|Acc:{a:.3f}"+(f"|Val:{v:.3f}" if v else ""); self.o.set_status(m); print(m)
            hist=self.model.fit(X,y,epochs=n_epochs,batch_size=32,validation_split=0.2,verbose=0,callbacks=[EpochCB(self,n_epochs)])
            print("Train complete.")

            self.model.save(MODEL_FILE);
            with open(ENCODER_FILE,'wb') as f: pickle.dump(self.label_encoder,f)
            print("Saved.")

            acc=hist.history['accuracy'][-1]; val=hist.history['val_accuracy'][-1]
            msg=f"Train OK!\nGest:{','.join(self.label_encoder.classes_)}\nAcc:{acc:.3f}|Val:{val:.3f}"
            self.model_trained=True; self.model_loaded=True
            if self.root: self.root.after(0,lambda: messagebox.showinfo("Success",msg)); self.set_status(f"Trained. Val:{val:.3f}")
        except Exception as e:
            err=f"Train Fail:{e}"; print("\n--- ERROR ---"); traceback.print_exc()
            if self.root: self.root.after(0,lambda: messagebox.showerror("Error",err)); self.set_status(err)
            self.model=None; self.label_encoder=None; self.model_trained=False; self.model_loaded=False
        finally: self._reenable_buttons()


    def _reenable_buttons(self):
        """ Safely re-enable buttons."""
        def upd():
            if not self.root: return
            self._set_controls_state(tk.NORMAL) # Enable most controls
            # Specifically handle model button based on trained status
            if self.model_btn: self.model_btn.config(state=tk.NORMAL if self.model_trained else tk.DISABLED)
            self._on_view_toggle_changed() # Ensure view toggle state is correct
        if self.root: self.root.after(0,upd)


    def load_model_files(self):
        """Loads model and encoder."""
        if self.model_loaded: return True
        if not self.model_trained: self.set_status("Model files missing."); return False
        self.set_status("Loading model..."); print("Loading model...")
        try:
            if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE): raise FileNotFoundError("File disappeared.")
            self.model = tf.keras.models.load_model(MODEL_FILE); print("Loading encoder...");
            with open(ENCODER_FILE,'rb') as f: self.label_encoder = pickle.load(f)
            self.model_loaded=True; self.set_status("Model loaded."); print("Loaded."); return True
        except FileNotFoundError as e: err=f"Load Err:{e}"; print(err); self.set_status(err); self.model=None; self.label_encoder=None; self.model_trained=False; self.model_loaded=False; return False
        except Exception as e: err=f"Load Err:{e}"; print(f"ERROR:{err}"); traceback.print_exc(); 
        if self.root: messagebox.showerror("Load Error",err+"\nCheck file compatibility."); self.model=None; self.label_encoder=None; self.model_loaded=False; self.set_status("Load failed."); return False


    def toggle_model(self):
        """Toggle model recognition."""
        if self.model_active: self.model_active=False; self.set_status("Rec stopped."); btn_txt="▶️ Start Rec"
        else:
            if not self.model_loaded:
                if not self.model_trained: messagebox.showerror("Error","Train first."); return
                if not self.load_model_files(): return
            if not self.model or not self.label_encoder: messagebox.showerror("Error","Model/encoder NA."); return
            self.model_active=True; self.set_status("Rec active."); btn_txt="⏹ Stop Rec"
        if self.model_btn: self.model_btn.config(text=btn_txt)


    def load_data(self):
        """Load saved gesture data"""
        self.set_status(f"Loading {DATA_FILE}..."); loaded={}; total=0; skipped=0
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE,'r') as f: raw=json.load(f)
                for g,s in raw.items():
                    if isinstance(s,list):
                        valid=[i for i in s if isinstance(i,list) and len(i)==self.feature_size]
                        if valid: loaded[g]=valid; total+=len(valid)
                        skip=len(s)-len(valid);
                        if skip>0: print(f"Warn: Skip {skip} for '{g}'."); skipped+=skip
                    else: print(f"Warn: Invalid fmt '{g}'.")
                msg=f"Loaded {total}."+(f" Skip {skipped}." if skipped else ""); self.set_status(msg)
            else: self.set_status("Data file not found.")
        except json.JSONDecodeError: err=f"Decode Err {DATA_FILE}."; self.set_status(err); messagebox.showerror("Load Error",err)
        except Exception as e: err=f"Load Err: {e}"; self.set_status(err); traceback.print_exc(); messagebox.showerror("Load Error",err)
        return loaded


    def save_data(self):
        """Save gesture data"""
        self.set_status(f"Saving {DATA_FILE}...");
        try:
            verified={}; total=0
            for g,s in self.data.items():
                valid=[i for i in s if isinstance(i,list) and len(i)==self.feature_size]
                if valid: verified[g]=valid; total+=len(valid)
                elif g in self.data: print(f"Warn: No valid for '{g}'.")
            with open(DATA_FILE,'w') as f: json.dump(verified,f,indent=2)
            self.data=verified; self.set_status(f"Saved ({total})."); return True
        except Exception as e: err=f"Save Err:{e}"; print(f"ERROR:{err}"); traceback.print_exc(); messagebox.showerror("Save Error",err); self.set_status("Save error."); return False


    def update_data_list(self):
        """Update data display in UI"""
        if not self.root or not self.data_list: return
        try:
            self.data_list.delete(0,tk.END)
            for g in sorted(self.data.keys()): self.data_list.insert(tk.END, f"{g}: {len(self.data.get(g,[]))} samples")
        except tk.TclError: pass
        except Exception as e: print(f"List update err:{e}")


    def delete_selected(self):
        """Delete selected gesture data"""
        if not self.data_list: return
        sel=self.data_list.curselection()
        if not sel: messagebox.showwarning("Warn","No selection."); return
        try: item=self.data_list.get(sel[0]); g=item.split(':')[0]
        except tk.TclError: return
        if g in self.data:
            if messagebox.askyesno("Confirm",f"Delete '{g}'?\nRequires retrain."):
                del self.data[g]
                if self.save_data():
                    self.update_data_list(); self.model_trained=False; self.model_loaded=False
                    self.model=None; self.label_encoder=None
                    if self.model_btn: self.model_btn.config(state=tk.DISABLED,text="▶️ Start Rec")
                    self.set_status(f"Deleted '{g}'. Retrain.")
                else: self.set_status("Save fail after delete.")


    def quit_app(self):
        """Clean up and quit application"""
        if not self.running: return
        print("Quit signal..."); self.running=False; self.set_status("Shutting down...")
        if hasattr(self,'cap') and self.cap: print("Releasing cam..."); cap_ref=self.cap; self.cap=None; 
        try: cap_ref.release(); print("Cam released.") 
        except Exception as e: print(f"Cam release err:{e}")
        if hasattr(self,'camera_thread') and self.camera_thread and self.camera_thread.is_alive(): print("Joining thread..."); self.camera_thread.join(timeout=1.5); print("Thread joined." if not self.camera_thread.is_alive() else "Warn: Thread timeout.")
        try: cv2.destroyAllWindows()
        except Exception as e: print(f"cv2 destroy err:{e}")
        if self.root: print("Destroying Tk..."); root_ref=self.root; self.root=None; 
        try: root_ref.destroy(); print("Tk destroyed.") 
        except Exception as e: print(f"Tk destroy err:{e}")
        print("Shutdown complete.")


    def run(self):
        """Start the application's main loop."""
        if self.root and self.running: self.root.mainloop()
        elif not self.root: print("App run failed: No UI.")
        else: print("App run failed: Shutdown.")


if __name__ == "__main__":
    print(f"Launching Hand Gesture Recognizer v2.3...") # Version bump
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1); print("High DPI set (Win).")
    except Exception: pass

    app = None
    try:
        app = HandGestureRecognizer()
        if app and app.root and app.running: app.run()
    except Exception as e:
         print(f"\n--- FATAL ERROR ---"); traceback.print_exc()
         try: messagebox.showerror("Fatal Error", f"Critical error:\n{e}\n\nCheck console.")
         except: pass
         if app: app.quit_app()
    finally:
         print("Application exiting.")
