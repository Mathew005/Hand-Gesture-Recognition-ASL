# app.py
import tkinter as tk
from tkinter import  messagebox
import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
from PIL import Image, ImageTk, ImageOps
import traceback

# Import components from other files
import config # General configuration and constants
import utils # Utility functions like centering window
from feature_extractor import extract_features # Specific function for feature extraction
from data_manager import load_gesture_data, save_gesture_data # Functions for data IO
from model_handler import ModelHandler # Class to manage the TF model
from ui.main_window import create_main_ui # Function to build the main UI
from ui.review_window import SampleReviewWindow # Class for the review window

class HandGestureRecognizer:
    def __init__(self):
        # --- State Variables ---
        self.recording = False
        self.model_active = False
        self.current_samples = []
        self.data = {} # Dictionary to hold loaded/recorded gesture data {gesture_name: [sample_list]}
        self.running = False # Flag to control main loops and threads

        # --- Core Components ---
        self.model_handler = ModelHandler() # Manages TF model loading, training, prediction
        self.root = None # Main Tkinter window
        self.cap = None # OpenCV VideoCapture object
        self.hands = None # MediaPipe Hands processing object
        self.mp_hands = None # MediaPipe Hands solution module
        self.mp_draw = None # MediaPipe drawing utilities
        self.camera_thread = None # Thread for camera processing loop
        self.last_frame_size = (0, 0) # Track video label size for efficient resizing
        self.camera_width = 640 # Default, updated on init
        self.camera_height = 480 # Default, updated on init
        self.camera_aspect_ratio = 640 / 480 # Default
        self.placeholder_bg = None # Pre-rendered background when camera feed is off

        # --- UI Element & Tkinter Variable Placeholders ---
        # These will be assigned widgets/variables created in ui.main_window.create_main_ui
        self._init_ui_elements_to_none()

        # --- Initialization Sequence ---
        try:
            print(f"Hand Gesture Recognizer v{self.get_version()}")
            print(f"Feature size: {config.FEATURE_SIZE}")

            # Ensure data/model directories exist (can also be handled by data/model managers)
            os.makedirs(config.DATA_DIR, exist_ok=True)
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            print(f"Data dir: '{config.DATA_DIR}', Model dir: '{config.MODEL_DIR}'")

            # Init MediaPipe
            self._initialize_mediapipe()

            # Create Root Window and UI structure
            print("Creating UI..."); self.root = tk.Tk()
            self.root.title(f"Hand Gesture Recognition v{self.get_version()}")
            self.root.minsize(config.MIN_WINDOW_WIDTH, config.MIN_WINDOW_HEIGHT)
            create_main_ui(self.root, self) # Build UI, passing self to link callbacks/widgets
            self._set_initial_window_geometry() # Position and size the window
            print("UI created.")

            # Init Camera
            self.init_camera()

            # Load Data
            print("Loading data..."); self._load_initial_data()

            # Check Model Status
            print("Checking model..."); self._update_model_status_ui() # Updates UI based on handler state

            # Final UI adjustments
            self.set_initial_sash_position() # Adjust internal pane divider
            self._setup_keyboard_shortcuts() # Bind keyboard events

            # Start background processing
            print("Starting camera thread..."); self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            # Setup clean exit
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            print("Initialization complete. Starting main loop.")

        except Exception as e:
            # Catch any error during the setup process
            self._handle_init_error(e)

    def get_version(self):
        """Returns the application version from config."""
        return config.APP_VERSION

    def _init_ui_elements_to_none(self):
        """Initialize placeholders for UI widgets and Tkinter variables."""
        self.main_pane = None; self.video_label = None; self.record_btn = None
        self.model_btn = None; self.train_btn = None; self.debug_check = None
        self.camera_check = None; self.wireframe_check = None; self.status_var = None
        self.sample_label = None; self.data_list = None; self.gesture_var = None
        self.gesture_combo = None; self.help_btn = None; self.quit_btn = None
        self.save_button_ref = None; self.clear_button_ref = None; self.delete_button_ref = None
        self.progress_bar = None; self.review_button_ref = None; self.prev_gesture_btn = None
        self.next_gesture_btn = None
        # Tk Variables initialized in create_main_ui
        self.show_camera_feed = None; self.show_wireframe = None; self.debug_mode = None

    def _initialize_mediapipe(self):
        """Initializes MediaPipe Hands."""
        print("Initializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        print("MediaPipe initialized.")

    def _handle_init_error(self, error):
        """Handles critical errors during initialization."""
        print("\n--- Initialization Error ---"); traceback.print_exc()
        try: messagebox.showerror("Initialization Error", f"Failed:\n{error}\n\nCheck console.")
        except: print("UI Error: Could not display messagebox.") # If Tkinter itself failed
        self.running = False # Prevent loops/threads from starting/continuing
        if self.root:
            try: self.root.destroy() # Attempt to close window if partially created
            except: pass

    def _set_initial_window_geometry(self):
        """Sets the window size and centers it or uses default position."""
        try:
            w = config.DEFAULT_WINDOW_WIDTH; h = config.DEFAULT_WINDOW_HEIGHT
            # Try centering first using the utility function
            centered = utils.center_window(self.root, w, h)
            if not centered and (config.DEFAULT_POS_X is not None and config.DEFAULT_POS_Y is not None):
                 # Fallback to default pos if centering failed AND default pos is set
                 x=config.DEFAULT_POS_X; y=config.DEFAULT_POS_Y
                 try: # Ensure position is on screen
                     sw=self.root.winfo_screenwidth(); sh=self.root.winfo_screenheight(); x=max(0,min(x,sw-w)); y=max(0,min(y,sh-h));
                 except tk.TclError: pass # Ignore screen info error
                 self.root.geometry(f'{w}x{h}+{x}+{y}')
            print(f"Initial geometry: {self.root.winfo_geometry()}") # Log final geometry
        except Exception as e:
            print(f"Warn: Window geometry/pos err ({e})")
            # Absolute fallback
            try: self.root.geometry(f'{config.DEFAULT_WINDOW_WIDTH}x{config.DEFAULT_WINDOW_HEIGHT}')
            except tk.TclError: pass

    def init_camera(self):
        """Initializes the OpenCV camera capture."""
        self.set_status("Initializing camera...")
        try:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not self.cap or not self.cap.isOpened(): raise ConnectionError("Could not open camera.")
            # Read camera properties
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.camera_width <= 0 or self.camera_height <= 0: raise ConnectionError("Invalid camera dimensions.")
            self.camera_aspect_ratio = self.camera_width / self.camera_height
            print(f"Cam: {self.camera_width}x{self.camera_height} (AR:{self.camera_aspect_ratio:.2f})")
            # Create placeholder background matching camera size
            self.placeholder_bg = np.full((self.camera_height, self.camera_width, 3), config.PLACEHOLDER_BG_COLOR, dtype=np.uint8)
            # Test read one frame
            ret, _ = self.cap.read()
            if not ret: raise ConnectionError("Could not read initial frame.")
            self.set_status("Camera initialized.")
        except Exception as e:
            # Ensure camera is released if error occurs during init
            if self.cap: self.cap.release(); self.cap = None
            raise ConnectionError(f"Camera Init Failed: {e}") # Propagate error

    def set_status(self, text):
        """Update the status bar safely from any thread."""
        if self.root and self.status_var:
            try:
                # Schedule the update in the main Tkinter thread
                self.root.after(0, lambda t=text: self.status_var.set(t))
            except tk.TclError: pass # Ignore if window is destroyed

    def _load_initial_data(self):
        """Loads initial gesture data using the data manager."""
        self.data, status = load_gesture_data() # Uses defaults from data_manager->config
        self.set_status(status) # Update status bar

    def save_data_external(self):
        """ Saves data using data manager. Called by review window upon saving changes. """
        # This allows the review window to trigger a save of the potentially modified self.data
        success, self.data, status_msg = save_gesture_data(data_dict=self.data) # Pass current app data
        self.set_status(status_msg) # Update main app status
        return success, self.data, status_msg # Return results to caller (review window)

    def _update_model_status_ui(self):
        """ Updates UI elements (like buttons) based on model handler's status. """
        self.model_handler._check_files_exist() # Ensure handler knows if files are present
        is_ready = self.model_handler.is_trained_files_exist
        if self.model_btn:
            self.model_btn.config(state=tk.NORMAL if is_ready else tk.DISABLED)
        # Optionally update status bar if model files are missing
        if not is_ready and self.status_var: # Check if status_var exists
            m = "Missing: "
            if not os.path.exists(config.MODEL_FILE): m += f"{os.path.basename(config.MODEL_FILE)} "
            if not os.path.exists(config.ENCODER_FILE): m += f"{os.path.basename(config.ENCODER_FILE)}"
            current_status = self.status_var.get()
            # Avoid overwriting critical error messages if possible
            if "Error" not in current_status and "Fail" not in current_status:
                 self.set_status(m.strip() + ". Train model." if m != "Missing: " else "Train model.")

    def set_initial_sash_position(self):
        """Sets the initial position of the PanedWindow sash."""
        if not self.root or not self.main_pane: return
        try:
            # Calculate based on default width for consistent startup
            target_video_width = int(config.DEFAULT_WINDOW_WIDTH * config.INITIAL_VIDEO_PANE_RATIO)
            # Ensure minimum width for controls pane
            final_sash_pos = max(100, min(target_video_width, config.DEFAULT_WINDOW_WIDTH - 250))
            # Use 'after' to allow window geometry to stabilize before placing sash
            self.root.after(100, lambda pos=final_sash_pos: self._place_sash(pos))
        except Exception as e: print(f"Sash Calc Err: {e}")

    def _place_sash(self, position):
        """Helper to actually place the sash via root.after."""
        if not self.root or not self.main_pane: return
        try:
            # Check if panes exist (PanedWindow adds them asynchronously sometimes)
            if len(self.main_pane.panes()) >= 1:
                 self.main_pane.sash_place(0, position, 0); # print(f"Sash placed @ {position}px.")
            else: print("Error: Panes not ready for sash placement.")
        except tk.TclError: pass # Ignore if window closing
        except Exception as e: print(f"Sash Place Err: {e}")

    def _on_view_toggle_changed(self):
        """Handles logic for view option checkboxes."""
        # Ensure all relevant widgets exist before accessing them
        if self.wireframe_check and self.show_camera_feed and self.show_wireframe:
             is_cam_on = self.show_camera_feed.get()
             # Enable/disable wireframe checkbox based on camera state
             self.wireframe_check.config(state=tk.NORMAL if is_cam_on else tk.DISABLED)
             # Force wireframe ON if camera is turned OFF
             if not is_cam_on and not self.show_wireframe.get():
                 self.show_wireframe.set(True)

    def _setup_keyboard_shortcuts(self):
        """ Binds keyboard keys to application actions. """
        print("Setting up shortcuts..."); SC = self._safe_shortcut_call
        self.root.bind('<KeyPress-r>', lambda e: SC(self.record_btn, self.toggle_recording)); self.root.bind('<KeyPress-R>', lambda e: SC(self.record_btn, self.toggle_recording))
        self.root.bind('<KeyPress-s>', lambda e: SC(self.save_button_ref, self.save_samples)); self.root.bind('<KeyPress-S>', lambda e: SC(self.save_button_ref, self.save_samples))
        self.root.bind('<KeyPress-x>', lambda e: SC(self.clear_button_ref, self.clear_samples)); self.root.bind('<KeyPress-X>', lambda e: SC(self.clear_button_ref, self.clear_samples))
        self.root.bind('<Left>', lambda e: SC(self.gesture_combo, self.select_previous_gesture)); self.root.bind('<Right>', lambda e: SC(self.gesture_combo, self.select_next_gesture))
        self.root.bind('<KeyPress-t>', lambda e: SC(self.train_btn, self.start_training_thread)); self.root.bind('<KeyPress-T>', lambda e: SC(self.train_btn, self.start_training_thread))
        self.root.bind('<KeyPress-p>', lambda e: SC(self.model_btn, self.toggle_model)); self.root.bind('<KeyPress-P>', lambda e: SC(self.model_btn, self.toggle_model))
        self.root.bind('<KeyPress-d>', lambda e: self._toggle_debug_shortcut()); self.root.bind('<KeyPress-D>', lambda e: self._toggle_debug_shortcut()); print("Shortcuts bound.")

    def _safe_shortcut_call(self, widget, command_func):
        """ Calls command_func only if the associated widget is enabled. """
        if not widget: return
        try:
            # Check state safely
            current_state = str(widget.cget('state'))
            if 'disabled' not in current_state:
                command_func()
        except tk.TclError: pass # Widget might be destroyed
        except Exception as e: print(f"Shortcut Err: {e}")

    def _toggle_debug_shortcut(self):
         """ Safely toggles the debug checkbox via shortcut """
         if self.debug_check and self.debug_mode:
             self._safe_shortcut_call(self.debug_check, lambda: self.debug_mode.set(not self.debug_mode.get()))
             # Call _on_view_toggle_changed() if debug toggle affects other UI elements (it doesn't currently)
             # self._on_view_toggle_changed()

    def select_previous_gesture(self):
        """Selects the previous gesture in the list."""
        if not self.gesture_var or not config.GESTURE_OPTIONS: return
        try: current_idx = config.GESTURE_OPTIONS.index(self.gesture_var.get()); prev_idx=(current_idx-1+len(config.GESTURE_OPTIONS))%len(config.GESTURE_OPTIONS); self.gesture_var.set(config.GESTURE_OPTIONS[prev_idx])
        except ValueError: # Handle case where current value isn't in list
             if config.GESTURE_OPTIONS: self.gesture_var.set(config.GESTURE_OPTIONS[0])

    def select_next_gesture(self):
        """Selects the next gesture in the list."""
        if not self.gesture_var or not config.GESTURE_OPTIONS: return
        try: current_idx = config.GESTURE_OPTIONS.index(self.gesture_var.get()); next_idx=(current_idx+1)%len(config.GESTURE_OPTIONS); self.gesture_var.set(config.GESTURE_OPTIONS[next_idx])
        except ValueError:
             if config.GESTURE_OPTIONS: self.gesture_var.set(config.GESTURE_OPTIONS[0])

    def show_help(self):
        """Displays the help message box."""
        # (Keep help text from v2.7, including Review section and shortcuts)
        help_text = """
Hand Gesture Recognition App - Usage Guide

1. Collect Data:
   - Select gesture: Dropdown or Left/Right arrows.
   - Record: Click 'Start Recording' or press 'R'. Click/Press 'R' again to stop.
   - Save: Click 'Save Samples' or press 'S'. (Buffer cleared on save).
   - Clear Buffer: Click 'Clear Current' or press 'X' (confirms).

2. Train Model:
   - Click 'Train Model' or press 'T' (confirms). Progress bar shows.

3. Recognize Gestures:
   - Click 'Start Recognition' or press 'P' (toggles).

4. View Options:
   - Use checkboxes or 'D' for Debug Info. Camera feed/Wireframe via checkboxes only.

5. Manage Data:
   - Review: Select gesture, click 'Review Samples...'.
     - Rotate 3D view with mouse. Use toolbar for zoom/pan/save.
     - Use '< Prev' / 'Next >' buttons / Go field to navigate.
     - Click 'Delete Sample' to remove current one (confirms).
     - Click 'Save & Close' to apply deletions (needs retraining), or 'X'/'No'/'Cancel' to discard.
   - Delete All: Select gesture, click 'Delete Selected' (confirms).

6. Quit: Click 'Quit' or close window ('X').

Tips: Good lighting, plain background, only hand in frame. More data helps. Review samples!
"""
        messagebox.showinfo("How to Use", help_text)

    # --- Camera Loop ---
    def camera_loop(self):
        """Main camera processing loop."""
        last_time = time.time()
        while self.running:
            if not self.cap or not self.cap.isOpened(): time.sleep(0.5); continue # Wait if camera lost

            ret, frame = self.cap.read()
            if not ret: self.set_status("Cam Read Err."); time.sleep(0.5); continue
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = None; gesture_text = "?"; conf_text = ""; landmarks_mp = None
            try: results = self.hands.process(rgb_frame)
            except Exception: time.sleep(0.1); continue # Ignore transient MP errors

            # Feature Extraction & Prediction
            if results and results.multi_hand_landmarks:
                landmarks_mp = results.multi_hand_landmarks[0]
                features = extract_features(landmarks_mp.landmark) # Call external function

                if features is not None:
                    # Recording
                    if self.recording:
                        self.current_samples.append(features)
                        # Schedule UI update safely
                        if self.root: self.root.after(0, self.update_sample_count)

                    # Prediction
                    if self.model_active:
                        pred_gesture, confidence = self.model_handler.predict(features) # Use handler
                        if pred_gesture is not None: # Gesture recognized above threshold
                             gesture_text = f"Gesture: {pred_gesture}"
                             if self.debug_mode.get(): conf_text = f"Conf: {confidence:.2f}"
                        else: # Gesture below threshold or prediction failed
                             gesture_text = "Gesture: ?"
                             if self.debug_mode.get() and confidence > 0: conf_text = f"Conf: {confidence:.2f}" # Show low score only in debug
                             # else: conf_text remains "" if confidence was 0 or debug off

            # --- Frame Display Logic ---
            # Ensure placeholder exists
            if self.placeholder_bg is None: self.placeholder_bg = np.full((self.camera_height, self.camera_width, 3), config.PLACEHOLDER_BG_COLOR, dtype=np.uint8)
            # Choose frame
            display_frame = frame.copy() if self.show_camera_feed.get() else self.placeholder_bg.copy()
            # Draw wireframe if enabled AND landmarks were detected this frame
            if self.show_wireframe.get() and landmarks_mp:
                 self.mp_draw.draw_landmarks(display_frame, landmarks_mp, self.mp_hands.HAND_CONNECTIONS)
            # Draw text overlays
            self.draw_overlay(display_frame, gesture_text, conf_text)
            # Update UI
            try:
                 img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                 self.update_video_label_with_resize(img_rgb)
            except Exception as e:
                 if self.running and self.root: print(f"Vid Update Err: {e}"); time.sleep(0.1)

            # Limit FPS
            elapsed=time.time()-last_time; time.sleep(max(0,(1/30)-elapsed)); last_time=time.time()
        print("Camera loop stopped.")

    def draw_overlay(self, frame, gesture_text, confidence_text):
        """Draws overlays (recording indicator, prediction text) on the frame."""
        alpha=0.6; thick=2; pad=5; font=cv2.FONT_HERSHEY_SIMPLEX; scale_g=0.8; scale_c=0.7
        # Recording indicator
        if self.recording:
            cv2.circle(frame,(30,30),15,(0,0,255),-1) # Solid Red circle
            cv2.putText(frame,"REC",(55,38),font,0.7,(255,255,255),thick)
        # Recognition text
        if self.model_active and gesture_text:
            y=30; (gw,gh),_=cv2.getTextSize(gesture_text,font,scale_g,thick); (cw,ch),_=(0,0),0
            if self.debug_mode.get() and confidence_text: # Only calc size if needed
                (cw,ch),_=cv2.getTextSize(confidence_text,font,scale_c,thick)
            maxw=max(gw,cw); bh=gh+(ch+pad if self.debug_mode.get() and confidence_text else 0)+pad*2
            x1,y1=10,y-pad; x2,y2=x1+maxw+pad*2,y1+bh
            try: # Protect against drawing outside bounds
                 if x1>=0 and y1>=0 and y2<=frame.shape[0] and x2<=frame.shape[1]:
                      sub=frame[y1:y2,x1:x2]; rect=np.zeros(sub.shape,dtype=np.uint8); res=cv2.addWeighted(sub,1-alpha,rect,alpha,0)
                      frame[y1:y2,x1:x2]=res # Put semi-transparent rect back
                      # Draw text on top
                      cv2.putText(frame,gesture_text,(x1+pad,y1+gh+pad),font,scale_g,(0,255,0),thick)
                      if self.debug_mode.get() and confidence_text: cv2.putText(frame,confidence_text,(x1+pad,y1+gh+ch+pad*2),font,scale_c,(200,200,200),thick)
            except Exception as e: print(f"Warn: Overlay Err: {e}") # Log drawing errors

    def update_video_label_with_resize(self, img_rgb):
        """ Updates video label, resizing image using PIL."""
        if not self.root or not self.video_label or not self.video_label.winfo_exists(): return # Check widget exists
        try:
            w=self.video_label.winfo_width(); h=self.video_label.winfo_height()
            if w<=1 or h<=1: return # Invalid size
            img=Image.fromarray(img_rgb); img_res=ImageOps.pad(img,(w,h),color='black')
            imgtk=ImageTk.PhotoImage(image=img_res)
            # Schedule update on main thread
            def _upd():
                try:
                     if self.video_label and self.video_label.winfo_exists():
                         self.video_label.imgtk=imgtk # Keep reference!
                         self.video_label.config(image=imgtk)
                except tk.TclError: pass # Ignore if destroyed during update
            self.root.after(0,_upd)
        except tk.TclError: pass # Ignore if main widget destroyed
        except Exception as e: print(f"PhotoImage/Resize err:{e}")

    # --- Recording / Data Handling ---
    def toggle_recording(self):
        """Toggles the recording state."""
        if self.recording:
            self.recording=False; self.set_status("Recording stopped.")
            btn_txt="⏺ Start Recording (R)"
        else:
            g = self.gesture_var.get() if self.gesture_var else None
            if not g: messagebox.showwarning("Input Needed","Select gesture first.", parent=self.root); return
            self.recording=True; self.set_status(f"Recording: {g}")
            btn_txt="⏹ Stop Recording (R)"
        if self.record_btn: self.record_btn.config(text=btn_txt)

    def save_samples(self):
        """Saves currently recorded samples."""
        if not self.current_samples: messagebox.showwarning("No Data", "No samples recorded.", parent=self.root); return
        g=self.gesture_var.get();
        if not g: messagebox.showwarning("Input Needed", "Select gesture first.", parent=self.root); return
        # Validate samples before adding to main data dict
        valid_samples=[s.tolist() for s in self.current_samples if isinstance(s,np.ndarray) and s.shape==(config.FEATURE_SIZE,)]
        if not valid_samples:
            messagebox.showwarning("Error", f"No valid samples found (size {config.FEATURE_SIZE}). Buffer cleared.", parent=self.root)
            self.current_samples=[]; self.update_sample_count(); return

        cnt=len(valid_samples)
        if g not in self.data: self.data[g]=[]
        self.data[g].extend(valid_samples) # Add validated samples

        # Attempt to save the updated data dictionary
        success, self.data, status_msg = save_gesture_data(data_dict=self.data)

        if success:
            self.set_status(f"Saved {cnt} for '{g}'. Buffer cleared.")
            self.current_samples = []; self.update_sample_count(); self.update_data_list()
            # Saving new data might make the loaded model less accurate, but doesn't invalidate it yet
            # self.invalidate_model() # Optional: Force retrain after any save
        else:
            self.set_status(status_msg) # Show specific error from save function

    def clear_samples(self):
        """Clears the temporary recording buffer after confirmation."""
        if not self.current_samples: return # Nothing to clear
        num = len(self.current_samples)
        if messagebox.askyesno("Confirm Clear", f"Discard {num} recorded sample(s)?", parent=self.root):
            self.current_samples = []; self.update_sample_count(); self.set_status("Buffer cleared.")
        else: self.set_status("Clear cancelled.")

    def update_sample_count(self):
        """Updates the 'Samples: X' label."""
        if self.sample_label:
            try: self.sample_label.config(text=f"Samples: {len(self.current_samples)}")
            except tk.TclError: pass

    # --- Training ---
    def start_training_thread(self):
        """Starts the model training process in a separate thread."""
        if not self.data: messagebox.showwarning("No Data", "No training data available!", parent=self.root); return
        # Check for minimum number of classes with enough samples before confirming
        classes_with_enough_data = 0
        for g, samples in self.data.items():
             valid_count = sum(1 for s in samples if isinstance(s, list) and len(s) == config.FEATURE_SIZE)
             if valid_count >= config.MIN_SAMPLES_PER_CLASS_TRAIN:
                  classes_with_enough_data += 1
        if classes_with_enough_data < 2:
             messagebox.showwarning("Insufficient Data", f"Need at least 2 gestures with >= {config.MIN_SAMPLES_PER_CLASS_TRAIN} valid samples each to train.", parent=self.root)
             return

        if messagebox.askyesno("Confirm Train", "Training will overwrite the current model file.\nProceed?", parent=self.root):
            self._set_controls_state(tk.DISABLED); self.set_status("Starting training...")
            if self.progress_bar: self.progress_bar.config(value=0, maximum=config.N_EPOCHS); self.progress_bar.grid()
            # Define callbacks for status/progress updates from the handler
            def status_cb(msg): self.set_status(msg)
            def progress_cb(epoch, total):
                if self.root and self.progress_bar: self.root.after(0, lambda e=epoch: self._update_progress(e))
            # Run training in thread
            threading.Thread(target=self._run_training, args=(status_cb, progress_cb), daemon=True).start()

    def _update_progress(self, epoch_value):
        """Safely updates the progress bar value from any thread."""
        try:
             if self.progress_bar and self.progress_bar.winfo_exists(): self.progress_bar.config(value=epoch_value)
        except tk.TclError: pass

    def _run_training(self, status_cb, progress_cb):
        """Target function for the training thread, calls model handler."""
        # Pass the current application data to the handler
        success, message = self.model_handler.train_model(self.data, status_cb, progress_cb)
        # Schedule UI updates back on the main thread
        if self.root: self.root.after(0, lambda s=success, m=message: self._training_finished(s, m))

    def _training_finished(self, success, message):
        """Handles UI updates after training completes or fails."""
        if success:
            messagebox.showinfo("Success", message, parent=self.root)
            final_status = "Training successful." # Generic success message
            try: # Try to parse final validation accuracy for status bar
                 val_acc_str = message.split('Val:')[1].split('\n')[0].strip() # More robust parsing
                 final_status = f"Trained. Val Acc: {float(val_acc_str):.3f}"
            except: pass
            self.set_status(final_status)
        else:
            messagebox.showerror("Training Error", message, parent=self.root)
            self.set_status(message if "Fail" in message else "Training failed.")
        # Cleanup UI elements
        if self.progress_bar: self.progress_bar.grid_remove()
        self._reenable_buttons()
        self._update_model_status_ui() # Refresh button states based on new model files

    def _set_controls_state(self, state):
        """ Helper to enable/disable multiple controls """
        widgets = [self.train_btn, self.record_btn, self.model_btn, self.help_btn,
                   self.quit_btn, self.camera_check, self.wireframe_check, self.debug_check,
                   self.data_list, self.save_button_ref, self.clear_button_ref,
                   self.delete_button_ref, self.review_button_ref, # Include review button
                   self.gesture_combo, self.prev_gesture_btn, self.next_gesture_btn]
        for w in widgets:
            if w: 
                try: w.config(state=state); 
                except tk.TclError: pass

    def _reenable_buttons(self):
        """ Safely re-enables controls after long operations. """
        def upd():
            if not self.root: return
            self._set_controls_state(tk.NORMAL) # Enable common controls
            # Specifically manage model button based on whether trained files exist
            if self.model_btn: self.model_btn.config(state=tk.NORMAL if self.model_handler.is_trained_files_exist else tk.DISABLED)
            self._on_view_toggle_changed() # Ensure view toggles are correctly enabled/disabled
        if self.root: self.root.after(0,upd)

    # --- Model Interaction ---
    def toggle_model(self):
        """Toggles model recognition state using model_handler."""
        if self.model_active:
            self.model_active = False; self.set_status("Recognition stopped."); btn_txt="▶️ Start Recognition (P)"
            if self.model_btn: self.model_btn.config(text=btn_txt)
        else:
            # Attempt to load model using the handler
            success, msg = self.model_handler.load_model()
            self.set_status(msg) # Show status from loading attempt
            if not success:
                # Show error only if it wasn't just "files not found" (already covered by status)
                if "not found" not in msg and "Missing" not in msg:
                     messagebox.showerror("Load Error", msg, parent=self.root)
                self._update_model_status_ui() # Ensure button state is correct
                return # Stop if load failed

            # Check handler state after load attempt
            if not self.model_handler.is_loaded:
                 messagebox.showerror("Error","Model failed to load.", parent=self.root)
                 self._update_model_status_ui() # Update button state
                 return

            # If loaded successfully
            self.model_active = True; self.set_status("Recognition active."); btn_txt="⏹ Stop Recognition (P)"
            if self.model_btn: self.model_btn.config(text=btn_txt)

        # self._update_model_status_ui() # Might be redundant here, called within load_model path


    # --- Data Management ---
    def update_data_list(self):
        """Updates the listbox showing saved gesture data."""
        if not self.root or not self.data_list: return
        try:
            self.data_list.delete(0,tk.END)
            # Populate list sorted alphabetically
            for g in sorted(self.data.keys()):
                count = len(self.data.get(g,[])) # Safely get count
                self.data_list.insert(tk.END, f"{g}: {count} samples")
        except tk.TclError: pass # Ignore if listbox destroyed
        except Exception as e: print(f"List update err:{e}")

    def delete_selected(self):
        """Deletes all samples for the selected gesture."""
        if not self.data_list: return
        sel = self.data_list.curselection()
        if not sel: messagebox.showwarning("No Selection","Select gesture to delete.", parent=self.root); return
        try: item=self.data_list.get(sel[0]); g=item.split(':')[0]
        except tk.TclError: return # Listbox might be gone
        if g in self.data:
            if messagebox.askyesno("Confirm Delete",f"Delete all samples for '{g}'?\nThis requires retraining the model.", parent=self.root):
                del self.data[g] # Delete from app's current data
                # Save the modified data dictionary
                success, self.data, status_msg = save_gesture_data(data_dict=self.data)
                if success:
                    self.invalidate_model() # Model is no longer valid
                    self.update_data_list() # Refresh UI list
                    self.set_status(f"Deleted '{g}'. Retraining required.")
                else:
                    # If save failed, maybe add the data back? Or just notify.
                    self.set_status(f"Save failed after deleting '{g}': {status_msg}")
                    # Consider reloading data from file if save failed to ensure consistency
                    # self._load_initial_data()

    def invalidate_model(self):
        """Resets model state flags and UI, typically after data changes."""
        print("Invalidating model state...")
        self.model_handler.reset_model_state() # Clear loaded model/encoder in handler
        self.model_handler._check_files_exist() # Re-check file status
        self.model_active = False # Stop recognition if active
        if self.model_btn: self.model_btn.config(text="▶️ Start Recognition (P)") # Reset button text
        self._update_model_status_ui() # Update button enable state

    def _open_review_window(self):
        """Opens the SampleReviewWindow for the selected gesture."""
        if not self.data_list: return
        selection = self.data_list.curselection()
        if not selection: messagebox.showwarning("No Selection", "Select a gesture from the list to review.", parent=self.root); return
        try: gesture_name = self.data_list.get(selection[0]).split(':')[0]
        except: messagebox.showerror("Error", "Could not get selected gesture.", parent=self.root); return

        # Check if data exists for the selected gesture
        if gesture_name not in self.data or not self.data.get(gesture_name):
            messagebox.showinfo("No Data", f"No samples found for '{gesture_name}'.", parent=self.root); return

        # IMPORTANT: Pass a DEEP COPY of the samples to the review window
        # so modifications in the review window don't affect the main data
        # until the user explicitly saves in the review window.
        # Using list() creates a shallow copy, need deep if samples contain complex objects,
        # but for lists of lists/floats, list() or [:] is usually sufficient.
        # Let's stick with .copy() for clarity for list of lists.
        samples_to_review = [s[:] for s in self.data[gesture_name]] # Create copies of inner lists too

        print(f"Opening review window for '{gesture_name}' with {len(samples_to_review)} samples.")
        review_win = SampleReviewWindow(self, gesture_name, samples_to_review) # Pass self as parent app
        # Make the review window modal
        review_win.grab_set(); self.root.wait_window(review_win); review_win.grab_release()
        print("Review window closed.") # Happens after review window is destroyed

    # --- Quit & Run ---
    def quit_app(self):
        """Cleans up resources and closes the application."""
        if not self.running: return # Prevent double execution
        print("Quit signal received..."); self.running=False; self.set_status("Shutting down...")
        # Release Camera
        if hasattr(self,'cap') and self.cap:
            print("Releasing camera..."); cap_ref=self.cap; self.cap=None;
            try: cap_ref.release(); print("Camera released.")
            except Exception as e: print(f"Camera release error: {e}")
        # Join Thread
        if hasattr(self,'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            print("Joining camera thread..."); self.camera_thread.join(timeout=1.5) # Wait reasonable time
            print("Camera thread joined." if not self.camera_thread.is_alive() else "Warn: Camera thread join timed out.")
        # Destroy OpenCV windows (just in case)
        try: cv2.destroyAllWindows(); 
        except Exception as e: print(f"cv2 destroy error: {e}")
        # Destroy Tkinter window
        if self.root:
            print("Destroying Tkinter window..."); root_ref=self.root; self.root=None;
            try: root_ref.destroy(); print("Tkinter window destroyed.")
            except Exception as e: print(f"Tkinter destroy error: {e}")
        print("Shutdown sequence complete.")

    def run(self):
        """Starts the Tkinter main event loop."""
        if self.root and self.running:
             print("Starting Tkinter main event loop...")
             self.root.mainloop()
             print("Exited Tkinter main event loop.")
        elif not self.root: print("App run failed: UI not initialized.")
        else: print("App run failed: Not running.")