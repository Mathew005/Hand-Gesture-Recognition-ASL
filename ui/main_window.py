# ui/main_window.py
import tkinter as tk
from tkinter import ttk, PanedWindow
import config # Use config for gesture options, defaults etc.

def create_main_ui(root, app_instance):
    """Creates the main UI widgets and layout within the root window."""

    root.grid_rowconfigure(0, weight=1); root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0) # Progress bar row
    root.grid_rowconfigure(2, weight=0) # Status bar row

    # --- Main Paned Window ---
    app_instance.main_pane = PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, background="#f0f0f0")
    app_instance.main_pane.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5,0)) # Reduced bottom pady

    # --- Left Pane (Video) ---
    video_outer_frame = ttk.Frame(app_instance.main_pane, style='Video.TFrame')
    video_outer_frame.grid_rowconfigure(0, weight=1); video_outer_frame.grid_columnconfigure(0, weight=1)
    style = ttk.Style(root); style.configure('Video.TFrame', background='black')
    app_instance.video_label = ttk.Label(video_outer_frame, anchor=tk.CENTER, background="black")
    app_instance.video_label.grid(row=0, column=0, sticky="nsew"); app_instance.main_pane.add(video_outer_frame)

    # --- Right Pane (Controls) ---
    controls_scrollable_frame = ttk.Frame(app_instance.main_pane)
    controls_scrollable_frame.grid_rowconfigure(0, weight=1); controls_scrollable_frame.grid_columnconfigure(0, weight=1)
    app_instance.main_pane.add(controls_scrollable_frame)
    controls_frame = ttk.Frame(controls_scrollable_frame, padding="10"); controls_frame.grid(row=0, column=0, sticky="nsew")
    controls_frame.grid_columnconfigure(0, weight=1); controls_frame.grid_rowconfigure(3, weight=1); controls_frame.grid_rowconfigure(4, weight=0) # Data management expands

    # --- Control Sections ---

    # 1. Data Collection
    collect_frame = ttk.LabelFrame(controls_frame, text="Data Collection", padding="10")
    collect_frame.grid(row=0, column=0, sticky="new", pady=(0, 10)); collect_frame.grid_columnconfigure(0, weight=0); collect_frame.grid_columnconfigure(1, weight=0); collect_frame.grid_columnconfigure(2, weight=1); collect_frame.grid_columnconfigure(3, weight=0)
    ttk.Label(collect_frame, text="Gesture:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
    app_instance.prev_gesture_btn = ttk.Button(collect_frame, text="<", width=2, command=app_instance.select_previous_gesture); app_instance.prev_gesture_btn.grid(row=0, column=1, sticky=tk.E, padx=(0, 2))
    app_instance.gesture_var = tk.StringVar()
    app_instance.gesture_combo = ttk.Combobox(collect_frame, textvariable=app_instance.gesture_var, values=config.GESTURE_OPTIONS, state="readonly", width=8); app_instance.gesture_combo.grid(row=0, column=2, sticky="ew", pady=2)
    if config.GESTURE_OPTIONS: app_instance.gesture_combo.current(0)
    app_instance.next_gesture_btn = ttk.Button(collect_frame, text=">", width=2, command=app_instance.select_next_gesture); app_instance.next_gesture_btn.grid(row=0, column=3, sticky=tk.W, padx=(2, 0))
    app_instance.sample_label = ttk.Label(collect_frame, text="Samples: 0"); app_instance.sample_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5,2))
    app_instance.record_btn = ttk.Button(collect_frame, text="⏺ Start Recording (R)", command=app_instance.toggle_recording); app_instance.record_btn.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(5, 5))
    btn_frame1 = ttk.Frame(collect_frame); btn_frame1.grid(row=3, column=0, columnspan=4, sticky="ew"); btn_frame1.grid_columnconfigure(0, weight=1); btn_frame1.grid_columnconfigure(1, weight=1)
    app_instance.save_button_ref = ttk.Button(btn_frame1, text="💾 Save Samples (S)", width=13, command=app_instance.save_samples); app_instance.save_button_ref.grid(row=0, column=0, sticky='ew', padx=(0,2), pady=2)
    app_instance.clear_button_ref = ttk.Button(btn_frame1, text="🗑 Clear Current (X)", width=13, command=app_instance.clear_samples); app_instance.clear_button_ref.grid(row=0, column=1, sticky='ew', padx=(2,0), pady=2)

    # 2. Model
    model_frame = ttk.LabelFrame(controls_frame, text="Model", padding="10"); model_frame.grid(row=1, column=0, sticky="new", pady=5); model_frame.grid_columnconfigure(0, weight=1)
    app_instance.train_btn = ttk.Button(model_frame, text="🧠 Train Model (T)", command=app_instance.start_training_thread); app_instance.train_btn.grid(row=0, column=0, sticky="ew", pady=3)
    app_instance.model_btn = ttk.Button(model_frame, text="▶️ Start Recognition (P)", command=app_instance.toggle_model); app_instance.model_btn.grid(row=1, column=0, sticky="ew", pady=3); app_instance.model_btn.config(state=tk.DISABLED)

    # Initialize Tkinter Variables (owned by app_instance)
    app_instance.show_camera_feed = tk.BooleanVar(value=True)
    app_instance.show_wireframe = tk.BooleanVar(value=True)
    app_instance.debug_mode = tk.BooleanVar(value=False)

    # 3. View Options
    view_frame = ttk.LabelFrame(controls_frame, text="View Options", padding="10"); view_frame.grid(row=2, column=0, sticky="new", pady=5); view_frame.grid_columnconfigure(0, weight=1)
    app_instance.camera_check = ttk.Checkbutton(view_frame, text="Show Camera Feed", variable=app_instance.show_camera_feed, command=app_instance._on_view_toggle_changed); app_instance.camera_check.grid(row=0, column=0, sticky="w", pady=2)
    app_instance.wireframe_check = ttk.Checkbutton(view_frame, text="Show Wireframe", variable=app_instance.show_wireframe, command=app_instance._on_view_toggle_changed); app_instance.wireframe_check.grid(row=1, column=0, sticky="w", pady=2)
    app_instance.debug_check = ttk.Checkbutton(view_frame, text="Show Debug Info (D)", variable=app_instance.debug_mode, command=app_instance._on_view_toggle_changed); app_instance.debug_check.grid(row=2, column=0, sticky="w", pady=2)

    # 4. Data Management
    data_frame = ttk.LabelFrame(controls_frame, text="Saved Data", padding="10"); data_frame.grid(row=3, column=0, sticky="nsew", pady=5); data_frame.grid_rowconfigure(0, weight=1); data_frame.grid_columnconfigure(0, weight=1)
    list_frame = ttk.Frame(data_frame); list_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5)); list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
    app_instance.data_list = tk.Listbox(list_frame, height=6); app_instance.data_list.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=app_instance.data_list.yview); scrollbar.grid(row=0, column=1, sticky="ns"); app_instance.data_list['yscrollcommand'] = scrollbar.set
    data_buttons_frame = ttk.Frame(data_frame); data_buttons_frame.grid(row=1, column=0, sticky="ew"); data_buttons_frame.grid_columnconfigure(0, weight=1); data_buttons_frame.grid_columnconfigure(1, weight=1)
    app_instance.review_button_ref = ttk.Button(data_buttons_frame, text="✏️ Review Samples...", command=app_instance._open_review_window)
    app_instance.review_button_ref.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(5,0))
    app_instance.delete_button_ref = ttk.Button(data_buttons_frame, text="❌ Delete Selected", command=app_instance.delete_selected)
    app_instance.delete_button_ref.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(5,0))

    # 5. Application Controls
    app_ctrl_frame = ttk.Frame(controls_frame, padding=(0, 5, 0, 0)); app_ctrl_frame.grid(row=4, column=0, sticky="sew", pady=(10, 0)); app_ctrl_frame.grid_columnconfigure(0, weight=1); app_ctrl_frame.grid_columnconfigure(1, weight=1)
    app_instance.help_btn = ttk.Button(app_ctrl_frame, text="❓ Help", width=10, command=app_instance.show_help)
    app_instance.help_btn.grid(row=0, column=0, sticky="sw", padx=(0, 5))
    app_instance.quit_btn = ttk.Button(app_ctrl_frame, text="Quit", width=10, command=app_instance.quit_app)
    app_instance.quit_btn.grid(row=0, column=1, sticky="se", padx=(5, 0))

    # --- Progress Bar & Status Bar (Widgets created here, controlled by app instance) ---
    app_instance.progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
    app_instance.progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 2)); app_instance.progress_bar.grid_remove() # Start hidden
    app_instance.status_var = tk.StringVar(); app_instance.status_var.set("Initializing...") # Status bar variable owned by app instance
    status_bar = ttk.Label(root, textvariable=app_instance.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5,2))
    status_bar.grid(row=2, column=0, sticky="ew")

    # Set initial state for wireframe toggle based on camera default
    app_instance._on_view_toggle_changed()