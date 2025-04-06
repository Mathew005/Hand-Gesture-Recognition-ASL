# config.py
import os

# --- File/Directory Paths ---
# Try to determine base directory relative to this config file
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ might not be defined if running in certain environments (like some IDEs directly running selections)
    # Fallback to current working directory, but this might be less reliable
    print("Warning: __file__ not defined. Using current working directory as BASE_DIR.")
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- Feature/Model Specific ---
# Using "sdn" for Scale+Distance Normalized features
DATA_FILENAME = 'gesture_data_sdn.json'
MODEL_FILENAME = 'gesture_model_sdn.h5'
ENCODER_FILENAME = 'label_encoder_sdn.pkl'

DATA_FILE = os.path.join(DATA_DIR, DATA_FILENAME)
MODEL_FILE = os.path.join(MODEL_DIR, MODEL_FILENAME)
ENCODER_FILE = os.path.join(MODEL_DIR, ENCODER_FILENAME)

# Base features: 20 landmarks * 3 coords = 60
# Added distance features: 4 adjacent finger tips + 5 fingertips to wrist = 9
FEATURE_SIZE = (20 * 3) + 9 # 69

# --- MediaPipe ---
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# --- UI / Layout ---
GESTURE_OPTIONS = [chr(i) for i in range(65, 91)] # A-Z
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 700
DEFAULT_WINDOW_WIDTH = 1000
DEFAULT_WINDOW_HEIGHT = 700
DEFAULT_POS_X = None # Center by default
DEFAULT_POS_Y = None # Center by default
INITIAL_VIDEO_PANE_RATIO = 0.65
PLACEHOLDER_BG_COLOR = (64, 64, 64)

# --- Model Training ---
N_EPOCHS = 30
DATA_IMBALANCE_WARN_RATIO = 10.0
MIN_SAMPLES_PER_CLASS_TRAIN = 10

# --- Prediction ---
PREDICTION_CONFIDENCE_THRESHOLD = 0.65

# --- Review Window ---
POINT_SIZE = 25
WRIST_COLOR_3D = "#FF4136"  # Red
POINT_COLOR_3D = "#0074D9"  # Blue
LINE_COLOR_3D = "#2ECC40"   # Green

# --- TensorFlow Logging ---
# Suppress TensorFlow INFO/WARNING messages (0=all, 1=info, 2=warning, 3=error)
TF_LOG_LEVEL = '1'
# Apply the log level setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_LOG_LEVEL

# --- Other ---
CAMERA_INDEX = 0 # Default camera index
HIGH_DPI_AWARENESS = True # Set High DPI on Windows if True
APP_VERSION = "2.7.1" # Application version