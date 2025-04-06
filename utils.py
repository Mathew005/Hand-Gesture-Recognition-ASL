# utils.py
import math
import tkinter as tk

def calculate_distance_3d(p1, p2):
    """Calculates Euclidean distance between two 3D points (landmark objects)."""
    try:
        # Ensure points have necessary attributes
        if not all(hasattr(p, attr) for p in [p1, p2] for attr in ['x', 'y', 'z']):
            raise AttributeError("Input points must have x, y, z attributes.")
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    except AttributeError as ae:
        print(f"Error calculating distance: {ae}")
        return 0.0
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 0.0

def center_window(window, width, height):
    """Centers a Tkinter window on the screen."""
    try:
        window.update_idletasks() # Ensure dimensions are calculated
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        pos_x = max(0, (screen_width // 2) - (width // 2))
        pos_y = max(0, (screen_height // 2) - (height // 2))
        window.geometry(f'{width}x{height}+{pos_x}+{pos_y}')
        print(f"Window centered to: {width}x{height}+{pos_x}+{pos_y}")
        return True
    except tk.TclError as e:
        # This often happens if the window is destroyed while trying to center
        if "application has been destroyed" not in str(e):
            print(f"Warning: TclError centering window ({e}). Might be closing.")
        # Fallback to just setting size if centering fails but window might exist
        try: window.geometry(f'{width}x{height}')
        except tk.TclError: pass # Ignore if window closing completely
        return False
    except Exception as e:
        print(f"Error centering window: {e}")
        # Fallback
        try: window.geometry(f'{width}x{height}')
        except tk.TclError: pass
        return False