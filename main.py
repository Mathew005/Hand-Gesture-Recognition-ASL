# main.py
import tkinter as tk
from tkinter import messagebox
import traceback
import config # For high DPI setting and APP_VERSION
from app import HandGestureRecognizer # Import the main application class

if __name__ == "__main__":
    print(f"Launching Hand Gesture Recognizer v{config.APP_VERSION}...")

    # Optional: Set high DPI awareness early (Windows specific)
    if config.HIGH_DPI_AWARENESS:
        try:
            from ctypes import windll
            # SetProcessDpiAwareness(1) for system aware, (2) for per-monitor aware
            # System aware (1) is usually sufficient and safer
            windll.shcore.SetProcessDpiAwareness(1)
            print("High DPI awareness set (System Aware).")
        except ImportError:
             print("Info: `ctypes` not available, cannot set High DPI (non-Windows?).")
        except AttributeError:
             print("Info: `SetProcessDpiAwareness` not available (Older Windows?).")
        except Exception as e:
             print(f"Warn: Failed to set High DPI awareness: {e}")

    app = None
    init_success = False # Flag to track if init constructor finishes
    try:
        # Create the main application instance
        app = HandGestureRecognizer()
        # If constructor finishes without throwing, mark init as potentially successful
        # (Internal errors might still set app.running=False)
        init_success = True

    except Exception as e:
         # Catch critical errors DURING app = HandGestureRecognizer() call
         print(f"\n--- FATAL ERROR DURING __init__ ---")
         traceback.print_exc()
         try: messagebox.showerror("Fatal Error", f"Critical error during initialization:\n{e}\n\nPlease check console for details.")
         except: pass # Ignore if messagebox fails (e.g., if Tk failed)
         if app: # Attempt cleanup if app object was partially created
             app.running = False # Ensure running flag is false
             app.quit_app()

    # --- Run the application ---
    try:
        # Start the Tkinter main loop only if fully initialized and running flag is still True
        if init_success and app and app.root and app.running:
             print("Initialization check passed. Calling app.run()...")
             app.run() # This contains root.mainloop()
        # Provide informative messages if app didn't start
        elif not init_success: print("Application did not start because __init__ failed critically.")
        elif not app: print("Application did not start because app object is None.")
        elif not app.root: print("Application did not start because root window is None.")
        elif not app.running: print("Application did not start because app.running is False (likely due to init error or early quit signal).")
        else: print("Application did not start for an unknown reason.")

    except Exception as e:
         # Catch unexpected errors during app.run() (e.g., within mainloop callbacks)
         print(f"\n--- FATAL APPLICATION ERROR (post-init) ---")
         traceback.print_exc()
         try: messagebox.showerror("Fatal Error", f"Critical error during execution:\n{e}\n\nPlease check console for details.")
         except: pass
         if app: app.quit_app() # Attempt cleanup
    finally:
         print("Application exiting.")