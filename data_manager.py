# data_manager.py
import os
import json
import traceback
import tkinter.messagebox as messagebox # Keep UI feedback here for now

import config # Use config for paths

def load_gesture_data(filepath=config.DATA_FILE, expected_feature_size=config.FEATURE_SIZE):
    """Loads gesture data from a JSON file, validating feature size."""
    print(f"Attempting to load data from {filepath}...")
    loaded_data = {}
    total_loaded = 0
    skipped_samples = 0
    status_msg = "" # Initialize status message
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                raw_data = json.load(f)

            for gesture, samples in raw_data.items():
                if isinstance(samples, list):
                    valid_samples = [s for s in samples if isinstance(s, list) and len(s) == expected_feature_size]
                    if valid_samples:
                        loaded_data[gesture] = valid_samples
                        total_loaded += len(valid_samples)
                    skipped_count = len(samples) - len(valid_samples)
                    if skipped_count > 0:
                        print(f"Warn: Skipped {skipped_count} samples for '{gesture}' (size != {expected_feature_size}).")
                        skipped_samples += skipped_count
                else:
                    print(f"Warn: Invalid data format for gesture '{gesture}'. Skipping.")
            status_msg = f"Loaded {total_loaded} samples."
            if skipped_samples > 0: status_msg += f" Skipped {skipped_samples} (size mismatch)."
            print(status_msg)
        else:
            status_msg = f"Data file '{os.path.basename(filepath)}' not found."
            print(status_msg)

    except json.JSONDecodeError:
        status_msg = f"Decode Error: File '{os.path.basename(filepath)}' might be corrupted."
        print(status_msg)
        messagebox.showerror("Load Error", status_msg) # Provide user feedback
        return {}, status_msg # Return empty dict and error status
    except Exception as e:
        status_msg = f"Load Error: {e}"
        print(status_msg); traceback.print_exc()
        messagebox.showerror("Load Error", status_msg)
        return {}, status_msg # Return empty dict and error status

    return loaded_data, status_msg # Return loaded data and success/info status

def save_gesture_data(filepath=config.DATA_FILE, data_dict={}, expected_feature_size=config.FEATURE_SIZE):
    """Saves gesture data to a JSON file, validating feature size."""
    print(f"Attempting to save data to {filepath}...")
    verified_data = {}
    total_saved = 0
    status_msg = ""
    success = False
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        for gesture, samples in data_dict.items():
            if isinstance(samples, list): # Check if samples is a list
                valid_samples = [s for s in samples if isinstance(s, list) and len(s) == expected_feature_size]
                if valid_samples:
                    verified_data[gesture] = valid_samples
                    total_saved += len(valid_samples)
                elif gesture in data_dict and samples: # Warn only if data existed but is now invalid/empty
                    print(f"Warn: No valid samples (size {expected_feature_size}) for '{gesture}'. Not saving this gesture.")
            else:
                print(f"Warn: Invalid sample structure for gesture '{gesture}'. Skipping.")


        with open(filepath, 'w') as f:
            json.dump(verified_data, f, indent=2)

        status_msg = f"Data saved successfully ({total_saved} samples)."
        print(status_msg)
        success = True

    except TypeError as e: # Often happens if data contains non-serializable types (like numpy arrays)
         status_msg = f"Save Error: Data type not serializable - {e}. Ensure samples are lists of floats."
         print(f"ERROR: {status_msg}"); traceback.print_exc()
         messagebox.showerror("Save Error", status_msg)
    except Exception as e:
        status_msg = f"Save Error: {e}"
        print(f"ERROR: {status_msg}"); traceback.print_exc()
        messagebox.showerror("Save Error", status_msg)

    return success, verified_data, status_msg # Return success flag, verified data, and status message