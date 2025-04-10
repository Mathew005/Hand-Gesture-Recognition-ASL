# feature_extractor.py
import numpy as np

# Import necessary items directly
import config
from utils import calculate_distance_3d

def extract_features(landmarks):
    """
    Extracts scale+distance normalized features (Size: config.FEATURE_SIZE).
    Expects the direct `landmarks.landmark` list or similar iterable of landmark objects.
    """
    if not landmarks or len(landmarks) < 21:
        # print("Warn: Insufficient landmarks for feature extraction.") # Can be noisy
        return None

    try:
        # Use landmark objects directly assuming input is landmarks.landmark list
        all_lm = landmarks
        wrist = all_lm[0]
        middle_mcp = all_lm[9] # Middle finger knuckle

        # --- 1. Calculate Scale Factor ---
        scale_distance = calculate_distance_3d(wrist, middle_mcp)
        if scale_distance < 1e-6: # Avoid division by zero or near-zero
            # print("Warn: Scale distance too small, cannot normalize.")
            return None

        # --- 2. Calculate Relative Coordinates ---
        relative_coords = []
        for i in range(1, 21):
            lm = all_lm[i]
            relative_coords.extend([
                (lm.x - wrist.x), (lm.y - wrist.y), (lm.z - wrist.z)
            ])

        # --- 3. Calculate Additional Distances ---
        finger_tip_indices = [4, 8, 12, 16, 20]
        additional_distances = []
        # Adjacent fingertip distances
        for i in range(len(finger_tip_indices) - 1):
             p1 = all_lm[finger_tip_indices[i]]
             p2 = all_lm[finger_tip_indices[i+1]]
             additional_distances.append(calculate_distance_3d(p1, p2))
        # Fingertip to wrist distances
        for i in finger_tip_indices:
             p1 = all_lm[i]
             additional_distances.append(calculate_distance_3d(wrist, p1))

        # --- 4. Normalize ALL features by Scale Factor & Combine ---
        normalized_relative_coords = [coord / scale_distance for coord in relative_coords]
        normalized_additional_distances = [dist / scale_distance for dist in additional_distances]
        features = normalized_relative_coords + normalized_additional_distances

        # --- 5. Final Check & Return ---
        if len(features) != config.FEATURE_SIZE:
            print(f"FATAL ERROR: Final feature size mismatch! Expected {config.FEATURE_SIZE}, got {len(features)}.")
            return None # Safer to return None

        return np.array(features, dtype=np.float32)

    except IndexError:
        # print("Warn: Landmark index out of bounds during feature extraction.")
        return None
    except AttributeError:
        print("Error: Input landmarks do not have expected x,y,z attributes.")
        return None
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
        return None