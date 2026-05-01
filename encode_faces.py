import face_recognition
import cv2
import os
import pickle

# --- Configuration Paths ---
FACES_DIR = "data\\registered_faces"
ENCODINGS_FILE = "data\\encodings.pkl"

def encode_known_faces():
    """
    Scans the faces directory, computes 128-d encodings using the HOG model,
    and serializes the data to a pickle file for instant loading later.
    """
    # Auto-create the directory if it doesn't exist
    os.makedirs(FACES_DIR, exist_ok=True)
    
    known_encodings = []
    known_names = []

    print("[INFO] Booting up Face Encoder...")
    
    # Check if the directory is empty
    if not os.listdir(FACES_DIR):
        print(f"[ERROR] The directory '{FACES_DIR}' is empty. Please add some .jpg images first.")
        return

    # Loop over the image files
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # The filename becomes the ID (e.g., "Bhavya.jpg" -> "Bhavya")
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(FACES_DIR, filename)

            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Could not read {filename}. Skipping.")
                continue

            # face_recognition requires RGB, but OpenCV loads in BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"[INFO] Processing: {name}...")

            # Detect face locations using the CPU-friendly 'hog' model
            boxes = face_recognition.face_locations(rgb_image, model="hog")

            # Compute the facial embedding (128-d vector)
            encodings = face_recognition.face_encodings(rgb_image, boxes)

            # Validate that exactly one face was found
            if len(encodings) == 1:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"  [+] Success: {name} encoded.")
            elif len(encodings) > 1:
                print(f"  [!] Warning: Multiple faces found in {filename}. Skipping to ensure clean data.")
            else:
                print(f"  [-] Failed: No face found in {filename}. Try a clearer photo.")

    # Serialize and save to disk
    print(f"\n[INFO] Saving encodings to {ENCODINGS_FILE}...")
    data = {"encodings": known_encodings, "names": known_names}

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[SUCCESS] Database updated! {len(known_names)} faces securely encoded.")

if __name__ == "__main__":
    encode_known_faces()