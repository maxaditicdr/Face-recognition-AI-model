import cv2
import numpy as np
import os
import pickle
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_training_image(image):
    """
    Process a training image to extract face features.
    
    Args:
        image: Numpy array of the image
    
    Returns:
        face_features: Features of the face found in the image, or None if no face is found
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no face is found, return None
    if len(faces) == 0:
        return None
    
    # Get the first face
    x, y, w, h = faces[0]
    
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to a standard size
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Flatten the image for feature extraction
    face_features = face_roi.flatten()
    
    # Normalize the features
    face_features = face_features / 255.0
    
    return face_features


def save_training_data(data_dir, encodings, names):
    """
    Save face features and corresponding names to disk.
    
    Args:
        data_dir: Directory to save the data
        encodings: List of face features
        names: List of corresponding names
    """
    data = {
        "encodings": encodings,
        "names": names
    }
    
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the data
    with open(os.path.join(data_dir, "face_data.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_training_data(data_dir):
    """
    Load face features and corresponding names from disk.
    
    Args:
        data_dir: Directory where the data is saved
    
    Returns:
        encodings: List of face features
        names: List of corresponding names
    """
    data_path = os.path.join(data_dir, "face_data.pkl")
    
    if not os.path.exists(data_path):
        return [], []
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    return data["encodings"], data["names"]


def recognize_faces(image, known_face_features, known_face_names, confidence_threshold=0.6):
    """
    Recognize faces in an image based on known face features.
    
    Args:
        image: Numpy array of the image
        known_face_features: List of known face features
        known_face_names: List of corresponding names
        confidence_threshold: Threshold for face recognition confidence
    
    Returns:
        result_image: Image with bounding boxes and names
        face_names: List of recognized face names
        confidence_scores: List of confidence scores for each recognized face
    """
    # Make a copy of the image to draw on
    result_image = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no faces are found, return None
    if len(faces) == 0:
        return None, [], []
    
    face_names = []
    confidence_scores = []
    
    # Loop through each face found in the image
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Flatten the image for feature extraction
        face_features = face_roi.flatten()
        
        # Normalize the features
        face_features = face_features / 255.0
        
        # Calculate similarity to all known faces
        similarities = []
        for known_feature in known_face_features:
            # Using the Euclidean distance as a simple similarity metric
            distance = np.linalg.norm(known_feature - face_features)
            similarity = 1 / (1 + distance)  # Convert distance to similarity (0 to 1)
            similarities.append(similarity)
        
        if similarities:
            # Find the best match
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            if best_similarity >= confidence_threshold:
                name = known_face_names[best_match_index]
                confidence_scores.append(best_similarity)
            else:
                name = "Unknown"
                confidence_scores.append(0)
        else:
            name = "Unknown"
            confidence_scores.append(0)
        
        face_names.append(name)
        
        # Draw a box around the face
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(result_image, (x, y+h), (x+w, y+h+30), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(result_image, name, (x+6, y+h+25), font, 0.6, (255, 255, 255), 1)
    
    return result_image, face_names, confidence_scores
