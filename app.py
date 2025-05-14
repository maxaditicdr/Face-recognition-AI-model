import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import tempfile
from face_recognition_utils import (
    process_training_image,
    save_training_data,
    load_training_data,
    recognize_faces
)

# Set page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="👤",
    layout="wide"
)

# Initialize session state variables
if 'trained_encodings' not in st.session_state:
    st.session_state.trained_encodings = []
if 'trained_names' not in st.session_state:
    st.session_state.trained_names = []
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'data_dir' not in st.session_state:
    # Create a temporary directory for storing training data
    st.session_state.data_dir = tempfile.mkdtemp()


def main():
    st.title("Face Recognition App")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Train Model", "Recognize Faces", "About"])
    
    with tab1:
        train_model_tab()
    
    with tab2:
        recognize_faces_tab()
    
    with tab3:
        about_tab()


def train_model_tab():
    st.header("Train Face Recognition Model")
    st.write("Upload training images for different people. Make sure each image contains only one face.")
    
    # Name input for the person in the training images
    name = st.text_input("Enter the name of the person in the training images:", key="train_name")
    
    # Upload training images
    uploaded_files = st.file_uploader(
        "Upload training images (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="train_files"
    )
    
    if uploaded_files and name:
        if st.button("Process Training Images"):
            with st.spinner("Processing training images..."):
                # Process each uploaded image
                features = []
                for file in uploaded_files:
                    image = Image.open(file)
                    # Convert PIL Image to numpy array
                    image_np = np.array(image)
                    
                    # Process the image to extract face features
                    face_features = process_training_image(image_np)
                    
                    if face_features is not None:
                        features.append(face_features)
                
                if features:
                    # Add the features to the session state
                    st.session_state.trained_encodings.extend(features)
                    st.session_state.trained_names.extend([name] * len(features))
                    
                    # Save the training data
                    save_training_data(
                        st.session_state.data_dir,
                        st.session_state.trained_encodings,
                        st.session_state.trained_names
                    )
                    
                    st.session_state.training_complete = True
                    st.success(f"Successfully processed {len(features)} images for {name}.")
                else:
                    st.error("No faces detected in the uploaded images. Please try again with clear face images.")
    
    # Show the current training data
    if st.session_state.trained_names:
        st.subheader("Current Training Data")
        # Count occurrences of each name
        name_counts = {}
        for name in st.session_state.trained_names:
            if name in name_counts:
                name_counts[name] += 1
            else:
                name_counts[name] = 1
        
        # Display the counts
        for name, count in name_counts.items():
            st.write(f"{name}: {count} face{'s' if count > 1 else ''}")
        
        if st.button("Clear Training Data"):
            st.session_state.trained_encodings = []
            st.session_state.trained_names = []
            st.session_state.training_complete = False
            st.rerun()


def recognize_faces_tab():
    st.header("Recognize Faces")
    
    if not st.session_state.training_complete:
        st.warning("Please train the model first with at least one face.")
        return
    
    st.write("Upload an image to recognize faces.")
    
    # Upload an image for recognition
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="recognize_file"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.05,
        help="Lower values will match more faces but may be less accurate."
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Recognize Faces"):
            with st.spinner("Recognizing faces..."):
                # Load training data
                features, names = load_training_data(st.session_state.data_dir)
                
                if not features:
                    st.error("No training data available. Please train the model first.")
                    return
                
                # Recognize faces in the uploaded image
                result_image, face_names, confidence_scores = recognize_faces(
                    image_np, features, names, confidence_threshold
                )
                
                # Display the result
                if result_image is not None:
                    st.image(result_image, caption="Recognition Result", use_column_width=True)
                    
                    if face_names:
                        st.subheader("Recognized Faces")
                        for name, confidence in zip(face_names, confidence_scores):
                            st.write(f"{name} (Confidence: {confidence:.2f})")
                    else:
                        st.info("No faces recognized with the current confidence threshold.")
                else:
                    st.error("No faces detected in the uploaded image.")


def about_tab():
    st.header("About the Face Recognition App")
    
    st.write("""
    This application uses OpenCV-based face recognition technology to identify faces in images.
    
    ### How it works:
    1. **Train Model**: Upload images of people along with their names to train the model.
    2. **Recognize Faces**: Upload a new image, and the app will identify known faces in it.
    
    ### Technologies Used:
    - **OpenCV**: For face detection, image processing, and visualization.
    - **NumPy**: For numerical operations and feature extraction.
    - **Streamlit**: For the web interface.
    
    ### Tips for Better Results:
    - Use clear, well-lit photos for training.
    - Make sure the face is clearly visible and facing the camera.
    - For best results, use multiple training images of each person from different angles.
    - Adjust the confidence threshold to fine-tune recognition sensitivity.
    """)


if __name__ == "__main__":
    main()
