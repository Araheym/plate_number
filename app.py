# COMMON LIBRARIES
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tempfile
import time
# 'git+https://github.com/facebookresearch/detectron2.git'


# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2.engine import DefaultPredictor

##################### LOADING MODEL AND ITS CONFIGURATIONS #################################
cfg_save_path = 'OD.cfg.pickle'

# LOAD THE CONFIGURATION
with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
# ADJUST THE PATH TO THE SAVED MODEL
cfg.MODEL.WEIGHTS = os.path.join('Model_V2', 'model_final.pth')
# SET THE THRESHOLD FOR THE PREDICTION
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# CREATING PREDICTOR
predictor = DefaultPredictor(cfg)

# FUNCTION TO PROCESS AND EXTRACT PLATE NUMBER
def process_image(image):
    im = np.array(image.convert('RGB')) # Convert PIL image to NumPy array
    output = predictor(im)
    
    # VISUALIZES PREDICTIONS 
    v = Visualizer(im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    
    # DISPLAY VISUALIZED IMAGE
    plt.figure(figsize=(10, 6))
    plt.imshow(v.get_image()[:, :, ::-1])  # Use RGB format
    plt.axis(False)
    st.pyplot(plt)

# FUNCTION TO PROCESS VIDEO
def process_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Make predictions on the frame
        predictions = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        # Convert BGR to RGB
        output_frame_rgb = output.get_image()[:, :, ::-1]
        # Display each frame in Streamlit
        st.image(output_frame_rgb, caption="Processed Frame", channels="RGB", use_column_width=True)
        # Simulate video playback by limiting the frame display speed
        time.sleep(1 / frame_rate)

    cap.release()
    
    
    
def process_camera(predictor):
    cap = cv2.VideoCapture(0)  # Open system camera
    # frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make predictions on the frame
        predictions = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        # Convert BGR to RGB
        # output_frame_rgb = output.get_image()[:, :, ::-1]
        frames.append(output.get_image())
        
        for frame in frames:
            st.image(frame,channels="RGB",caption="Processed Frame",use_column_width=True)
            # Display each frame in Streamlit
            # st.image(output_frame_rgb, caption="Processed Frame", channels="RGB", use_column_width=True)
        
        # Simulate video playback by limiting the frame display speed
        # time.sleep(1 / frame_rate)
    cap.release()


# Streamlit interface
st.title("License Plate Detection App")

# Allow users to choose between image and video
upload_choice = st.selectbox("What would you like to upload?", ["Image", "Video","Camera"])

if upload_choice == "Image":
    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Predict and display results
        if st.button("Make Prediction"):
            st.write("Processing image...")
            process_image(image)

elif upload_choice == "Video":
    # Video upload
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        # Display the uploaded video
        st.video(tfile.name)

        # Process and predict on the uploaded video
        if st.button("Process Video"):
            st.write("Processing video...")
            process_video(tfile.name, predictor)
            
elif upload_choice == "Camera":
    # Use the system's camera
    if st.button("Start Camera"):
        st.write("Accessing camera... Press 'q' to quit.")
        process_camera(predictor)
