# Import library
import cv2
import glob
import numpy as np
from PIL import Image
import streamlit as st

from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

# Initialize models
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('./models/yolo_classification_model.keras')

# Function to perform pose classification
def pose_classification(img):
    # Read uploaded image
    image = Image.open(img)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform keypoint detection
    results = detection_keypoint(image_cv)
    results_keypoint = detection_keypoint.get_xy_keypoint(results)

    # Perform keypoint classification
    input_classification = results_keypoint[10:]
    results_classification = classification_keypoint(input_classification)

    # Visualize results
    image_draw = results.plot(boxes=False)
    x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
    image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)),(int(x_max), int(y_max)), (0,0,255), 2)
    (w, h), _ = cv2.getTextSize(results_classification.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)), (0,0,255), -1)
    image_draw = cv2.putText(image_draw, f'{results_classification.upper()}', (int(x_min), int(y_min)-4),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    
    return image_draw, results_classification

# Set page configuration
st.set_page_config(layout="wide", page_title="YoloV8 Keypoint Classification")

# Main content
st.write("## YoloV8 Keypoint Yoga Pose Classification")
st.write(":dog: Upload an image to classify yoga poses like Downdog, Goddess, Plank, Tree, Warrior2 :grin:")
st.sidebar.write("## Upload Image :gear:")

# Sidebar
img_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Main content layout
col1, col2 = st.columns([1, 2])

if img_upload is not None:
    # Perform pose classification
    image, classification = pose_classification(img_upload)

    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(image)

    # Display keypoint result and classification
    with col2:
        st.subheader("Keypoint Result")
        st.image(image)
        st.write(f"Pose Classification: **{classification}**")
