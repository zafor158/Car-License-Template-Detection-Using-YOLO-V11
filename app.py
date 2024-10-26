import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load your trained YOLO model
# model_path = "best_model.pt"
model = YOLO(best_model.pt)
st.title("License Plate Detection App")

# Sidebar for options
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose Input Method:", ("Upload Image", "Upload Video", "Take a Picture"))

# Define a function to load and preprocess images
def load_image(img):
    return np.array(Image.open(img))

# License plate detection function with support for multiple detections and confidence adjustment
def detect_license_plate(image, confidence_threshold=0.1):
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Run the model on the image with a specified confidence threshold
    results = model(image, conf=confidence_threshold)  # Set confidence threshold for detection

    # Create a copy of the image to annotate
    annotated_image = image.copy()

    # Draw all bounding boxes on the image
    boxes = results[0].boxes  # Extract all bounding boxes from results
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            if float(box.conf) >= confidence_threshold:  # Only display boxes above threshold
                # Draw each box on the annotated image
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates for the box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                confidence = float(box.conf)  # Confidence level for the box
                cv2.putText(annotated_image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_image, boxes  # Returns the annotated image and all boxes

# Streamlit app workflow
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect license plates
        st.write("Detecting license plates...")
        annotated_image, boxes = detect_license_plate(image)

        # Display detected image with all bounding boxes
        st.image(annotated_image, caption="Detection Result with All Plates", use_column_width=True)

        # Display details of each detected license plate if any are found
        if boxes is not None and len(boxes) > 0:
            st.write("Detected License Plates:")
            for idx, box in enumerate(boxes):
                confidence = float(box.conf)  # Convert tensor to float
                st.write(f"License Plate {idx + 1}: Confidence = {confidence:.2f}")
        else:
            st.write("No license plates detected.")

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.write("Detecting license plates in video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Run detection on each frame
            annotated_frame, boxes = detect_license_plate(frame)

            # Display results
            stframe.image(annotated_frame, channels="RGB")
        cap.release()

elif option == "Take a Picture":
    picture = st.camera_input("Take a Picture")
    if picture:
        image = load_image(picture)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Detect license plates
        st.write("Detecting license plates...")
        annotated_image, boxes = detect_license_plate(image)

        # Display detected image with bounding boxes
        st.image(annotated_image, caption="Detection Result with All Plates", use_column_width=True)

        # Display details of each detected license plate if any are found
        if boxes is not None and len(boxes) > 0:
            st.write("Detected License Plates:")
            for idx, box in enumerate(boxes):
                confidence = float(box.conf)  # Convert tensor to float
                st.write(f"License Plate {idx + 1}: Confidence = {confidence:.2f}")
        else:
            st.write("No license plates detected.")

st.sidebar.write("Made with ❤️ using YOLO and Streamlit")
