import os
import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from utils import load_class_names, output_layer_names, postprocess, draw_detections, letterbox

st.set_page_config(
    page_title="FastYolo - Edge Object Detection",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    cfg = ROOT / "models" / "yolov4-tiny.cfg"
    weights = ROOT / "models" / "yolov4-tiny.weights"
    if not cfg.exists() or not weights.exists():
        return None
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

@st.cache_data
def load_classes():
    names_path = ROOT / "models" / "coco.names"
    if names_path.exists():
        return load_class_names(str(names_path))
    return []

st.title("⚡ FastYolo: Real-Time Edge Object Detection")
st.write("Interactive web interface for YOLOv4-Tiny optimized for low-power edge hardware simulation.")

net = load_model()
class_names = load_classes()

if net is None or not class_names:
    st.error("Model weights or configuration files not found! Please run `python models/download_weights.py` first.")
else:
    st.sidebar.header("Inference Settings")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    nms_thresh = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)
    input_size = st.sidebar.selectbox("Input Resolution", [416, 320, 256], index=0)

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_pil, use_container_width=True)

        with col2:
            st.subheader("FastYolo Detection Output")
            
            if st.button("Run Object Detection", type="primary"):
                with st.spinner("Processing frame..."):
                    h, w = frame_bgr.shape[:2]
                    padded, scale, pad_w, pad_h = letterbox(frame_bgr, input_size)
                    blob = cv2.dnn.blobFromImage(
                        padded, 1/255.0, (input_size, input_size), [0, 0, 0], swapRB=True, crop=False
                    )
                    net.setInput(blob)
                    
                    out_names = output_layer_names(net)
                    start_time = cv2.getTickCount()
                    outputs = net.forward(out_names)
                    end_time = cv2.getTickCount()
                    
                    latency = (end_time - start_time) / cv2.getTickFrequency() * 1000.0
                    fps = 1000.0 / latency if latency > 0 else 0

                    detections = postprocess(outputs, input_size, scale, pad_w, pad_h, conf_thresh, nms_thresh)
                    result_bgr = draw_detections(frame_bgr, detections, class_names)
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

                    st.image(result_rgb, use_container_width=True)
                    st.success(f"Detected **{len(detections)}** object(s) in **{latency:.1f} ms** (~{fps:.1f} FPS)")
            else:
                st.info("Click the button above to run inference.")