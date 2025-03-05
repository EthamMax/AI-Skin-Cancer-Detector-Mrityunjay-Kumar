import os
os.system("pip install huggingface-hub tf-explain opencv-python-headless")  # Install required packages

import streamlit as st
st.set_page_config(page_title="AI Skin Cancer Detector", layout="wide", initial_sidebar_state="expanded")

import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tf_explain.core.grad_cam import GradCAM

# --- Helper Function: Skin Detection ---
def is_skin_image(image, threshold=0.10):
    """
    Converts the input image to YCrCb, applies a skin-tone mask,
    and returns True if the ratio of skin-colored pixels exceeds the threshold.
    """
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_YCrCb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skinMask = cv2.inRange(image_YCrCb, lower, upper)
    skinPixels = cv2.countNonZero(skinMask)
    totalPixels = image_cv.shape[0] * image_cv.shape[1]
    return (skinPixels / totalPixels) > threshold

# --- Force a Light Theme Even if Dark Mode is On ---
st.markdown(
    """
    <style>
    :root, html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #fafafa !important;
        color: #333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Custom CSS for a Modern, Clean Design ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    body, .stApp {
      font-family: 'Poppins', sans-serif;
    }
    /* Gradient Hero Section */
    .hero {
      background: linear-gradient(135deg, #ffb385, #ff6f69);
      height: 400px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: #fff;
      animation: fadeIn 1.5s ease-in-out;
    }
    .hero-title {
      font-size: 46px;
      font-weight: 600;
      margin: 0 20px;
    }
    /* Title & Credits */
    .main-title {
      font-size: 40px;
      font-weight: 600;
      text-align: center;
      margin-top: 30px;
      color: #333;
      animation: fadeIn 1.2s ease-in-out;
    }
    .credit {
      font-size: 18px;
      text-align: center;
      color: #666;
      margin-bottom: 20px;
      animation: fadeIn 1.2s 0.5s ease-in-out;
    }
    /* Introduction */
    .introduction {
      font-size: 17px;
      text-align: center;
      margin: 0 auto 30px;
      max-width: 700px;
      color: #444;
      line-height: 1.6;
      animation: fadeIn 1.2s 0.8s ease-in-out;
    }
    /* Section Headers */
    .section-header {
      font-size: 24px;
      font-weight: 500;
      text-align: center;
      margin: 20px 0 10px;
      color: #e74c3c;
      animation: fadeInUp 1s ease-out;
    }
    /* Results Container */
    .result-section {
      margin: 20px auto;
      max-width: 600px;
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      text-align: center;
      animation: fadeInUp 1s ease-out;
    }
    .result-section strong {
      font-weight: 600;
      color: #333;
    }
    /* Note / Disclaimer */
    .note {
      font-size: 14px;
      text-align: center;
      margin-top: 20px;
      color: #888;
      max-width: 700px;
      margin-left: auto;
      margin-right: auto;
      animation: fadeIn 1.5s ease-in-out;
    }
    /* Animations */
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    /* Responsive Adjustments */
    @media (max-width: 768px) {
      .hero { height: 300px; }
      .hero-title { font-size: 32px; }
      .main-title { font-size: 28px; }
      .introduction { font-size: 15px; }
      .section-header { font-size: 20px; }
      .result-section { width: 90%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Hero Section ---
components.html(
    """
    <div class="hero">
      <div class="hero-title">AI Skin Cancer Detector</div>
    </div>
    """,
    height=400,
)

# --- Main Title & Credits ---
st.markdown('<div class="main-title">AI Skin Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="credit">Developed by Mrityunjay Kumar, Biomedical Science Student at Acharya Narendra College (University of Delhi)</div>', unsafe_allow_html=True)

# --- Introduction ---
st.markdown(
    """
    <div class="introduction">
      Hello, I'm Mrityunjay Kumar. My vision is to harness AI for early detection of skin cancer—empowering individuals with accessible tools 
      for preliminary risk assessment. This project merges cutting-edge technology with healthcare to offer a glimpse into what AI can do. 
      Please upload a clear image of a skin lesion below.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Image Upload Section ---
st.markdown('<div class="section-header">Upload an Image</div>', unsafe_allow_html=True)
image_for_prediction = None
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_for_prediction = Image.open(uploaded_file)
    st.image(image_for_prediction, caption="Uploaded Image", width=400)

# --- Analysis Section ---
if image_for_prediction:
    if not is_skin_image(image_for_prediction, threshold=0.10):
        st.error("The uploaded image does not appear to be a skin image. Please upload a clear image of a skin lesion.")
    else:
        if image_for_prediction.mode != "RGB":
            image_for_prediction = image_for_prediction.convert("RGB")
        IMG_SIZE = (224, 224)
        processed_image = image_for_prediction.resize(IMG_SIZE)
        img_array = np.array(processed_image) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        # Diagnosis mapping
        diagnosis_mapping = {
            0: ('Actinic Keratoses (akiec)', 'Malignant'),
            1: ('Basal Cell Carcinoma (bcc)', 'Malignant'),
            2: ('Benign Keratosis-like Lesions (bkl)', 'Benign'),
            3: ('Dermatofibroma (df)', 'Benign'),
            4: ('Melanoma (mel)', 'Malignant'),
            5: ('Melanocytic Nevi (nv)', 'Benign'),
            6: ('Vascular Lesions (vasc)', 'Benign')
        }

        # Load Model & Weights
        model_repo_id = "MrityuTron/SkinCancerAI-Model"
        filename = "best_model.weights.h5"
        weights_file_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        base_model.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        output_layer = layers.Dense(7, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(weights_file_path)

        # Prediction
        prediction = model.predict(img_expanded)
        predicted_class_index = np.argmax(prediction[0])
        predicted_probability = prediction[0][predicted_class_index] * 100
        predicted_diagnosis, risk_level = diagnosis_mapping[predicted_class_index]

        # Display Results
        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-section">
                <strong>Predicted Diagnosis:</strong> {predicted_diagnosis}<br>
                <strong>Risk Level:</strong> {risk_level}<br>
                <strong>Confidence:</strong> {predicted_probability:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # Grad-CAM
        st.markdown('<div class="section-header">Grad-CAM Visualization</div>', unsafe_allow_html=True)
        grad_cam_explainer = GradCAM()
        grad_cam_heatmap = grad_cam_explainer.explain(
            validation_data=(img_expanded, None),
            model=model,
            class_index=predicted_class_index,
            layer_name='out_relu'
        )
        heatmap_resized = tf.image.resize(grad_cam_heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:, :, 0]
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_image_resized = np.array(processed_image)
        overlayed_image = cv2.addWeighted(original_image_resized, 0.6, heatmap_colored, 0.4, 0)
        st.image(overlayed_image, caption=f"Grad-CAM Overlay: {predicted_diagnosis}", width=400)

# --- Disclaimer / Note ---
st.markdown(
    """
    <div class="note">
      <em>This tool is for educational purposes only and does not replace a professional diagnosis. Always consult a dermatologist for medical advice.</em>
    </div>
    """,
    unsafe_allow_html=True
)
