import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2

from predictions import predict

# Paths
project_folder = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(project_folder, 'images')
weights_folder = os.path.join(project_folder, 'weights')

# Load models
@st.cache_resource
def load_body_part_model():
    return tf.keras.models.load_model(os.path.join(weights_folder, "ResNet50_BodyParts.h5"))

@st.cache_resource
def load_fracture_model(bone):
    model_map = {
        "Elbow": "ResNet50_Elbow_frac.h5",
        "Hand": "ResNet50_Hand_frac.h5",
        "Shoulder": "ResNet50_Shoulder_frac.h5"
    }
    return tf.keras.models.load_model(os.path.join(weights_folder, model_map[bone]))

# Grad-CAM
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    # ✅ Convert to NumPy
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image)
    overlayed = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)

# PDF generation
def generate_pdf(img, part, result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Bone Fracture Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Generated: {timestamp}")
    c.drawString(50, height - 120, f"Detected Body Part: {part}")
    c.drawString(50, height - 140, f"Prediction Result: {result}")

    temp_img = os.path.join(project_folder, "temp_img.jpg")
    img.save(temp_img)
    c.drawImage(temp_img, 50, height - 400, width=200, height=200)

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.set_page_config(page_title="Bone Fracture Detection", layout="centered")
st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to automatically detect the body part and predict fracture status.")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict body part
    body_model = load_body_part_model()
    body_classes = ["Elbow", "Hand", "Shoulder"]
    body_result = body_classes[np.argmax(body_model.predict(img_array))]

    # Predict fracture
    fracture_model = load_fracture_model(body_result)
    fracture_classes = ["Normal", "Fractured"]
    fracture_result = fracture_classes[np.argmax(fracture_model.predict(img_array))]

    st.markdown(f"### 🦴 Detected Body Part: **{body_result}**")
    if fracture_result == "Fractured":
        st.markdown(f"### 🔴 Result: **{fracture_result}**")
    else:
        st.markdown(f"### 🟢 Result: **{fracture_result}**")

    # Grad-CAM
    try:
        heatmap = get_gradcam_heatmap(fracture_model, img_array)
        heatmap_img = overlay_heatmap(heatmap, img_resized)
        st.image(heatmap_img, caption="🔥 Grad-CAM Visualization", use_container_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {str(e)}")

    # PDF Download
    report_pdf = generate_pdf(image, body_result, fracture_result)
    st.download_button(
        label="📄 Download PDF Report",
        data=report_pdf,
        file_name="fracture_report.pdf",
        mime="application/pdf"
    )

# Sidebar Info
with st.sidebar:
    st.subheader("ℹ️ Prediction Guidelines")
    rules_img_path = os.path.join(image_folder, "rules.jpeg")
    if os.path.exists(rules_img_path):
        st.image(rules_img_path, caption="Prediction Guidelines", use_container_width=True)
