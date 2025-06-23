import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import os

# Load models with error checking
try:
    model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
    model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
    model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
    model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")
except OSError as e:
    raise RuntimeError(f"Model loading failed: {e}")

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def predict(img_path, model="Parts"):
    try:
        size = 224
        if model == 'Parts':
            chosen_model = model_parts
        else:
            model_map = {
                'Elbow': model_elbow_frac,
                'Hand': model_hand_frac,
                'Shoulder': model_shoulder_frac
            }
            chosen_model = model_map.get(model)
            if chosen_model is None:
                raise ValueError(f"Invalid model name: {model}")

        temp_img = image.load_img(img_path, target_size=(size, size))
        x = image.img_to_array(temp_img)
        x = np.expand_dims(x, axis=0)
        prediction = np.argmax(chosen_model.predict(x), axis=1)

        return categories_parts[prediction.item()] if model == 'Parts' else categories_fracture[prediction.item()]

    except Exception as e:
        return f"Prediction error: {str(e)}"
