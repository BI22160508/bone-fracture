import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import os

# Get the absolute base path of the current script
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the models with correct absolute paths
model_elbow_frac = tf.keras.models.load_model(os.path.join(base_path, "weights", "ResNet50_Elbow_frac.h5"))
model_hand_frac = tf.keras.models.load_model(os.path.join(base_path, "weights", "ResNet50_Hand_frac.h5"))
model_shoulder_frac = tf.keras.models.load_model(os.path.join(base_path, "weights", "ResNet50_Shoulder_frac.h5"))
model_parts = tf.keras.models.load_model(os.path.join(base_path, "weights", "ResNet50_BodyParts.h5"))

# Label categories
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ["fractured", "normal"]

# Prediction function
def predict(img, model="Parts"):
    size = 224

    # Select the appropriate model
    if model == "Parts":
        chosen_model = model_parts
    else:
        if model == "Elbow":
            chosen_model = model_elbow_frac
        elif model == "Hand":
            chosen_model = model_hand_frac
        elif model == "Shoulder":
            chosen_model = model_shoulder_frac
        else:
            raise ValueError(f"Unknown model: {model}")

    # Preprocess image
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Predict
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # Return human-readable label
    if model == "Parts":
        return categories_parts[prediction.item()]
    else:
        return categories_fracture[prediction.item()]

