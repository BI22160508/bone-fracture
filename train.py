import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------------------
# Step 1: Load Paths & Labels from Directory Structure
# ------------------------------------------------------------------------------
def load_path(base_path, body_part_target):
    """
    Traverse the dataset directory and return a list of dicts with:
        - body_part: which body part folder it came from
        - patient_id: ID folder (if applicable)
        - label: 'fractured' or 'normal'
        - image_path: full path to the image file
    The function only picks the subfolder with a name matching body_part_target.
    """
    dataset = []
    # Loop over patients or case directories
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # Folder may contain different body parts (e.g., Elbow, Hand, Shoulder)
            for body in os.listdir(folder_path):
                if body == body_part_target:
                    body_path = os.path.join(folder_path, body)
                    # Each body folder may have multiple patient IDs
                    for patient_id in os.listdir(body_path):
                        patient_path = os.path.join(body_path, patient_id)
                        # Each patient folder may contain subfolders with labels in their names (e.g., "patient0001_positive" or "patient0001_negative")
                        for label_folder in os.listdir(patient_path):
                            # Determine label based on folder name (you may customize this logic)
                            if label_folder.split('_')[-1].lower() == 'positive':
                                label = 'fractured'
                            elif label_folder.split('_')[-1].lower() == 'negative':
                                label = 'normal'
                            else:
                                continue  # skip folders not following naming convention
                            label_path = os.path.join(patient_path, label_folder)
                            # Each label folder should contain images
                            for img_name in os.listdir(label_path):
                                img_path = os.path.join(label_path, img_name)
                                dataset.append({
                                    'body_part': body,
                                    'patient_id': patient_id,
                                    'label': label,
                                    'image_path': img_path
                                })
    return dataset

# ------------------------------------------------------------------------------
# Step 2: Define Function for Training a Specific Body Part
# ------------------------------------------------------------------------------
def trainPart(part):
    """
    Trains a model for a given body part.
    Loads the data, splits it, creates data generators, builds a
    ResNet50-based model, trains it with early stopping, evaluates, and saves
    the model and plots.
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(THIS_FOLDER, 'Dataset')
    
    # Load images and labels for the specific body part (e.g., 'Elbow', 'Hand', 'Shoulder')
    data = load_path(dataset_dir, part)
    
    # Create a DataFrame from the collected paths and labels
    filepaths = [d['image_path'] for d in data]
    labels = [d['label'] for d in data]
    df = pd.DataFrame({'Filepath': filepaths, 'Label': labels})
    
    # Split: 90% for training (which will be further split into training/validation) and 10% for testing
    train_df, test_df = train_test_split(df, train_size=0.9, shuffle=True, random_state=1, stratify=df['Label'])
    
    # Define image data generators for training/validation and testing.
    # Apply horizontal flip augmentation and ResNet50 preprocessing.
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.2  # 20% of train_df will be used for validation
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    # Create generators from the dataframe
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',  # two classes -> categorical encoding
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    # ------------------------------------------------------------------------------
    # Step 3: Build the Model using ResNet50 as a Base (Feature Extractor)
    # ------------------------------------------------------------------------------
    pretrained_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,      # exclude final classification layer
        weights='imagenet',
        pooling='avg'           # global average pooling for feature reduction
    )
    
    # Freeze the ResNet50 base to speed up training and prevent overfitting on small datasets.
    pretrained_model.trainable = False
    
    # Add custom dense layers on top
    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # two classes: fractured and normal
    
    model = tf.keras.Model(inputs, outputs)
    
    print("-------Training " + part + "-------")
    model.summary()
    
    # ------------------------------------------------------------------------------
    # Step 4: Compile and Train the Model
    # ------------------------------------------------------------------------------
    # Use a low learning rate with Adam optimizer for fine adjustments.
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Implement early stopping to avoid overfitting; restore best weights.
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25,
        callbacks=[callbacks]
    )
    
    # ------------------------------------------------------------------------------
    # Step 5: Save the Model and Evaluate on the Test Set
    # ------------------------------------------------------------------------------
    weights_dir = os.path.join(THIS_FOLDER, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    model_filename = os.path.join(weights_dir, "ResNet50_" + part + "_frac.h5")
    model.save(model_filename)
    
    results = model.evaluate(test_generator, verbose=0)
    print(f"{part} Test Results:", results)
    print(f"Test Accuracy for {part}: {np.round(results[1]*100, 2)}%")
    
    # ------------------------------------------------------------------------------
    # Step 6: Create and Save Accuracy & Loss Plots
    # ------------------------------------------------------------------------------
    plots_dir = os.path.join(THIS_FOLDER, "plots", "FractureDetection", part)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy - ' + part)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    acc_plot_file = os.path.join(plots_dir, "_Accuracy.jpeg")
    plt.savefig(acc_plot_file)
    plt.close()
    
    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss - ' + part)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    loss_plot_file = os.path.join(plots_dir, "_Loss.jpeg")
    plt.savefig(loss_plot_file)
    plt.close()

# ------------------------------------------------------------------------------
# Step 7: Train Models for Each Body Part
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # List of body parts to train a model on (adjust as needed)
    parts = ["Elbow", "Hand", "Shoulder"]
    for part in parts:
        trainPart(part)


