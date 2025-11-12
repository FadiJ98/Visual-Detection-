import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === DATASET PATH ===
DATASET_DIR = "Resources/FaceAngles"  # using front, left, right folders
MODEL_PATH = "face_angle_classifier.h5"

# === AUGMENTATION & PREPROCESSING ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.5, 1.5],
    zoom_range=0.2,
    rotation_range=10,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=16,
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=16,
    subset='validation'
)

# === MODEL ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: front, left, right
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAINING ===
history = model.fit(train_data, validation_data=val_data, epochs=12)

# === SAVE MODEL ===
model.save(MODEL_PATH)
print(f"Model trained and saved as {MODEL_PATH}")
