import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATASET_DIR = "C:/Users/sunni/OneDrive/Desktop/age-gender-project-code/Age_Gender/archive/UTKFace"
IMG_SIZE = 64
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ages, genders, images = [], [], []

print("[INFO] Loading and preprocessing images...")
for img_name in os.listdir(DATASET_DIR):
    try:
        age, gender = map(int, img_name.split("_")[:2])
        path = os.path.join(DATASET_DIR, img_name)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            images.append(face)
            ages.append(age)
            genders.append(gender)
            break
    except:
        continue

images = np.array(images, dtype=np.float32) / 255.0
genders = to_categorical(genders, 2)

# Age bucketing
def age_bucket(age):
    if age <= 2: return 0
    elif age <= 6: return 1
    elif age <= 12: return 2
    elif age <= 20: return 3
    elif age <= 32: return 4
    elif age <= 43: return 5
    elif age <= 53: return 6
    else: return 7

age_classes = to_categorical([age_bucket(age) for age in ages], 8)

# Train/test split
X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    images, genders, age_classes, test_size=0.2, random_state=42)

# Data Augmentation
aug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Model Architecture
def build_model(output_classes):
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(output_classes, activation='softmax')
    ])
    return model

# Callbacks
def get_callbacks(name):
    return [
        EarlyStopping(patience=6, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5),
        ModelCheckpoint(f"best_{name}_model.h5", save_best_only=True)
    ]

# Train Gender Model
print("[INFO] Training gender model...")
gender_model = build_model(2)
gender_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

gender_history = gender_model.fit(
    aug.flow(X_train, y_gender_train, batch_size=64),
    validation_data=(X_test, y_gender_test),
    epochs=25,
    callbacks=get_callbacks("gender")
)

# Train Age Model
print("[INFO] Training age model...")
age_model = build_model(8)
age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

age_history = age_model.fit(
    aug.flow(X_train, y_age_train, batch_size=64),
    validation_data=(X_test, y_age_test),
    epochs=25,
    callbacks=get_callbacks("age")
)

# Save final models
gender_model.save("gender_model.h5")
age_model.save("age_model.h5")

# Plotting function
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

plot_history(gender_history, "Gender")
plot_history(age_history, "Age")
