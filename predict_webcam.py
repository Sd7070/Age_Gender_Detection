import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained models
gender_model = load_model("C:/Users/sunni/OneDrive/Desktop/age-gender-project-code/Age_Gender/best_gender_model.h5")
age_model = load_model("C:/Users/sunni/OneDrive/Desktop/age-gender-project-code/Age_Gender/best_age_model.h5")

# Labels
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(21-32)', '(33-43)', '(44-53)', '(54+)']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 64

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            face_input = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_input = face_input.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            # Predict gender
            gender_pred = gender_model.predict(face_input)[0]
            gender_label = gender_list[np.argmax(gender_pred)]

            # Predict age
            age_pred = age_model.predict(face_input)[0]
            age_label = age_list[np.argmax(age_pred)]

            label = f"{gender_label}, {age_label}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print("Error processing face:", e)

    cv2.imshow("Real-Time Gender & Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
