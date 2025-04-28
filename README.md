# Age and Gender Detection

A deep learning project that detects a person's age and gender in real-time using computer vision and convolutional neural networks.

## Overview

This project uses deep learning to predict a person's gender and approximate age range from facial images. It consists of two main components:
- A training script that builds and trains CNN models on the UTKFace dataset
- A real-time prediction script that uses a webcam to detect faces and predict age and gender

## Features

- Face detection using Haar Cascade Classifier
- Gender classification (Male/Female)
- Age classification into 8 categories: (0-2), (3-6), (7-12), (13-20), (21-32), (33-43), (44-53), (54+)
- Real-time prediction using webcam
- Data augmentation for improved model performance
- Model training with early stopping and learning rate reduction

## Requirements

- Python 3.6+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/age-gender-detection.git
   cd age-gender-detection
   ```

2. Install the required packages:
   ```
   pip install tensorflow opencv-python numpy matplotlib scikit-learn
   ```

3. Download the [UTKFace dataset](https://susanqq.github.io/UTKFace/) and place it in the `archive/UTKFace` directory.

## Usage

### Training the Models

Run the training script to train the gender and age models:

```
python gender_age_detection.py
```

This script will:
1. Load and preprocess the UTKFace dataset
2. Train a CNN model for gender classification
3. Train a CNN model for age classification
4. Save the trained models as `gender_model.h5` and `age_model.h5`
5. Save the best models during training as `best_gender_model.h5` and `best_age_model.h5`
6. Display accuracy plots for both models

### Real-time Prediction

After training the models, run the webcam prediction script:

```
python predict_webcam.py
```

This will:
1. Start your webcam
2. Detect faces in the video stream
3. Predict gender and age for each detected face
4. Display the results in real-time

Press 'q' to quit the application.

## Model Architecture

The project uses a CNN architecture with the following layers:
- Convolutional layers with batch normalization
- MaxPooling layers
- Dropout for regularization
- Dense layers with softmax activation for classification

## Dataset

The [UTKFace dataset](https://susanqq.github.io/UTKFace/) contains over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variations in pose, facial expression, illumination, occlusion, resolution, etc.

## Future Improvements

- Implement ethnicity prediction
- Use more advanced face detection methods (MTCNN, RetinaFace)
- Deploy as a web application
- Improve model accuracy with transfer learning

## License

[MIT License](LICENSE)

## Acknowledgments

- UTKFace dataset creators
- OpenCV and TensorFlow communities
