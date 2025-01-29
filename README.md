# Hand Gesture Recognition

This is a Hand Gesture Recognition application built using OpenCV, MediaPipe, TensorFlow, and Tkinter. The app allows users to collect gesture data, train a model for gesture recognition, and then use that model to identify gestures in real-time.

## Features:
- **Collect Gesture Data**: Record hand gesture samples and associate them with a gesture label (A-Z).
- **Train Model**: Train a machine learning model on the recorded gesture data to recognize gestures.
- **Recognition Mode**: Use the trained model to recognize gestures in real-time.
- **Debug Mode**: Display additional information about the recognized gesture and confidence score.
- **Save, Load, and Manage Data**: Save and load collected gesture data and trained models. View and delete saved gestures.

## Requirements

This application requires the following Python libraries:

- OpenCV
- Mediapipe
- NumPy
- TensorFlow
- scikit-learn
- Tkinter (Usually bundled with Python)
- Pickle

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
