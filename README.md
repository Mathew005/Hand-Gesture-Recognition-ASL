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
```

# Hand Gesture Recognition

## How to Use

### 1. **Setting Up the Application**
- Install the required dependencies.
- Run the script to launch the GUI interface.

### 2. **Recording Gestures**
- Select a gesture from the dropdown menu (A-Z).
- Click the "Start Recording" button to begin recording hand gesture samples.
- Perform the gesture in front of the webcam.
- Click "Stop Recording" to stop recording.
- Click "Save Samples" to save the current samples associated with the selected gesture.

### 3. **Training the Model**
- After collecting enough samples for each gesture, click the "Train Model" button to train the model on the collected data.
- The app will create a model and save it as `gesture_model.h5` along with the label encoder as `label_encoder.pkl`.
- Training will output the accuracy of the model and save it for future use.

### 4. **Gesture Recognition**
- Once the model is trained, click "Start Recognition" to start the real-time gesture recognition mode.
- Perform gestures in front of the webcam, and the app will display the predicted gesture and confidence score.
- Click "Stop Recognition" to stop recognition.

### 5. **Debug Mode**
- Click "Enable Debug" to toggle debug mode, which will show additional information about the prediction such as the confidence value.

### 6. **Managing Data**
- The "Delete Selected" button allows you to delete saved gesture data from the list.
- The "Clear Samples" button clears the current gesture samples without saving them.
- The "Quit" button closes the application.

## Troubleshooting
- **Camera not working**: Ensure that your webcam is working and properly connected.
- **Model training failed**: Ensure you have sufficient data collected for each gesture (at least a few dozen samples per gesture).
- **Performance issues**: The app uses real-time video processing, which can be CPU-intensive. Close any unnecessary applications to improve performance.

## Saving & Loading Data
- The gesture data is saved in a `gesture_data.json` file.
- The trained model and label encoder are saved as `gesture_model.h5` and `label_encoder.pkl`, respectively.
- You can load and use the saved model by clicking the "Start Recognition" button.


## GUI

![Screenshot 2025-01-29 193950](https://github.com/user-attachments/assets/88747ad5-23c6-41b5-a7c3-ed389732f1e2)

## Note
The more varied the samples are, the better the results will be.


