Here's a `README.md` file for your project, which includes explanations for each code segment:

---

# Eye State Detection and Automated Shutdown

This project uses a deep learning approach to detect eye states (open or closed) in real-time and automatically shut down a system if a certain threshold of closed eyes is met. The project is implemented using Python and employs various libraries including OpenCV, MediaPipe, Keras, and scikit-learn.

## Project Structure

1. **Data Collection Script**
2. **Data Augmentation and CSV Generation**
3. **Model Training**
4. **Real-time Eye State Detection and Automated Shutdown**

## 1. Data Collection Script

### `data_collection.py`

This script captures images of eyes from a webcam and saves them in two folders: `open eyes` and `close eyes`. It distinguishes between open and closed eyes based on user interaction and saves a large number of eye images for training the model.

#### Key Features:
- Uses MediaPipe for face mesh detection.
- Captures eye images from the webcam.
- Saves images of both open and closed eyes with data augmentation.

### Usage:
Run this script in a Python environment. Ensure your webcam is working and permissions are set.

## 2. Data Augmentation and CSV Generation

### `data_preprocessing.py`

This script processes the collected images, performs data augmentation (such as rotation and flipping), and saves the processed images and their labels into a CSV file (`data.csv`).

#### Key Features:
- Loads images from folders.
- Converts images to grayscale and resizes them.
- Applies augmentation techniques to increase the dataset's diversity.
- Saves the data to a CSV file for model training.

### Usage:
Run this script after collecting the images. It will create a CSV file with the image data and labels.

## 3. Model Training

### `model_training.py`

This script trains a Convolutional Neural Network (CNN) model to classify eye states as open or closed using the dataset prepared in the previous step.

#### Key Features:
- Uses Keras to define and train a CNN model.
- Evaluates the model and saves it to `model.h5`.
- Displays accuracy and confusion matrix.

### Usage:
Run this script after generating the `data.csv` file. The trained model will be saved as `model.h5`.

## 4. Real-time Eye State Detection and Automated Shutdown

### `real_time_detection.py`

This script loads the trained model and performs real-time eye state detection using the webcam. It counts the number of open and closed eye predictions and shuts down the system if the probability of closed eyes exceeds a specified threshold.

#### Key Features:
- Captures real-time video from the webcam.
- Uses the trained model to predict eye states.
- Automatically shuts down the system if closed eye probability exceeds 60%.

### Usage:
Run this script to start real-time eye state detection. Ensure your webcam is active and accessible. Adjust the threshold and time interval as needed.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Keras
- Matplotlib

Install the required packages using:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn keras matplotlib
```

## Notes

- Ensure the proper paths are set in the scripts for image folders and the CSV file.
- The system shutdown command is specific to Windows. Modify it if using a different OS.
- Ensure the webcam and necessary permissions are set up correctly for real-time detection.

---

