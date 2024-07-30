#First run this code if you have not downloaded


import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

def load_images_from_folder(folder, label, max_images=6000, augment=True):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images:
            break
        img_path = os.path.join(folder, filename)
        try:
            image = cv2.imread(img_path)  # Read image
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(gray_img, (64, 64))  # Resize to a fixed size (64x64)
            
            # Flatten the image and add original
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
            
            if augment:
                # Rotate and flip the image
                for angle in [90, 180, 270]:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if angle == 90 else
                                              cv2.ROTATE_180 if angle == 180 else
                                              cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_img_array = np.array(rotated_img).flatten()
                    images.append(rotated_img_array)
                    labels.append(label)
                
                # Flip horizontally and vertically
                flipped_img_h = cv2.flip(img, 1)
                flipped_img_v = cv2.flip(img, 0)
                flipped_img_h_array = np.array(flipped_img_h).flatten()
                flipped_img_v_array = np.array(flipped_img_v).flatten()
                images.append(flipped_img_h_array)
                labels.append(label)
                images.append(flipped_img_v_array)
                labels.append(label)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

# Paths to the folders containing the images
open_eye_folder = 'C:/Users/sushm/OneDrive/Documents/PROJECT/AutoShut-Deep-Learning-Based-Sleep-Detection-and-Automated-Laptop-Shutdown/data/open eyes'
closed_eye_folder = 'C:/Users/sushm/OneDrive/Documents/PROJECT/AutoShut-Deep-Learning-Based-Sleep-Detection-and-Automated-Laptop-Shutdown/data/close eyes'

# Load images with augmentation
open_eye_images, open_eye_labels = load_images_from_folder(open_eye_folder, label=1, max_images=3000, augment=True)  # Label 1 for open eyes
closed_eye_images, closed_eye_labels = load_images_from_folder(closed_eye_folder, label=0, max_images=3000, augment=True)  # Label 0 for closed eyes

# Combine the data
images = open_eye_images + closed_eye_images
labels = open_eye_labels + closed_eye_labels

# Convert to DataFrame
df = pd.DataFrame(images)
df['label'] = labels

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle and reset index

# Save shuffled DataFrame to CSV
df.to_csv("data.csv", index=False)

print("Data augmentation completed and saved to eyes_train.csv")
