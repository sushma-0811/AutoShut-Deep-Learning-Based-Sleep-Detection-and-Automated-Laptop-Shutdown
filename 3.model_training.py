import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Conv2D, Dense, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data.csv")

# Separate the features and labels
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# Ensure the data type is float for reshaping
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Reshape the flattened images to (64, 64, 1)
X_train = X_train.reshape(len(X_train), 64, 64, 1)
X_test = X_test.reshape(len(X_test), 64, 64, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)  # Should print (number_of_images, 64, 64, 1)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=28, validation_split=0.1)
model.evaluate(X_test,y_test)

model.save('model.h5')

# Evaluate and predict
y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(np.int32).flatten()  # Convert probabilities to binary predictions

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
cmd.plot()

plt.show()
