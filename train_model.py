import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 64
NUM_CLASSES = 5  # Adjust as necessary

def load_data(image_folder):
    X = []
    y = []
    for class_label in range(NUM_CLASSES):
        class_folder = os.path.join(image_folder, f'class{class_label}')
        for image_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, image_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            X.append(img)
            y.append(class_label)
    
    X = np.array(X, dtype='float32') / 255.0
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    return X, y

def train_model():
    image_folder = 'traffic_sign_dataset'
    X, y = load_data(image_folder)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save('traffic_sign_model.h5')  # Ensure this path is correct

if __name__ == '__main__':
    train_model()
