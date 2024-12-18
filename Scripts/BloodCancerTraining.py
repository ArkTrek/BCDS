## I used 80 - 20 split of cancer blood images. For one class. 
##This gave me accurate results accordingly. 
##So whenever I give a normal blood sample as input, it provides an accurate output. 
##Also I have fine tuned the output validation on the other script as well.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility
tf.random.set_seed(42)

# Set dataset path
dataset_path = 'Blood_Cancer_Dataset_path'

# Image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32

# Function to load and resize TIFF images
def load_and_resize_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    image = Image.open(image_path)  # Open the TIFF image
    image = image.convert('RGB')  # Convert to RGB if it's in a different format
    image = image.resize(target_size)  # Resize to target size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Custom Data Generator for TIFF Images
class TiffImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        images = np.array([load_and_resize_image(image_path, self.target_size) for image_path in batch_image_paths])
        return images, np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Function to get image paths and labels from the dataset
def get_image_paths_and_labels(dataset_path):
    image_paths = []
    labels = []
    class_names = os.listdir(dataset_path)  # Assumes each class is in a separate folder
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_folder):
            if image_name.endswith('.tiff') or image_name.endswith('.tif'):  # Check for TIFF files
                image_paths.append(os.path.join(class_folder, image_name))
                labels.append(class_to_label[class_name])
    
    return image_paths, labels

# Load dataset
image_paths, labels = get_image_paths_and_labels(dataset_path)

# Split the dataset into training and validation sets (80-20)
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Create data generators
train_generator = TiffImageDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))
val_generator = TiffImageDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Save the model
model.save('BCDS.h5')

# Plot training and validation metrics
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
