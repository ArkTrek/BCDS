import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('BCDS.h5')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(64, 64)):
    # Load the image using PIL
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert to RGB if not already
    img = img.resize(target_size)  # Resize to the input size of the model
    img = np.array(img) / 255.0  # Normalize the pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

test_image_path = "Test_Image_Path"  # Replace with the actual JPEG image path

# Preprocess the image
test_image = preprocess_image(test_image_path)

# Make a prediction using the model
prediction = model.predict(test_image)

img = Image.open(test_image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

if prediction[0] > 0.8015:
    print("Cancer Cell Detected!")
else:
    print("This Cell is considered Normal.")
    
print(f"Confidence score: {prediction[0][0]:.4f}")
