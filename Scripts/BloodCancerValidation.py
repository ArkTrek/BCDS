#input dataset format = Any Image Format
#input dataset splitup = 80 - 20
#image dimension = 64 * 64
#epochs = 20
#accuracy = 79.80%
#logic = Training was done for pure cancer cells only. So, when I show something that was not trained, it will show non-cancerous cell.


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('BCDS.h5')

def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

test_image_path = "Test_Image_Path"
test_image = preprocess_image(test_image_path)\
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
