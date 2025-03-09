import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the saved model
model = load_model('pcos_detection_model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]

    # Display image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')

    # Display prediction
    if prediction > 0.5:
        plt.title('Predicted: Infected')
    else:
        plt.title('Predicted: Not Infected')
    plt.show()

# Test the model with a new image
img_path = './testing_sample/fake.jpg'  # Replace with your image path
predict_image(img_path)

# You can test multiple images by looping through a folder
# for img_file in os.listdir('test_images_folder'):
#     predict_image(os.path.join('test_images_folder', img_file))
