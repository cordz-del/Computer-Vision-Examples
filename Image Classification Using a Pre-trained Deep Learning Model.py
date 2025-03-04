import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained MobileNetV2 model (weights trained on ImageNet)
model = MobileNetV2(weights="imagenet")

# Load an image file that contains a target object, resize it to 224x224 pixels as required by the model
img_path = "example.jpg"  # Replace with your image file path
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)

# Expand dimensions to match model input and preprocess the image
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions on the image
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]

print("Top Predictions:")
for i, pred in enumerate(decoded_predictions):
    print(f"{i+1}. {pred[1]}: {pred[2]*100:.2f}%")
