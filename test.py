import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model.h5')

def preprocess_image(img_path):
    # Load the image
    img = Image.open(img_path)
    # Resize the image to (28, 28)
    img = img.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Invert colors
    img_array = np.invert(img_array)
    # Flatten the image
    img_flat = img_array.ravel()
    # Reshape to match the expected input shape of the model
    img_flat = img_flat.reshape(1, -1)
    return img_flat

img_path = "sample2.png"
img = preprocess_image(img_path)
print(img.shape) 

def predict_with_model(model, img):
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    print("Prediction for test image:", predicted_label)

predict_with_model(model, img)