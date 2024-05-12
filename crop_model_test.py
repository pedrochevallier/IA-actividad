from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('main_model.keras')
train_path='crop_prediction'

# Load and preprocess your image
img_path = '/Users/pchevallier/Downloads/stripeRust-on-wheat.jpeg'
img = image.load_img(img_path, target_size=(100, 100))  # Specify the target size expected by your model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension


# Make prediction
predictions = model.predict(img_array)
print(predictions)
# Interpret predictions
predicted_class = np.argmax(predictions)
class_names = ['corn', 'rice', 'wheat']# Replace with your actual class names


predicted_label = class_names[predicted_class]

print("Predicted label:", predicted_label)