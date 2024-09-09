from src.utils.load_model import load_model
from src.utils.predict_utils import preprocess_and_normalize, load_image

image_path = "path/to/image.jpg"

# Load the image
input_image = load_image(image_path)

# Preprocess the image
preprocessed_image = preprocess_and_normalize(input_image)

# Load the model and print the architecture
model = load_model('model.h5')
model.print_information()

# Get confidence score and class name
confidence, class_name = model.predict(preprocessed_image)

print(f"Confidence: {confidence}, Class Name: {class_name}")
