from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Define the Keras model
def create_dr_model():
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Global pooling to reduce dimensions
        GlobalAveragePooling2D(),  # Outputs (batch_size, 128) based on 128 filters
        
        # Dense layers
        Dense(256, activation='relu'),  # Outputs 256 units to match dense_1 expectation
        Dropout(0.5),  # Prevent overfitting
        Dense(5, activation='softmax', name='dense_1')  # 5 classes for DR grading
    ])
    return model

# Load or create the model
try:
    # If you have a pre-trained model, load it here
    # dr_grading_model = tf.keras.models.load_model('path_to_model.h5')
    # For this example, we create a new model
    dr_grading_model = create_dr_model()
    # Compile the model (adjust optimizer/loss as per your training setup)
    dr_grading_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model loaded and compiled successfully")
    dr_grading_model.summary()  # Print model architecture for debugging
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Preprocess image
def preprocess_image(image_file):
    try:
        # Load image from file stream
        img = load_img(BytesIO(image_file.read()), target_size=(224, 224))
        # Convert to array
        img_array = img_to_array(img)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        logging.debug(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            logging.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image file
        image_file = request.files['image']
        
        # Preprocess image
        img = preprocess_image(image_file)
        
        # Verify input shape
        if img.shape != (1, 224, 224, 3):
            logging.error(f"Invalid image shape: {img.shape}")
            return jsonify({'error': f"Expected shape (1, 224, 224, 3), got {img.shape}"}), 400
        
        # Make prediction
        logging.debug("Making prediction...")
        predictions = dr_grading_model.predict(img)[0]
        logging.debug(f"Predictions: {predictions}")
        
        # Convert predictions to list for JSON response
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        
        return jsonify({'predictions': result})
    
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    # Test model with dummy input
    dummy_input = np.random.rand(1, 224, 224, 3)
    try:
        dummy_pred = dr_grading_model.predict(dummy_input)
        logging.info("Dummy prediction successful")
    except Exception as e:
        logging.error(f"Dummy prediction failed: {e}")
        raise
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)