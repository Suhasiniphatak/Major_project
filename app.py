from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load models
retina_detector_model = load_model('models/retina_detector_model.h5')  # Retina vs. Non-retina
dr_grading_model = load_model('models\dr_detection_train_dr_1.h5')        # DR grading

# Class labels
dr_classes = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative'
}

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    confidence = {}
    uploaded_image = None  # <-- track uploaded image name

    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    uploaded_image = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image)
    file.save(filepath)

    # Step 1: Retina Detection
    img = preprocess_image(filepath)
    retina_pred = retina_detector_model.predict(img)[0][0]

    if retina_pred < 0.8:
        return render_template('index.html',
                               prediction="Not a Retinal Image!! Please upload retinal image",
                               confidence=None,
                               uploaded_image=uploaded_image)

    # Step 2: DR Grading
    dr_preds = dr_grading_model.predict(img)[0]
    dr_class = np.argmax(dr_preds)
    prediction = f"{dr_classes[dr_class]}"

    for i, cls in dr_classes.items():
        confidence[cls] = round(dr_preds[i] * 100, 2)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           uploaded_image=uploaded_image)


if __name__ == '__main__':
    app.run(debug=True)
