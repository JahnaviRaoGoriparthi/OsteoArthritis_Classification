import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your Keras model
model = tf.keras.models.load_model('model_keras.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    img = cv2.resize(img, (224, 224))  # Change size to match model input
    img = img.astype(np.float32) / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            processed_image = preprocess_image(file_path)
            if processed_image is None:
                return "Error in image preprocessing"
            
            prediction = model.predict(np.expand_dims(processed_image, axis=0))
            print(f"Prediction: {prediction}")
            if prediction[0]>=0.5:
                predicted_class=1
            else:
                predicted_class=0
          # Change this line if your model outputs different format
            print(f"Predicted Class: {predicted_class}")
            
            return render_template('index.html', filename=filename, prediction=predicted_class)
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
