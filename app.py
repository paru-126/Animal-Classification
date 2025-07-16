from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.saving import register_keras_serializable

@register_keras_serializable()
def weighted_loss(y_true, y_pred):
    class_weights = {
        0: 1.56, 1: 0.95, 2: 1.05, 3: 0.99, 4: 1.53,
        5: 1.06, 6: 1.00, 7: 1.46, 8: 1.00, 9: 1.00,
        10: 1.54, 11: 0.99, 12: 0.96, 13: 1.00, 14: 0.95
    }
    weights = tf.reduce_sum(tf.constant(list(class_weights.values()), dtype=tf.float32) * y_true, axis=1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * weights)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 
               'Panda', 'Tiger', 'Zebra']
CONFIDENCE_THRESHOLD = 0.7

# Load model with custom objects
model = load_model('animal_classifier.keras', custom_objects={'weighted_loss': weighted_loss})
model.make_predict_function()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)[0]
    max_prob = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    
    if max_prob < CONFIDENCE_THRESHOLD:
        return 'other', max_prob
    return predicted_class, max_prob

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        class_name, confidence = predict_image(filepath)
        
        return jsonify({
            'class': class_name,
            'confidence': float(confidence),
            'image_url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)