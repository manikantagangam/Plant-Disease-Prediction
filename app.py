## Done by : Manikanta Gangam

from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template

# Define a flask app
app = Flask(__name__)

# Model path
MODEL_PATH ='Tomato_Disease_Detection.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img, model):
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize pixel values

    # Make prediction
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    if preds==0:
        preds="Bacterial_spot"
    elif preds==1:
        preds="Early_blight"
    elif preds==2:
        preds="Late_blight"
    elif preds==3:
        preds="Leaf_Mold"
    elif preds==4:
        preds="Septoria_leaf_spot"
    elif preds==5:
        preds="Spider_mites Two-spotted_spider_mite"
    elif preds==6:
        preds="Target_Spot"
    elif preds==7:
        preds="Tomato_Yellow_Leaf_Curl_Virus"
    elif preds==8:
        preds="Tomato_mosaic_virus"
    else:
        preds="Healthy"
        
    
    
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if file is present
        if file:
            # Read the file directly without saving
            img = image.load_img(file, target_size=(224, 224))
            
            # Make prediction
            prediction = model_predict(img, model)
            return render_template('result.html', prediction=prediction)
    
    # Render the upload form
    return render_template('index.html')

from PIL import Image
from io import BytesIO

from PIL import Image
from io import BytesIO

def get_class_name(class_index):
    class_names = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
        "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
        "Healthy"
    ]
    return class_names[class_index]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        # Read the file data
        file_data = file.read()
        
        # Load the image using PIL
        img = Image.open(BytesIO(file_data))
        img = img.resize((224, 224))  # Resize the image if needed
        
        # Convert the image to a numpy array and preprocess it if required
        img_array = np.array(img) / 255.0  # Example: Normalize pixel values
        
        # Make prediction using your model
        preds = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class_index = np.argmax(preds)
        
        # Get the class name corresponding to the predicted class index
        predicted_class_name = get_class_name(predicted_class_index)
        
        # Return the predicted class name as a response
        return predicted_class_name





if __name__ == '__main__':
    app.run(port=5001, debug=True)


