# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Akhil Kasare
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Response
#from gevent.pywsgi import WSGIServer
import sys
import os

# Set default encoding to UTF-8
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='Real-Forge-Signature-Detection/forge_real_signature_model3.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(512, 512))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    # Prepare the result based on prediction
    if preds == 0:
        preds = "THE SIGNATURE IS FRAUDULENT"
    else:
        preds = "THE SIGNATURE IS ORIGINAL"
        
    # Ensure the result is a string
    return str(preds)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')

        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Return result as a response with UTF-8 encoding
        return Response(preds, content_type='text/plain; charset=utf-8')
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
