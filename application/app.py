import os
from flask import Flask, jsonify, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.models import load_model


NN_MODEL = load_model('mobile_net_v1_rocauc_0.907.h5', compile=False)

# Configure application
app = Flask(__name__)

# Ensure responses aren't cached
# @app.after_request
# def after_request(response):
#     response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     response.headers["Expires"] = 0
#     response.headers["Pragma"] = "no-cache"
#     return response

# Main page
@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/data_inference', methods=['POST'])
@app.route("/data_inference", methods=['POST'])
def data_inference():
	file = request.files['user_image']
	io_bytes = BytesIO(file.read())
	image = (Image.open(io_bytes))
	desired_size = max(image.size)
	old_size = image.size
	delta_w = desired_size - old_size[0]
	delta_h = desired_size - old_size[1]
	padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
	new_image = np.asarray(ImageOps.expand(image, padding))
	new_image = cv2.resize(new_image, (224, 224))

	print(new_image.shape)

	# proba = NN_MODEL.predict(np.expand_dims(tensorflow.keras.applications.mobilenet.preprocess_input(new_image),
 #                                                  axis=0))[0][0]

	proba = 0.9588999
	probability = '{:.2%}'.format(proba)
	return render_template("index.html", probability=probability)


