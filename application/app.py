import os
import base64

from flask import Flask, jsonify, render_template, request, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

import innvestigate

import io
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model


app = Flask(__name__)
# Limit number of requests
# limiter = Limiter(
#     app,
#     key_func=get_remote_address,
#     default_limits=["150 per day", "30 per hour"]
# )
# 1 Mb maximum file upload allowed
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

############################################################
# Constants
############################################################
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


############################################################
# Load NeuralNet model
############################################################
global graph
graph = tf.get_default_graph()

vgg = keras.applications.vgg16.VGG16()

# Set architecture
x = vgg.layers[-4].output
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(1000, activation='tanh')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation='sigmoid')(x)
NN_MODEL = Model(inputs=vgg.input, outputs=predictions)
NN_MODEL.load_weights('./weights_VGG_v7_auc_0.901.h5')


############################################################
# Helper function
############################################################
def resize_image(image):
	desired_size = max(image.size)
	old_size = image.size
	delta_w = desired_size - old_size[0]
	delta_h = desired_size - old_size[1]
	padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
	resized_image = ImageOps.expand(image, padding)
	resized_image = resized_image.resize((224, 224))
	return np.asarray(resized_image)


def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


############################################################
# Routes
############################################################
# Error too big files
@app.errorhandler(413)
def request_entity_too_large(error):
	return render_template("index.html", error_msg='File too big. Max size is 1 Mb.')
    # return 'File Too Large', 413

# Main page
@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template("index.html")

	if 'user_image' not in request.files:
		# flash('No file part')
		return render_template("index.html", error_msg='No image uploaded.')

	file = request.files['user_image']

	if file.filename == '':
		# flash('No selected file')
		return render_template("index.html", error_msg='No image uploaded.')

	if file and allowed_file(file.filename):
		# filename = secure_filename(file.filename)
		try:
			original_img_io = BytesIO(file.read())
			image = (Image.open(original_img_io))
		except:
			return render_template("index.html", error_msg='Not allowed file type. Only PNG and JPG are accepted.')
		try:
			resized_image = resize_image(image)
		except:
			return render_template("index.html", error_msg='Couldn\'t resize image.')
		####################################
		# Must update preprocessing
		####################################
		# preprocessed_image = keras.applications.mobilenet.preprocess_input(resized_image)
		preprocessed_image = keras.applications.vgg16.preprocess_input(resized_image)
		keras_image = np.expand_dims(preprocessed_image, axis=0)

		with graph.as_default():
			proba = NN_MODEL.predict(keras_image)[0][0]

		probability = '{:.2%}'.format(proba)
		
		####################################
		# Original image render
		####################################
		return_original_img_io = io.BytesIO()
		pil_original_img = Image.fromarray(np.uint8(resized_image))
		pil_original_img.save(return_original_img_io, format='png')
		original_img_base64 = base64.b64encode(return_original_img_io.getvalue())
		# End: Original image render

		####################################
		# LRP image render
		####################################
		return_lrp_img_io = io.BytesIO()

		analyzer_dt = innvestigate.create_analyzer("deep_taylor", NN_MODEL)
		with graph.as_default():
			analysis_dt = analyzer_dt.analyze(keras_image)
		a = analysis_dt.sum(axis=np.argmax(np.asarray(analysis_dt.shape) == 3))
		a /= np.max(np.abs(a))
		_ = plt.axis('off')
		_ = plt.imshow(a[0], cmap="seismic", clim=(-1, 1))

		plt.gca().set_axis_off()
		plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
		plt.margins(0, 0)
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())

		plt.savefig(fname=return_lrp_img_io, bbox_inches='tight', pad_inches=0)
		lrp_img_base64 = base64.b64encode(return_lrp_img_io.getvalue())
		# End: LRP image render

		return render_template("index.html", pr=probability,
			original_img_base64=original_img_base64.decode('ascii'), 
			lrp_img_base64=lrp_img_base64.decode('ascii'))

	
	return render_template("index.html", error_msg='Not allowed file type. Only PNG and JPG are accepted.')

@app.route("/terms", methods=['GET', 'POST'])
def terms():
	return render_template("terms.html")

if __name__ == '__main__':
	app.run(debug=True)




