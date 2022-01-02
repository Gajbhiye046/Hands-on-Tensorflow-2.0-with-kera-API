#Stage 1: Import all project dependencies

import os
import requests
import numpy as np
import tensorflow as tf

from imageio import imwrite,imread
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify
dir_name ="C:/Users/HP/A_Complete_Guide_on_TensorFlow_2.0_using_Keras_API/Tensorflow_flask_API"
print(tf.__version__)

#Creating dataset with 5 randow picture from test dataset
# Don't run if the files exist already
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
for i in range(5):
    imwrite(os.path.join(dir_name,"uploads","{}.png".format(i)),X_test[i])

#Stage 2: Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("fashion_model_flask.h5")

#Stage 3: Creating the Flask API
#Starting the Flask application
app = Flask(__name__)

#Defining the classify_image function
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Define the uploads folder
    upload_dir = os.path.join(dir_name,"uploads","")
    #Load an uploaded image
    image = imread(upload_dir + img_name)
    
    #Define the list of class names 
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #Perform predictions with pre-trained model
    prediction = model.predict([image.reshape(1, 28*28)])

    #Return the prediction to the user
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#Start the Flask application
app.run(port=5000, debug=False)