from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import cv2
# dimensions of our images
img_width, img_height = 224, 224
# load the model we saved
model = load_model('dDiabetic_Retinopathy_Detection.h5')
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img=cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        img = img /255
# predicting images
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        preds= model.predict_classes(images, batch_size=80)
        if(preds[0]==0):
            result = 'Pas de rétinopathie diabétique'
        elif(preds[0]==1):
            result = 'RD Minime'
        elif(preds[0]==2):
            result='RD Modérée'
        elif(preds[0]==3):
            result='RD Sévère'
        else:
            result='RD Proliferative '   
        return result

    return None
    


if __name__ == '__main__':
    app.run(debug=True)

