import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from keras.models import load_model 

import warnings
warnings.filterwarnings("ignore")


def predict(path):
    img_width = 180
    img_height = 180
    
    data_cat = ['glioma', 'meningioma', 'pituitary']
    loaded_model = tf.keras.models.load_model(r'D:\Mayur_Project\Model\image_classifier.keras')
    
    image = path
    
    image = tf.keras.utils.load_img(image,target_size = (img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr,0)
    
    predict = loaded_model.predict(img_bat)
    
    score = tf.nn.softmax(predict)
    
    label = data_cat[np.argmax(score)]
    
    final_score = np.max(score)*100
    print(type(score))

    
    return label, final_score