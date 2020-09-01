# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:31:38 2020

@author: kosaraju vivek
"""



import numpy as np
import streamlit as st

# Keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K
from werkzeug.utils import secure_filename
import h5py
import os
import io
from PIL import Image, ImageOps


def models():
    model=load_model('model')
    model.summary()
    return model
    


    
def model_predict(img_path, model):
    # Preprocessing the image
    image = Image.open(img_path).convert('RGB')
    size = (224,224)
    image = ImageOps.fit(image, size)
    image = img_to_array(image)
    image= np.expand_dims(image, axis = 0)
    image = np.array(image) / 255.0
    outs=['COVID +ve','COVID -ve']
    preds=model.predict(image)
    preds=preds.argmax(axis=1)
    preds=outs[preds[0]]
    return preds


 
def main():
    st.set_option('deprecation.showfileUploaderEncoding',False)
    st.title("COVID-19 detection")
    html1 ="""
    <div style="padding:5px">
    <h5 style="color:blue;text-align:right;font-weight:bold;font-style:arial;">created by &copy;Vivek Kosaraju</h5>
    </div>
    """
    st.markdown(html1,unsafe_allow_html=True)
    html2="""
    <h3 style="color:red;text-align:center;">Please upload your chest X-ray Image &#128071;</h3>
    """
    st.markdown(html2,unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file:
        st.image(image_file,caption="uploaded image",width=10,use_column_width=True)
    if st.button("Predict"):
        if image_file is None:
            raise Exception("image not uploaded, please refresh page and upload the image")
        with st.spinner("Predicting......"):
            model = models()
            result=model_predict(image_file,model)
            st.success('The output is {}'.format(result))
         
    hide_streamlit_style ="""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    html_temp3="""
    <p>This application can able to detect COVID-19 from chest X-ray.This application was developed by &copy; Kosaraju Vivek<br>
        you can connect with me :<a href="https://www.linkedin.com/in/vivek-kosaraju/">Let's connect</a>
    </p>
    """
    if st.button("About"):
        st.markdown(html_temp3,unsafe_allow_html=True)
    
if __name__=='__main__':
    main()
    





    
