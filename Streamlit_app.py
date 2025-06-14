import streamlit as st
import numpy as np
import tempfile

import wx
import os
import cv2
from PIL import Image
from io import StringIO
import pandas as pd
import Predict

#####################################

st.set_page_config(page_title='Brain Tumor App', page_icon="📊", initial_sidebar_state="expanded", layout='wide')
from streamlit.components.v1 import html

with st.container():
    
    html("""
    <script>
        // Locate elements
        var decoration = window.parent.document.querySelectorAll('[data-testid="stDecoration"]')[0];
        var sidebar = window.parent.document.querySelectorAll('[data-testid="stSidebar"]')[0];
        // Observe sidebar size
        function outputsize() {
            decoration.style.left = `${sidebar.offsetWidth}px`;
        }
        new ResizeObserver(outputsize).observe(sidebar);
        // Adjust sizes
        outputsize();
        decoration.style.height = "5.0rem";
        decoration.style.right = "0px";
        // Adjust text decorations
        decoration.innerText = "Brain Tumor Detection"; // Replace with your desired text
        decoration.style.fontSize = '40px'
        decoration.style.fontWeight = "bold";
        decoration.style.display = "flex";
        decoration.style.justifyContent = "center";
        decoration.style.alignItems = "center";
        
    </script>
""", width=0, height=0)

#####################################


col1, col2, col3 = st.columns([1,20,1])

with col2:
    uploaded_file = st.file_uploader("Choose an image ", type = "jpg")

if uploaded_file is not None:

    #st.write(uploaded_file._file_urls.upload_url)
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
    #path = uploaded_file._file_urls.upload_url + "\/" + uploaded_file.name
   

    label, score = Predict.predict(path)

    
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype = np.uint8)
    opencv_image = cv2.imdecode(file_bytes,1)
    
    col1, col2, col3 = st.columns([1,10,10])

    with col2:
        st.image(opencv_image,width=400, use_container_width=40,channels="BGR")
    
    with col3:
        label = f"<p style='font-size:50px;'>Tumor Type : {label}</p>"
        confidence = f"<p style='font-size:50px;'>confidence : {score}</p>"
        st.markdown(label, unsafe_allow_html=True)
        st.markdown(confidence, unsafe_allow_html=True)
    
    




