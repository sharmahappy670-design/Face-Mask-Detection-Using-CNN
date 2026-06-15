import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
model=tf.keras.models.load_model("face_mask_model.h5")
st.title("Face Mask Detection")
uploaded_file=st.file_uploader("Uploadn Image",type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image")
    img=image.resize((64,64))
    img=np.array(img)
    img=img[:,:,::-1]
    img=img.astype("float32")/255.0
    img=np.expand_dims(img,axis=0)
    prediction=model.predict(img)
    score=prediction[0][0]
    st.write("Prediction Score ",score)
    if score>0.5:
        st.success("Not Mask Detected")
    else:
        st.error("Mask Detected")
    
