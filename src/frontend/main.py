import streamlit as st
import numpy as np
from PIL import Image 
from keras.models import load_model 
from streamlit_drawable_canvas import st_canvas 

# Load the trained model
model = load_model('src/model/trained_model.h5') 
st.title("Handwritten Digit Recognition")

# Canvas to write on 
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=40,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=600,
    height=600,
    drawing_mode="freedraw",
    key="canvas",
)

# Image preprocessing before inputing into model
img_rgba = canvas_result.image_data.astype("uint8")
img_gray = Image.fromarray(img_rgba, mode="RGBA").convert("L")
img_28 = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
arr_flat = np.array(img_28).reshape(1, 784)
tensor = arr_flat / 255.0
tensor = tensor.reshape(-1, 28, 28, 1)

# Perfrom the predictions
predictions = model.predict(tensor)
predicted_labels = np.argmax(predictions, axis=1)
confidence = np.max(predictions)

if st.button("Predict"):
    st.metric(label ="Predicted digit", value =str(predicted_labels[0]))
    st.metric(label = "Confidence", value = f"{confidence*100:.2f}%")


