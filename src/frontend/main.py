import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model 
from streamlit_drawable_canvas import st_canvas 

# Load the trained model
model = load_model('src/model/trained_model.h5') 

def preprocess_rgba_to_28x28x1(rgba: np.ndarray):
    rgba_u8 = rgba.astype("uint8")
    img_gray = Image.fromarray(rgba_u8, mode="RGBA").convert("L")
    arr = np.array(img_gray)
    if arr.mean() > 127:
        img_gray = ImageOps.invert(img_gray)
    img_28 = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    arr_28 = np.array(img_28).astype("float32") / 255.0
    arr_28 = arr_28.reshape(1, 28, 28, 1)
    return arr_28, img_28

def top_k_probs(probs: np.ndarray, k: int = 3):
    probs = probs.flatten()
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]

st.title("Handwritten Digit Recognition")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=600,
    height=600,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict") and canvas_result.image_data is not None:
    x_tensor, img_28 = preprocess_rgba_to_28x28x1(canvas_result.image_data)
    probs = model.predict(x_tensor, verbose=0)
    top3 = top_k_probs(probs, k=3)
    best_digit, best_prob = top3[0]
    st.metric(label="Predicted digit", value=str(best_digit), delta=f"{best_prob*100:.1f}% confidence")
    st.write("Top-3 probabilities:")
    st.dataframe({"digit": [d for d, _ in top3], "probability": [p for _, p in top3]}, use_container_width=True)
    st.write("Model input (28×28) preview:")
    st.image(img_28.resize((140, 140), Image.Resampling.NEAREST))