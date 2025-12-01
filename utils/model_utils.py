import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Optional EfficientNet preprocess
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    def preprocess_input(x):
        return x.astype("float32") / 255.0

MODEL_PATH = "pashu_model.h5"

# Breed labels
BREEDS = [
'Alambadi','Amritmahal','Ayrshire','Banni','Bargur','Bhadawari','Brown_Swiss',
'Dangi','Deoni','Gir','Guernsey','Hallikar','Hariana','Holstein_Friesian',
'Jaffrabadi','Jersey','Kangayam','Kankrej','Kasargod','Kenkatha','Kherigarh',
'Khillari','Krishna_Valley','Malnad_gidda','Mehsana','Murrah','Nagori','Nagpuri',
'Nili_Ravi','Nimari','Ongole','Pulikulam','Rathi','Red_Sindhi','Sahiwal',
'Tharparkar','Toda','Umblachery','Vechur'
]

@st.cache_resource
def load_cnn_model():
    # Decompress if needed
    if not os.path.exists(MODEL_PATH) and os.path.exists(MODEL_PATH + ".gz"):
        import gzip
        import shutil
        try:
            with gzip.open(MODEL_PATH + ".gz", 'rb') as f_in:
                with open(MODEL_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            st.error(f"Error decompressing model: {e}")
            return None

    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def preprocess_pil(img: Image.Image, size=224):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_breed(model, img: Image.Image):
    x = preprocess_pil(img)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return {"label": BREEDS[idx], "confidence": float(probs[idx])}

@st.cache_data
def get_cached_prediction(_model, input_array):
    """
    Wrapper for model.predict that caches the result.
    The model argument is prefixed with '_' to prevent Streamlit from hashing it.
    The input_array (numpy array) is hashed automatically by Streamlit.
    """
    preds = _model.predict(input_array)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return {"label": BREEDS[idx], "confidence": float(probs[idx])}
