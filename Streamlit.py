import streamlit as st
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.xception import Xception  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from pickle import load
import warnings

@st.cache_resource
def load_Model():
    model = load_model("models/best_model.h5")
    return model

# Function to extract features from an image using a CNN model
def extract_features(image, model):

    image = image.resize((299, 299))
    image = image.convert("RGB") 
    image = np.array(image)

    # Convert 4-channel images (RGBA) to 3-channel (RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Preprocess image for model input
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0  

    feature = model.predict(image)
    return feature

# Function to map a predicted word index to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'starttoken'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred) 
        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word
        
        if word == 'endtoken':
            break

    return " ".join(in_text.split()[1:-1])

st.title("Image Caption Generator")
model = load_Model()
tokenizer = load(open("tokenizer.p", "rb"))
xception_model = Xception(include_top=False, pooling="avg")
max_length = 35

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    photo = extract_features(image, xception_model)
    if photo is not None:
        description = generate_desc(model, tokenizer, photo, max_length)
    st.subheader("Generated Caption:")
    st.write(description)
