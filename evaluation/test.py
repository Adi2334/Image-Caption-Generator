import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import torch
from tensorflow.keras.applications.xception import Xception # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from pickle import load
import warnings
from utils import data_loader as DL
import os
import pdb

# Function to extract features from an image using a CNN model
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        print(f"ERROR: Can't open image! Ensure that the image path and extension are correct. {e}")
        return None
    
    image = image.resize((299, 299))
    image = image.convert("RGB")  # Ensure it's in RGB mode
    image = np.array(image)

    # Convert 4-channel images (RGBA) to 3-channel (RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Preprocess image for model input
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0  # Normalize to range [-1,1]

    # Extract features using the model
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
        pred = np.argmax(pred)  # Get the word index with the highest probability
        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word
        
        if word == 'endtoken':
            break

    return " ".join(in_text.split()[1:-1])


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Argument parser to take image path as input
    # ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', required=True, help="Image Path")
    # args = vars(ap.parse_args())
    # img_path = args['image']
    # Dataset paths
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    test_filename = os.path.join(dataset_text, "Flickr_8k.testImages.txt")

    test_imgs = DL.load_photos(test_filename)
    img_path = os.path.join('/home/adi/img_cap_gen/Data/Flickr8k_Dataset',str(test_imgs[100]))

    # Load pre-trained models
    max_length = 35
    tokenizer = load(open("tokenizer.p", "rb"))  # Load tokenizer (sholud be generated using only train set descriptions)
    model = load_model('models_3/best_model.h5')  # Load trained image captioning model
    xception_model = Xception(include_top=False, pooling="avg")  # Load feature extractor

    # Extract features from the input image
    photo = extract_features(img_path, xception_model)
    # Ensure features were extracted successfully
    if photo is not None:
        img = Image.open(img_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        # Display the result
        print("\nGenerated Caption:")
        print(description)
        plt.imshow(img)
        plt.axis("off")  # Hide axis
        plt.show()
    else:
        print("Feature extraction failed. Exiting.")
