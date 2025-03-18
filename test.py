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

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        print(f"ERROR: Can't open image! Ensure that the image path and extension are correct. {e}")
        return None
    
    image = image.resize((299, 299))
    image = image.convert("RGB")  
    image = np.array(image)

    # Convert 4-channel images (RGBA) to 3-channel (RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]

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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_path = args['image']

    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    test_filename = os.path.join(dataset_text, "Flickr_8k.testImages.txt")

    # test_imgs = DL.load_photos(test_filename)
    # img_path = os.path.join('/home/adi/img_cap_gen/Data/Flickr8k_Dataset',str(test_imgs[100]))

    # max_length = 35
    train_filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    train_imgs = DL.load_photos(train_filename)
    train_descriptions = DL.load_clean_descriptions("data/descriptions.txt", train_imgs)
    tokenizer = DL.create_tokenizer(train_descriptions)
    max_length = DL.max_length(train_descriptions)

    # tokenizer = load(open("data/tokenizer.p", "rb"))  # Load tokenizer (sholud be generated using only train set descriptions)
    model = load_model('models_3/best_model.h5')  # Load trained image captioning model
    xception_model = Xception(include_top=False, pooling="avg")  # Load feature extractor

    photo = extract_features(img_path, xception_model)
    if photo is not None:
        img = Image.open(img_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        print("\nGenerated Caption:")
        print(description)
        plt.imshow(img)
        plt.axis("off") 
        plt.show()
    else:
        print("Feature extraction failed. Exiting.")
