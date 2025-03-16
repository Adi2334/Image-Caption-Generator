import numpy as np
import torch
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from pickle import load
import warnings
from utils import data_loader as DL
import os
import pdb
import pickle

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

def generate_desc_batch(model, tokenizer, photos, max_length):
    batch_size = len(photos)
    in_texts = ['starttoken'] * batch_size  # Start token for each image

    for _ in range(max_length):
        sequences = tokenizer.texts_to_sequences(in_texts)  # Convert all captions to sequences
        sequences = pad_sequences(sequences, maxlen=max_length)  # Pad all sequences

        # Predict the next word in batch
        preds = model.predict([photos, sequences], verbose=0)
        # pdb.set_trace()
        preds = np.argmax(preds, axis=1)  # Get highest probability index for each image

        new_words = [word_for_id(pred, tokenizer) for pred in preds]  # Convert indices to words

        # Update captions for each image
        for i in range(batch_size):
            if new_words[i] is None or new_words[i] == 'endtoken':
                continue  # Stop adding words if endtoken is reached
            in_texts[i] += ' ' + new_words[i]

    # Remove 'starttoken' and 'endtoken' from all captions
    return [" ".join(text.split()[1:-1]) for text in in_texts]



if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    features_file = "features_aug.p"

    # Load test data
    test_filename = os.path.join(dataset_text, "Flickr_8k.testImages.txt")
    test_imgs = DL.load_photos(test_filename)
    test_features = DL.load_features(test_imgs,features_file)   # dict

    # Load pre-trained models
    max_length = 35
    tokenizer = load(open("tokenizer.p", "rb"))  # Load tokenizer (sholud be generated using only train set descriptions)
    model = load_model('models_3/best_model.h5')  # Load trained image captioning model

    N = len(test_imgs)
    print('Number of test images = ',N)

    '''Single image caption generation'''
    # test_cap = {}
    # i=1
    # for key, features in test_features.items():
    #     print(i)
    #     feature = np.expand_dims(features[0],axis=0)
    #     caption = generate_desc(model,tokenizer,feature,max_length)
    #     test_cap[key] = caption
    #     i = i+1

    test_cap = {}
    keys = list(test_features.keys())
    batch_size = 10

    for i in range(0, N, batch_size):
        batch_keys = keys[i:i + batch_size]  # Get batch keys
        batch_features = np.array([test_features[key][0] for key in batch_keys])  # Prepare batch
        # pdb.set_trace()
        captions = generate_desc_batch(model, tokenizer, batch_features, max_length)  # Generate captions in batch

        for key, caption in zip(batch_keys, captions):
            test_cap[key] = caption

        print(f'Processed batch {i//batch_size + 1}/{(N // batch_size) + 1}')
    
    
    with open("test_captions.p", "wb") as file:
        pickle.dump(test_cap, file)

    print('Task completed')
        


