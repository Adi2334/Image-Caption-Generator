import os
from pickle import load
import torch
import pickle  # Missing import for saving the tokenizer
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pdb

# Load the text file contents
def load_doc(filename):
    with open(filename, "r") as file:
        text = file.read()
    return text

# Load list of image names from a file
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]  # Fixing the incorrect split
    return photos

# Load and clean descriptions
def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}

    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue
        
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            # start and end tokens are important because they tell the model how to initialize and when to stop
            # desc = '<start> ' + " ".join(image_caption) + ' <end>' 
            # tokenizer will ignore '<' and '>' making a problem. so use another start and end token also it will remove any punctuations
            desc = 'starttoken ' + " ".join(image_caption) + ' endtoken' 
            descriptions[image].append(desc)

    return descriptions

# Load extracted image features
def load_features(photos, features_file):
    all_features = load(open(features_file, "rb"))
    features = {k: all_features[k] for k in photos if k in all_features}
    return features

# Convert dictionary to a clear list of descriptions
def dict_to_list(descriptions):
    all_desc = []  # Initialize an empty list
    for key in descriptions.keys():
        all_desc.extend(descriptions[key])  # Append all descriptions for each key
    return all_desc

# Create a tokenizer from descriptions
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)  # Convert dictionary to list of sentences
    tokenizer = Tokenizer()  # Initialize tokenizer
    # tokenizer = Tokenizer(filters='_')
    tokenizer.fit_on_texts(desc_list)  # Fit tokenizer on text corpus
    return tokenizer

# Calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)  # Convert dictionary to list
    return max(len(d.split()) for d in desc_list)  # Compute max sentence length


if __name__ == "__main__":
    # Define dataset paths
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    features_file = "features_aug.p"
    train_filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    val_filename = os.path.join(dataset_text, "Flickr_8k.devImages.txt")  # Validation images file
    test_filename = os.path.join(dataset_text, "Flickr_8k.testImages.txt")

    train_imgs = load_photos(train_filename)
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    train_features = load_features(train_imgs,features_file)
    pdb.set_trace()

    val_imgs = load_photos(val_filename)
    val_descriptions = load_clean_descriptions("descriptions.txt", val_imgs)
    val_features = load_features(val_imgs,features_file)

    test_imgs = load_photos(test_filename)
    test_descriptions = load_clean_descriptions("descriptions.txt", test_imgs)
    test_features = load_features(test_imgs,features_file)

    print("Number of training images:", len(train_imgs))
    print("Number of training features loaded:", len(train_features))

    print("Number of validation images:", len(val_imgs))
    print("Number of validation features loaded:", len(val_features))

    print("Number of testing images:", len(test_imgs))
    print("Number of testing features loaded:", len(test_features))

    # Combine train and validation descriptions
    all_descriptions = {**train_descriptions, **val_descriptions}  # Merge dictionaries
    print("Number of descriptions loaded:", len(all_descriptions))

    # Create tokenizer using both train + validation descriptions
    tokenizer = create_tokenizer(all_descriptions)
    # pdb.set_trace()
    # print('<start> index: ',tokenizer.word_index['<start>'])
    pickle.dump(tokenizer, open('tokenizer.p', 'wb'))  # Save tokenizer to a file

    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    print(f"Vocabulary Size: {vocab_size}")  # Print the vocabulary size

    max_len = max_length(train_descriptions)  # Compute max length from training descriptions
    print(f"Max description length: {max_len}")  # Print max sentence length
