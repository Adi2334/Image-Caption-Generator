import os
import torch
from pickle import load
import numpy as np
from keras.preprocessing.text import Tokenizer
import data_loader as DL
from keras.utils import to_categorical
import pdb
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

def data_generator_batch(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Generator function to yield batches of image features, input sequences, and target words."""
    while True:
        inp_image_batch, inp_seq_batch, op_word_batch = [], [], []
        
        for key, description_list in descriptions.items():
            feature = features[key][0]
            inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
            for i in range(len(inp_image)):
                inp_image_batch.append(inp_image[i])
                inp_seq_batch.append(inp_seq[i])
                op_word_batch.append(op_word[i])
                
                if len(inp_image_batch) == batch_size:
                    yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]
                    inp_image_batch, inp_seq_batch, op_word_batch = [], [], []

        if len(inp_image_batch) > 0:
            yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]

def data_generator_batch_aug1(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Generator function to yield batches of image features, input sequences, and target words."""
    while True:
        inp_image_batch, inp_seq_batch, op_word_batch = [], [], []
        
        for key, description_list in descriptions.items():
            feature = features[key]
            # pdb.set_trace()
            inp_image, inp_seq, op_word = create_sequences_aug(tokenizer, max_length, description_list, feature, vocab_size)
            for i in range(len(inp_image)):
                inp_image_batch.append(inp_image[i])
                inp_seq_batch.append(inp_seq[i])
                op_word_batch.append(op_word[i])
                
                if len(inp_image_batch) == batch_size:
                    yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]
                    inp_image_batch, inp_seq_batch, op_word_batch = [], [], []

        if len(inp_image_batch) > 0:
            yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]

## The above (old) data augmentor was not helping in training. The model trained using it was useless and gave onlt start and end tokens as output
def data_generator_batch_aug(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Generator function to yield batches of image features, input sequences, and target words."""
    while True:
        inp_image_batch, inp_seq_batch, op_word_batch = [], [], []
        
        for key, description_list in descriptions.items():
            feature = features[key][0]
            # pdb.set_trace()
            inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
            # pdb.set_trace()
            for i in range(len(inp_image)):
                inp_image_batch.append(inp_image[i])
                inp_seq_batch.append(inp_seq[i])
                op_word_batch.append(op_word[i])
                
                if len(inp_image_batch) == batch_size:
                    yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]
                    inp_image_batch, inp_seq_batch, op_word_batch = [], [], []
        
        for key, description_list in descriptions.items():
            feature = features[key][1]
            # pdb.set_trace()
            inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
            # pdb.set_trace()
            for i in range(len(inp_image)):
                inp_image_batch.append(inp_image[i])
                inp_seq_batch.append(inp_seq[i])
                op_word_batch.append(op_word[i])
                
                if len(inp_image_batch) == batch_size:
                    yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]
                    inp_image_batch, inp_seq_batch, op_word_batch = [], [], []

        # Yield remaining batch if not empty
        if len(inp_image_batch) > 0:
            yield [[np.array(inp_image_batch), np.array(inp_seq_batch)], np.array(op_word_batch)]
            

def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    """Generator function to yield image features, input sequences, and target words."""
    while True:
        for key, description_list in descriptions.items():
            feature = features[key][0]
            inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
            yield [[inp_image, inp_seq], op_word]


def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    """Converts descriptions into input-output sequence pairs."""
    x_1, x_2, y = list(), list(), list()

    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]

        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

            x_1.append(feature)
            x_2.append(in_seq)
            y.append(out_seq)

    return np.array(x_1,dtype=np.float32), np.array(x_2,dtype=np.int32), np.array(y,dtype=np.float32)


def create_sequences_aug(tokenizer, max_length, desc_list, feature_list, vocab_size):
    """Converts descriptions into input-output sequence pairs, supporting both original and flipped image features."""
    x_1, x_2, y = [], [], []

    for feature in feature_list:  
        # pdb.set_trace()
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                x_1.append(feature)   # Image feature (original or flipped)
                x_2.append(in_seq)    # Padded text sequence
                y.append(out_seq)     # Next word (one-hot encoded)
                # pdb.set_trace()

    return np.array(x_1, dtype=np.float32), np.array(x_2, dtype=np.int32), np.array(y, dtype=np.float32)


def load_dataset(dataset_text_path, descriptions_file, features_file, data_files):
    filename = os.path.join(dataset_text_path, data_files)
    
    data_imgs = DL.load_photos(filename)
    data_descriptions = DL.load_clean_descriptions(descriptions_file, data_imgs)
    data_features = DL.load_features(data_imgs, features_file)

    tokenizer = DL.create_tokenizer(data_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = DL.max_length(data_descriptions)

    return data_imgs, data_descriptions, data_features, tokenizer, vocab_size, max_len


if __name__ == "__main__":
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    train_files = "Flickr_8k.trainImages.txt"
    descriptions_file = "descriptions.txt"
    features_aug_file = "data/features_aug.p"
    features_file = "data/features.p"

    train_imgs, train_descriptions, train_features, tokenizer, vocab_size, max_len = load_dataset(dataset_text, descriptions_file, features_file, train_files)

    print("Number of training images:", len(train_imgs))
    print("Number of descriptions loaded:", len(train_descriptions))
    print("Number of features loaded:", len(train_features))
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max description length: {max_len}")

    features = load(open(features_file, "rb"))
    features_aug = load(open(features_aug_file, "rb"))
    # [a, b], c = next(data_generator(train_descriptions, features, tokenizer, max_len, vocab_size))
    # [a, b], c = next(data_generator_batch(train_descriptions, features, tokenizer, max_len, vocab_size,2))
    [a, b], c = next(data_generator_batch_aug(train_descriptions, features_aug, tokenizer, max_len, vocab_size,2))
    pdb.set_trace()
    print(a.shape, b.shape, c.shape)


