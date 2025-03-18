import numpy as np
from PIL import Image
import os
import string
from pickle import dump, load
import pdb
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()


def load_fp(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def img_capt(filename):
    file = load_fp(filename)
    captions = file.strip().split('\n')
    descriptions = {}
    for caption in captions:
        img, caption_text = caption.split('\t')
        img_key = img[:-2]
        if img_key not in descriptions:
            descriptions[img_key] = [caption_text]
        else:
            descriptions[img_key].append(caption_text)
    return descriptions

def txt_clean(captions):
    table = str.maketrans('', '', string.punctuation)   # removes all puncutations
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            descp = img_caption.split()
            descp = [wrd.lower() for wrd in descp]
            descp = [wrd.translate(table) for wrd in descp]
            descp = [wrd for wrd in descp if len(wrd) > 1]
            descp = [wrd for wrd in descp if wrd.isalpha()]
            captions[img][i] = ' '.join(descp)
    return captions

def txt_vocab(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def save_descriptions(descriptions, filename):
    lines = [key + '\t' + desc for key, desc_list in descriptions.items() for desc in desc_list]
    with open(filename, "w") as file:
        file.write("\n".join(lines))

dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"

filename = os.path.join(dataset_text, "Flickr8k.token.txt")
descriptions = img_capt(filename)
print("Length of descriptions =", len(descriptions))

clean_descriptions = txt_clean(descriptions)
pdb.set_trace()

vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary =", len(vocabulary))

save_descriptions(clean_descriptions, "data/descriptions.txt")
