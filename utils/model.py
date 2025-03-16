import os
import torch
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, BatchNormalization
from keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from keras.utils import plot_model
from utils import data_loader as DL
from keras.regularizers import l2

def model_1(vocab_size, max_length):
    """Define the image captioning model with Batch Normalization."""
    
    # Image feature extractor
    inputs1 = Input(shape=(2048,),dtype=tf.float32)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)  # Add BN after Dense layer

    # Sequence processor
    inputs2 = Input(shape=(max_length,),dtype=tf.int32)
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    se4 = BatchNormalization()(se3)  # Add BN after LSTM

    # Merging both models
    decoder1 = add([fe3, se4])
    decoder2 = Dense(256, activation='relu')(decoder1)
    decoder3 = BatchNormalization()(decoder2)  # BN before final output layer
    outputs = Dense(vocab_size, activation='softmax')(decoder3)

    # Define model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def model_2(vocab_size, max_length):
    """Define the image captioning model with Batch Normalization."""
    
    # Image feature extractor
    inputs1 = Input(shape=(2048,),dtype=tf.float32)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)  # Add BN after Dense layer

    # Sequence processor
    inputs2 = Input(shape=(max_length,),dtype=tf.int32)
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    se4 = BatchNormalization()(se3)  # Add BN after LSTM

    # Merging both models
    decoder1 = add([fe3, se4])
    decoder2 = Dense(256, activation='relu')(decoder1)
    decoder3 = BatchNormalization()(decoder2)  # BN before final output layer
    decoder4 = Dropout(0.5)(decoder3)
    outputs = Dense(vocab_size, activation='softmax')(decoder4)

    # Define model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def model_3(vocab_size, max_length):
    """Define the image captioning model with Batch Normalization."""
    
    # Image feature extractor
    inputs1 = Input(shape=(2048,),dtype=tf.float32)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)  # Add BN after Dense layer

    # Sequence processor
    inputs2 = Input(shape=(max_length,),dtype=tf.int32)
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)

    # mixing sequence and CNN output
    merg1 = add([fe3, se2])
    mix1 = BatchNormalization()(merg1)
    mix2 = LSTM(256)(mix1)
    mix3 = BatchNormalization()(mix2)  # Add BN after LSTM

    # Merging CNN output and LSTM output
    decoder1 = add([fe3, mix3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    decoder3 = BatchNormalization()(decoder2)  # BN before final output layer
    decoder4 = Dropout(0.5)(decoder3)
    outputs = Dense(vocab_size, activation='softmax')(decoder4)

    # Define model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def model_4(vocab_size, max_length):

    input1 = Input(shape=(2048,))
    input2 = Input(shape=(max_length,))

    img_features = Dense(256, activation='relu')(input1)
    img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

    sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
    merged = concatenate([img_features_reshaped,sentence_features],axis=1)
    sentence_features = LSTM(256)(merged)
    x = Dropout(0.5)(sentence_features)
    x = add([x, img_features])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=[input1,input2], outputs=output)
    return model

if __name__ == "__main__":
    
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    features_file = 'features.p'
    train_imgs = DL.load_photos(filename)
    train_descriptions = DL.load_clean_descriptions("descriptions.txt", train_imgs)
    train_features = DL.load_features(train_imgs,features_file)
    tokenizer = DL.create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    max_len = DL.max_length(train_descriptions)  # Compute max length from training descriptions
    l2_lambda=0.01

    model = model_2(vocab_size, max_len)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Summary & plot
    print(model.summary())
    plot_model(model, to_file='model_1.png', show_shapes=True)
