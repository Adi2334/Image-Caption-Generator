import os
import torch
from utils import data_genrator as DG
from utils import data_loader as DL
from utils import model as M
import warnings
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad # type: ignore

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Dataset paths
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    train_filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    val_filename = os.path.join(dataset_text, "Flickr_8k.devImages.txt")  # Validation file
    features_file = "data/features_aug.p"

    # Load training data
    train_imgs = DL.load_photos(train_filename)
    train_descriptions = DL.load_clean_descriptions("data/descriptions.txt", train_imgs)
    train_features = DL.load_features(train_imgs,features_file)

    # Load validation data
    val_imgs = DL.load_photos(val_filename)
    val_descriptions = DL.load_clean_descriptions("data/descriptions.txt", val_imgs)
    val_features = DL.load_features(val_imgs,features_file)

    # Tokenizer (only trained on train descriptions)
    tokenizer = DL.create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    max_len = DL.max_length(train_descriptions)  # Compute max length from training descriptions

    # Training hyperparameters
    epochs = 5
    batch_size = 128
    steps_per_epoch = len(train_descriptions) // batch_size
    val_steps = len(val_descriptions) // batch_size  # Steps for validation

    # Print dataset details
    print('Dataset: ', len(train_imgs))
    print('Train Descriptions:', len(train_descriptions))
    print('Train Photos:', len(train_features))
    print('Validation Descriptions:', len(val_descriptions))
    print('Validation Photos:', len(val_features))
    print('Vocabulary Size:', vocab_size)
    print('Max Description Length:', max_len)
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)
    #l2_lambda=0.01
    lr = 1e-3
    # Load or initialize model
    model = M.model_3(vocab_size,max_len)
    #model = load_model('/home/adi/img_cap_gen/code/weights/models_31/best_model.h5')
    optimizer = Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    ''' It is observed during training of this model that as the batch size increases learning rate can also be choodes higher 
        best - batch = 128; lr = 1e-4 '''

    # Create models directory if it doesn't exist
    os.makedirs("/home/adi/img_cap_gen/code/weights/models_31", exist_ok=True)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # Read best validation loss if exists, otherwise set to a high value
    best_val_loss = float('inf')
    best_val_loss_file = "/home/adi/img_cap_gen/code/weights/models_3_nb/best_val_loss.txt"

    if os.path.exists(best_val_loss_file):
        with open(best_val_loss_file, "r") as f:
            best_val_loss = float(f.read().strip())

    def lr_decay(epoch, lr):
        if epoch < 4:
            return lr  # Keep the initial learning rate for the first 10 epochs
        elif epoch < 8:
            return lr*0.1
        elif epoch < 12:
            return lr*0.01
        else:
            return lr*0.001

    # Training loop
    #best_val_loss = float('inf')  # Track lowest validation loss
    #best_val_loss = 5.6619
    for i in range(epochs):
        print(f'--------------- Epoch {i+1}/{epochs} ---------------')
        
        #new_lr = lr_decay(i,lr)
        #model.optimizer.learning_rate.assign(new_lr)
        #print(f'learning rate: {new_lr}')
        
        # Training and validation generator
        #train_generator = DG.data_generator_batch(train_descriptions, train_features, tokenizer, max_len, vocab_size, batch_size)
        #val_generator = DG.data_generator_batch(val_descriptions, val_features, tokenizer, max_len, vocab_size, batch_size)
        train_generator = DG.data_generator_batch_aug(train_descriptions, train_features, tokenizer, max_len, vocab_size, batch_size)
        val_generator = DG.data_generator_batch_aug(val_descriptions, val_features, tokenizer, max_len, vocab_size, batch_size)
        #train_generator = DG.data_generator(train_descriptions, train_features, tokenizer, max_len, vocab_size)
        #val_generator = DG.data_generator(val_descriptions, val_features, tokenizer, max_len, vocab_size)


        # Train and validate
        history = model.fit_generator(
            train_generator, 
            epochs=1, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=val_generator, 
            validation_steps=val_steps, 
            verbose=1
        )
        
        # Save model at each epoch
        # model.save(f"models_1/model_{i}.h5")
        
        # Save best model based on validation loss
        val_loss = history.history['val_loss'][0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save("/home/adi/img_cap_gen/code/weights/models_31/best_model.h5")
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
            # Save best val_loss to a file
            with open(best_val_loss_file, "w") as f:
                f.write(str(best_val_loss))
            
    print("Training completed!")
    
    
