import os
import torch
from utils import data_genrator as DG
from utils import data_loader as DL
from utils import model as M
import warnings
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad # type: ignore
# As there is problem with the version of CUDA used by tf and that present in system torch is needed 
# even if it is not used as it will handle it (MIRACLE MIRACLE ....)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Dataset paths
    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    train_filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    val_filename = os.path.join(dataset_text, "Flickr_8k.devImages.txt")
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
    vocab_size = len(tokenizer.word_index) + 1 
    max_len = DL.max_length(train_descriptions) 

    # Training hyperparameters
    epochs = 5
    batch_size = 128
    steps_per_epoch = len(train_descriptions) // batch_size
    val_steps = len(val_descriptions) // batch_size 

    print('Dataset: ', len(train_imgs))
    print('Train Descriptions:', len(train_descriptions))
    print('Train Photos:', len(train_features))
    print('Validation Descriptions:', len(val_descriptions))
    print('Validation Photos:', len(val_features))
    print('Vocabulary Size:', vocab_size)
    print('Max Description Length:', max_len)
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)

    lr = 1e-3
    model = M.model_3(vocab_size,max_len)
    #model = load_model('/home/adi/img_cap_gen/code/weights/models_31/best_model.h5')
    optimizer = Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    os.makedirs("/home/adi/img_cap_gen/code/weights/models_3", exist_ok=True)

    best_val_loss = float('inf')
    best_val_loss_file = "/home/adi/img_cap_gen/code/weights/models_3/best_val_loss.txt"

    if os.path.exists(best_val_loss_file):
        with open(best_val_loss_file, "r") as f:
            best_val_loss = float(f.read().strip())

    def lr_decay(epoch, lr):
        if epoch < 4:
            return lr  
        elif epoch < 8:
            return lr*0.1
        elif epoch < 12:
            return lr*0.01
        else:
            return lr*0.001

    for i in range(epochs):
        print(f'--------------- Epoch {i+1}/{epochs} ---------------')
        
        #new_lr = lr_decay(i,lr)
        #model.optimizer.learning_rate.assign(new_lr)
        #print(f'learning rate: {new_lr}')
        
        train_generator = DG.data_generator_batch_aug(train_descriptions, train_features, tokenizer, max_len, vocab_size, batch_size)
        val_generator = DG.data_generator_batch_aug(val_descriptions, val_features, tokenizer, max_len, vocab_size, batch_size)

        history = model.fit_generator(
            train_generator, 
            epochs=1, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=val_generator, 
            validation_steps=val_steps, 
            verbose=1
        )
        
        # model.save(f"models_1/model_{i}.h5")
        
        val_loss = history.history['val_loss'][0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save("/home/adi/img_cap_gen/code/weights/models_31/best_model.h5")
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
            with open(best_val_loss_file, "w") as f:
                f.write(str(best_val_loss))
            
    print("Training completed!")
    
    
