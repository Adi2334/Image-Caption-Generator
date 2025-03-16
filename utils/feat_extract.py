import numpy as np
import os
import warnings
import tensorflow as tf
import torch
import gc
from PIL import Image
from pickle import dump, load
from tqdm import tqdm
from tensorflow.keras.applications.xception import Xception # type: ignore

warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, image_directory, model_name='Xception', feature_file='features.p'):
        self.image_directory = image_directory
        self.feature_file = feature_file
        self.model = self._load_model(model_name)
        # self._enable_gpu()

    def _enable_gpu(self):
        """Enable GPU support if available."""
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Num GPUs Available:", len(physical_devices))

    def _load_model(self, model_name):
        """Load the specified pre-trained model for feature extraction."""
        if model_name == 'Xception':
            return Xception(include_top=False, pooling='avg')
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def extract_features(self):
        """Extract features from all images in the specified directory."""
        features = {}
        for pic in tqdm(os.listdir(self.image_directory)):
            file_path = os.path.join(self.image_directory, pic)
            image = self._preprocess_image(file_path)
            feature = self.model.predict(image)
            features[pic] = feature
        return features
    
    def extract_features_opt(self, batch_size=16):
        """Extract features efficiently using batching."""
        features = {}
        image_files = os.listdir(self.image_directory)

        # Divide into 4 parts
        num_parts = 4
        total_images = len(image_files)
        part_size = total_images // num_parts
        remainder = total_images % num_parts
        start_idx = 0

        for part in range(num_parts):
            end_idx = start_idx + part_size + (1 if part < remainder else 0)
            part_files = image_files[start_idx:end_idx]

            print(f"Processing part {part + 1}/{num_parts} with {len(part_files)} images...")

            # **Process in batches**
            for i in tqdm(range(0, len(part_files), batch_size)):
                batch_files = part_files[i:i + batch_size]

                # Load and preprocess images in a batch
                batch_images = [self._preprocess_image(os.path.join(self.image_directory, pic),False) for pic in batch_files]
                batch_images = np.vstack(batch_images)  # Convert list to array

                # **Use GPU-efficient batch processing**
                batch_features = self.model.predict(batch_images, batch_size=batch_size)

                for j, pic in enumerate(batch_files):
                    features[pic] = batch_features[j]

                # **Clear memory every batch**
                del batch_images, batch_features
                gc.collect()

            start_idx = end_idx  # Move to next part

        return features
    

    def extract_features_aug(self, batch_size=16):
        """Extract features efficiently using batching."""
        features = {}
        image_files = os.listdir(self.image_directory)

        # Divide into 4 parts
        num_parts = 4
        total_images = len(image_files)
        part_size = total_images // num_parts
        remainder = total_images % num_parts
        start_idx = 0

        for part in range(num_parts):
            end_idx = start_idx + part_size + (1 if part < remainder else 0)
            part_files = image_files[start_idx:end_idx]

            print(f"Processing part {part + 1}/{num_parts} with {len(part_files)} images...")

            # **Process in batches**
            for i in tqdm(range(0, len(part_files), batch_size)):
                batch_files = part_files[i:i + batch_size]

                # Load and preprocess images in a batch
                batch_images = [self._preprocess_image(os.path.join(self.image_directory, pic),False) for pic in batch_files]
                batch_images = np.vstack(batch_images)  # Convert list to array

                # Load and preprocess flipped images in a batch
                batch_images_aug = [self._preprocess_image(os.path.join(self.image_directory, pic),True) for pic in batch_files]
                batch_images_aug = np.vstack(batch_images_aug)

                batch_features = self.model.predict(batch_images, batch_size=batch_size)
                batch_features_aug = self.model.predict(batch_images_aug, batch_size=batch_size)

                
                for j, pic in enumerate(batch_files):
                    features[pic] = []
                    features[pic].append(batch_features[j])
                # adding features of flipped images
                for j, pic in enumerate(batch_files):
                    features[pic].append(batch_features_aug[j])

                # **Clear memory every batch**
                del batch_images, batch_images_aug, batch_features, batch_features_aug
                gc.collect()

            start_idx = end_idx  # Move to next part

        return features


    def _preprocess_image(self, file_path, flip=False):
        """Load and preprocess an image for the model."""
        image = Image.open(file_path)
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)   
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5 - 1.0  # Normalize

        return image

    def save_features(self, features):
        """Save extracted features to a pickle file."""
        dump(features, open(self.feature_file, "wb"))
        print(f"Features saved to {self.feature_file}")

    def load_features(self):
        """Load features from a pickle file."""
        return load(open(self.feature_file, "rb"))

if __name__ == "__main__":
    dataset_images = "/home/adi/img_cap_gen/Data/Flickr8k_Dataset"
    extractor = FeatureExtractor(dataset_images,feature_file='features_aug.p')
    features = extractor.extract_features_aug()
    # features = extractor.extract_features_opt()
    extractor.save_features(features)
