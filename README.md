# **Image Caption Generator using Deep Learning**  

## **Overview**  
This project implements an **image captioning model** using a **CNN-LSTM architecture**. The model takes an image as input and generates a descriptive caption using natural language processing techniques. It is trained on a dataset containing images and their corresponding textual descriptions.

## **Dataset**  
- The model is trained on **Flickr8k** dataset.  
- It consists of **8000** images with multiple captions per image.  
#### **Data Augmentation**  
To improve model performance, images were **horizontally flipped**.

## **Model Architecture**  
The model consists of three main components:  
1. **Image Feature Extractor (CNN)**  
   - Uses **Xception** to extract feature from images.  
2. **Sequence Processor (LSTM)**  
   - An **embedding layer** processes input text sequences.  
   - An **LSTM network** learns dependencies between words in a sentence.  
3. **Decoder (Dense Layer with Softmax)**  
   - Combines image features and text sequences.  
   - Generates the next word in the caption.  

![model-image](/home/adi/img_cap_gen/code/model_1.png)
To view the model architecture in detail you may use [Netron](https://netron.app/) by uploading saved model.

<!-- ## **Training Details**  
- **Optimizer:** Adam  
- **Loss function:** Categorical Crossentropy  
- **Batch size:** 64  
- **Training & Validation split:** 30,000 images for training, 1,000 images for validation.   -->

## **Evaluation Metrics**  
The model is evaluated using the following metrics:  
📌 **BLEU-1:** 0.6131  
📌 **BLEU-2:** 0.5453  
📌 **BLEU-3:** 0.4483  
📌 **BLEU-4:** 0.3635  
📌 **ROUGE-L:** 0.3314  
📌 **CIDEr:** 0.0497  
📌 **SPICE:** 0.0451  

## **How to Use**  
#### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
```

#### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

#### **3. Run the Model**  
To test the model with your own images:  
```bash
python predict.py --image_path path/to/image.jpg
```

#### **4. Streamlit Web App**  
Run the **Streamlit** interface for uploading images and generating captions:  
```bash
streamlit run app.py
```

## **Results**  
Example output from the model:  

| **Input Image** |  ![example-image](path/to/image.jpg) |
|---------------|-------------------|
|**Generated Caption** | *"A man is riding a bicycle on the street."* |

## **Future Improvements**  
🔹 Train on a larger dataset for improved generalization.  
🔹 Experiment with **Transformer-based models** (e.g., ViT + GPT-2, BLIP, OFA).  
🔹 Implement **beam search** for better caption generation.  
🔹 Optimize the model using **reinforcement learning (CIDEr optimization)**.  

## **Contributor**  
👤 **Aditya Nikam** student at IIT Kanpur
    - contact (23alpha34@gmail.com / adityarn21@iitk.ac.in) 
  

