# Sneaker Classification using ResNet50
This repository contains a TensorFlow/Keras project for classifying sneakers into 50 different classes using a fine-tuned ResNet50 model. The project includes data loading, visualization, data augmentation, model building, training, and evaluation.

# Dataset
The dataset used in this project is the "Sneakers Classification" dataset, which is expected to be located at [/kaggle/input/sneakers-classification/sneakers-dataset/sneakers-dataset](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification). The dataset statistics are provided in a dataset_stats.csv file, which is loaded and visualized to understand the class distribution.

## Project Description

In this project, I built a deep learning-based image classification system to identify various types of sneakers using a real-world dataset from Kaggle. The core objective was to apply transfer learning with a pre-trained convolutional neural network (CNN), adapt it to the sneaker dataset, and achieve high classification performance.

The pipeline began with loading and preprocessing the sneaker images using TensorFlow’s image_dataset_from_directory. The dataset was split into training and validation sets, and all images were resized to 224×224 pixels to match the input shape required by ResNet50. Preprocessing also included shuffling, batching, and normalization using the preprocess_input method from tensorflow.keras.applications.

Before modeling, I performed an initial data analysis with pandas and matplotlib to visualize image statistics and check the class distribution. This step helped ensure the dataset was balanced and suitable for supervised learning.

For the model architecture, I used a pre-trained ResNet50 as a feature extractor, removing its top classification layers and freezing its convolutional base. On top of this, I added custom fully connected layers, including GlobalAveragePooling2D, batch normalization, dropout for regularization, and a dense softmax output layer matching the number of sneaker classes.

The model was compiled with the Adam optimizer and categorical cross-entropy loss, and it was trained with callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to optimize performance and avoid overfitting. The training process was monitored using accuracy and loss metrics for both training and validation sets. I also created a custom function to plot training history for easier visualization and performance analysis.

After training, I evaluated the model using classification metrics from scikit-learn, including accuracy, precision, recall, and F1-score, and visualized the results using a confusion matrix. The model demonstrated strong generalization capabilities and performed well on sneaker classification tasks.

This project showcases my skills in computer vision, deep learning, transfer learning, and model evaluation using TensorFlow and the broader Python data science stack, including NumPy, pandas, matplotlib, and scikit-learn.

---

## Features

✅ Image classification using Convolutional Neural Networks (CNN)  
✅ Transfer Learning with ResNet50 backbone  
✅ Fine-tuning and hyperparameter optimization  
✅ Overfitting prevention with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint  
✅ Model evaluation using accuracy and classification reports  
✅ Data visualization with Matplotlib and Seaborn  

---

## Tech Stack

- Python 3.x  
- TensorFlow & Keras  
- ResNet50 
- Matplotlib & Seaborn for visualization  
- Scikit-learn for evaluation metrics  

---

