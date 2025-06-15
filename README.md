# Sneaker Classification using ResNet50
This repository contains a TensorFlow/Keras project for classifying sneakers into 50 different classes using a fine-tuned ResNet50 model. The project includes data loading, visualization, data augmentation, model building, training, and evaluation.

# Dataset
The dataset used in this project is the "Sneakers Classification" dataset, which is expected to be located at [/kaggle/input/sneakers-classification/sneakers-dataset/sneakers-dataset](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification). The dataset statistics are provided in a dataset_stats.csv file, which is loaded and visualized to understand the class distribution.

# Project Structure
The notebook is structured as follows:

Import Libraries: Imports all necessary libraries for data manipulation, visualization, and model building.
Visualizing Data: Includes code to load dataset statistics and visualize the image count and contribution per class using bar and pie charts.
Data Loading and Preprocessing: Loads the image dataset from the specified directory, splits it into training and validation sets, and demonstrates sample images with their labels.
Creating Test Data: Splits the validation dataset further into validation and test sets.
Configure the Dataset: Applies prefetching to the datasets for performance improvement.
Use Data Augmentation: Defines and demonstrates data augmentation techniques to increase the size and variability of the training data.
Creating the Model using Pretrained ResNet50: Loads a pre-trained ResNet50 model, freezes its initial layers, and builds a new model by adding a classification head on top of the pre-trained model.
Model Compilation: Defines the optimizer, loss function, and metrics for the model.
Define Callbacks: Sets up callbacks for early stopping, model checkpointing, and learning rate reduction.
Train the Model: Trains the model using the prepared training and validation datasets.
Evaluating the Model: Evaluates the trained model on the test dataset and prints the test accuracy and loss.
Classification Report: Generates a detailed classification report showing precision, recall, and f1-score for each class.
# Requirements
TensorFlow 2.x
Keras
Matplotlib
Seaborn
NumPy
Pandas
Scikit-learn
