# Leaf-Disease-Detection
This project implements a real-time leaf disease detection system using deep learning. The model is built with ResNet18 and trained on the Kaggle "New Plant Diseases" dataset, which includes 38 plant diseases and healthy leaf categories.

Key Features:

Deep Learning Model: Used ResNet18 CNN model, fine-tuned for plant disease classification.

Real-Time Detection: Integrated with a webcam using OpenCV to predict leaf diseases or healthy status in real-time.

Model Accuracy: The system predicts disease types or healthy leaves with high accuracy based on live webcam input.

Image Preprocessing: Applied resizing, normalization, and augmentation techniques to improve model performance.

Confidence Scores: Displays the predicted disease and the associated confidence score for each prediction.


Technologies Used:

Python, PyTorch, OpenCV, ResNet18, Kaggle Dataset

Transfer Learning and Data Augmentation techniques


This project aims to assist farmers and agricultural researchers by enabling early detection of leaf diseases, helping improve crop health and yields.

How to Use:

1. Clone the repository.


2. Install the required dependencies (see requirements.txt).


3. Run the leaf_disease_detection.py script to start the webcam-based disease detection.

4. I have taken the dataset from kaggle the link is here https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
     you can download the dataset from here and train the model


