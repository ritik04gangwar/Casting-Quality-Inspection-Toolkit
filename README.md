# Casting Inspection Using Deep Learning

This repository contains a Jupyter notebook for casting inspection, where an image is classified as defective or perfect using a deep learning neural network.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Introduction

Casting inspection is a critical process in manufacturing industries to ensure the quality of cast products. This project utilizes deep learning to classify casting images as either defective or perfect. The goal is to automate the inspection process, reduce human error, and increase efficiency.

## Dataset

The dataset used for this project consists of images of castings that are labeled as either defective or perfect. The dataset is split into training, validation, and test sets.

## Model Architecture

The deep learning model used for this project is a convolutional neural network (CNN) designed to handle image classification tasks. The architecture includes multiple convolutional layers followed by pooling layers, and fully connected layers leading to the output layer.

## Training

The model is trained using the labeled dataset. The training process includes data augmentation, loss calculation, and optimization using backpropagation. The training process is monitored using validation accuracy and loss.

## Evaluation

The model's performance is evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score. Confusion matrix and ROC curves are also used to visualize the performance.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
