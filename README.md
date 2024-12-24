# Infosys_internship

# Brain Tumor and MNIST Classification Models

This repository contains three deep learning models for image classification tasks. Each model was trained and evaluated using different datasets. The models include:

1. **MLP (Multilayer Perceptron) for Brain Tumor Classification**
2. **Basic CNN (Convolutional Neural Network) for Brain Tumor Classification**
3. **LeNet Architecture for MNIST Digit Classification**

## Datasets

- **Brain Tumor Dataset**: Used for both the MLP and CNN models. The dataset consists of MRI scan images that are labeled as either "tumor" or "no tumor".
- **MNIST Dataset**: Used for the LeNet model. This dataset consists of 28x28 grayscale images of handwritten digits (0-9).

## Models Overview

### 1. **MLP for Brain Tumor Classification**

The Multilayer Perceptron (MLP) model is a fully connected neural network used to classify brain tumor images. The dataset is preprocessed to resize the images, flatten them, and normalize pixel values before training the model. This model achieves classification based on learned patterns in the data.

#### Key Steps:
- Load and preprocess the brain tumor dataset.
- Build and train an MLP model.
- Evaluate the model on the test dataset.

### 2. **Basic CNN for Brain Tumor Classification**

The Basic CNN model uses convolutional layers to automatically extract features from the brain tumor dataset. It includes several convolutional and pooling layers followed by dense layers for classification.

#### Key Steps:
- Load and preprocess the brain tumor dataset.
- Build and train a basic CNN model with convolutional and pooling layers.
- Evaluate the model on the test dataset.

### 3. **LeNet Architecture for MNIST Digit Classification**

LeNet is a classical Convolutional Neural Network architecture designed for digit classification. This model is trained using the MNIST dataset to classify handwritten digits (0-9). The network consists of convolutional layers followed by pooling and fully connected layers.

#### Key Steps:
- Load and preprocess the MNIST dataset.
- Build and train the LeNet model.
- Evaluate the model on the test dataset.

## Requirements

- Python 3.x
- TensorFlow or Keras
- NumPy
- Matplotlib
- scikit-learn

You can install the required libraries using the following:

```bash
pip install -r requirements.txt
