# SE_4458_Pytorch_Project
 Training a CNN(Lenet 5) to distinguish between Mnist digits using Pytorch
 # Mnist Digits Classification with LeNet-5 using PyTorch

This project demonstrates how to use a Convolutional Neural Network (CNN), specifically the LeNet-5 architecture, to classify handwritten digits from the Mnist dataset. The CNN model is implemented using PyTorch.

## Introduction

In this project, we utilize the LeNet-5 architecture, originally developed by Yann LeCun, to build a model capable of identifying digits from 0 to 9 in images. The Mnist dataset is used for training and validation, which is directly accessible through PyTorch.

#LeNet-5Architecture

LeNet-5 is a Convolutional Neural Network (CNN) architecture developed by Yann LeCun and colleagues in 1998. LeNet-5 is designed for use in simple image classification tasks such as recognition of handwritten digits. Here are the key components of the LeNet-5 architecture:

## Convolutional Layers

LeNet-5 contains two convolutional layers. Each uses a bank of filters and extracts feature maps from input images. While 6 filters are used in the first convolutional layer, 16 filters are used in the second layer.

## Pooling Layers

Pooling layers are used to reduce the size of feature maps and also provide translation invariance. In LeNet-5, there is a pooling layer after each convolutional layer.

## Fully Connected Layers

Fully connected layers are used to perform classification after flattening the feature maps. LeNet-5 has two fully connected layers and one output layer.

## Activation Functions

In LeNet-5, ReLU (Rectified Linear Unit) activation functions are generally used. These functions increase the speed of learning and prevent over-adaptation.

## Summary

LeNet-5 is a simple yet effective CNN architecture and forms the basis of modern deep learning models. It provides successful results in basic classification tasks such as recognition of handwritten digits and has become a turning point in the field of deep learning.

![image](https://github.com/OzlemKlc/SE_4458_Pytorch_Project/assets/122043812/d9f7e58c-630b-41ed-b6e0-e8e71062b3dc)

![image](https://github.com/OzlemKlc/SE_4458_Pytorch_Project/assets/122043812/bcc2687b-d10e-4ec6-a9ec-581cceb5b2fb)

## Why PyTorch for CNN?

PyTorch provides several advantages for building CNNs:

1. **Dynamic Computation Graphs**: PyTorch's dynamic computation graph feature allows for more flexibility during model construction and debugging. This makes it easier to define complex architectures and modify them as needed.

2. **Pythonic Syntax**: PyTorch's Pythonic syntax is intuitive and easy to understand, making it accessible to both beginners and experienced developers. It allows for rapid prototyping and experimentation.

3. **Efficient GPU Utilization**: PyTorch seamlessly integrates with CUDA for GPU acceleration, enabling faster training and inference on compatible hardware. This significantly improves the efficiency of CNNs, especially for large-scale datasets and complex architectures.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- pandas
- numpy

## Results

The trained model achieves an accuracy of 100% on the validation dataset, demonstrating its effectiveness in classifying handwritten digits.

## Conclusion

In conclusion, this project illustrates the successful implementation of a CNN model using PyTorch for Mnist digit classification. PyTorch's flexibility, Pythonic syntax, and efficient GPU utilization make it an excellent choice for building and training CNNs for various computer vision tasks.
