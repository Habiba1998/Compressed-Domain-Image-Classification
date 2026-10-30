# Compressed-Domain-Image-Classification

## 1. Project Overview

Image classification is a computer vision task that aims to assign a category label to an input image. Traditional image classification models operate on decoded RGB images, which requires JPEG images to be decompressed into the pixel domain before inference.

In this project, we investigate **Compressed-Domain Image Classification**, an approach that aims to classify images directly from JPEG Discrete Cosine Transform (DCT) coefficients without fully decoding them into RGB pixels.

We will use the CIFAR-10 dataset provided by TensorFlow/Keras and compare a conventional RGB-based classification model with a DCT-domain classifier. The main objective is to investigate the trade-off between classification accuracy and inference speed while exploring whether operating in the compressed domain can reduce computational overhead.

## 2. Project Objectives

- Extract JPEG DCT coefficients and prepare them as inputs for a compressed-domain classifier.
- Train a conventional ResNet-based model using decoded RGB images as the baseline.
- Design and train a classifier that operates directly on JPEG DCT coefficients.
- Compare the classification accuracy and inference time of both approaches.
- Investigate the importance of different DCT frequency bands through ablation experiments.
- Evaluate the potential benefits and limitations of compressed-domain image classification.

## 3. Dataset

**Dataset:** CIFAR-10

**Source:** TensorFlow/Keras (`tensorflow.keras.datasets.cifar10`)

CIFAR-10 is a standard image classification dataset containing 60,000 color images of size 32 × 32 pixels, divided into 10 classes.

The dataset contains:
- 50,000 training images.
- 10,000 test images.
- 10 object categories, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The original CIFAR-10 images will be used to train and evaluate the RGB baseline. For the compressed-domain approach, images will be encoded as JPEG and their quantized DCT coefficients will be extracted directly from the JPEG files.

**Important:** The original CIFAR-10 images are not provided as JPEG files by TensorFlow/Keras. Therefore, JPEG encoding is required before extracting JPEG DCT coefficients. The JPEG quality setting and encoding procedure will be kept consistent throughout the experiments.

## 4. Proposed Approaches

### Approach 1: RGB-Domain Baseline

A ResNet-based classifier will be trained using decoded RGB images.

The pipeline is:

1. Load CIFAR-10 from TensorFlow/Keras.
2. Preprocess and normalize the RGB images.
3. Train the ResNet-based classifier.
4. Record test accuracy and inference time, including image decoding where applicable.

This model will serve as the reference for evaluating the compressed-domain approach.

### Approach 2: JPEG DCT-Domain Classification

A neural network will be trained using JPEG DCT coefficients rather than fully decoded RGB pixels.

The pipeline is:

1. Encode the CIFAR-10 images as JPEG.
2. Extract the quantized DCT coefficients from the JPEG files.
3. Prepare the coefficients in a suitable tensor representation.
4. Design and train a DCT-domain classifier.
5. Record test accuracy and inference time, including DCT extraction or parsing costs where applicable.

This approach aims to investigate whether image classification can be performed efficiently without reconstructing the complete RGB image.

### Approach 3: DCT Frequency-Band Ablation

An ablation study will investigate how different DCT frequency components affect classification performance.

We will compare models using:
- Low-frequency DCT coefficients.
- Low- and mid-frequency coefficients.
- A broader range of DCT coefficients.

The results will help identify which frequency components contribute most to classification accuracy and whether restricting the input to selected coefficients improves computational efficiency.

## 5. Evaluation Metrics

The two main approaches will be evaluated using the following metrics:

- **Classification accuracy:** The percentage of correctly classified test images.
- **Inference time:** The time required to process a fixed number of images.
- **Throughput:** The number of images processed per second.
- **Model parameters:** The number of trainable parameters, to help ensure a fair architectural comparison.
- **Frequency-band performance:** The change in accuracy and speed when different DCT frequency components are used.

All timing experiments will use consistent hardware, batch sizes, and measurement procedures. We will distinguish model-only inference time from end-to-end processing time, including JPEG decoding or coefficient extraction as appropriate.

## 6. Expected Outcomes

By the end of this project, we expect to:

- Develop a working JPEG DCT extraction pipeline.
- Train an RGB-domain baseline and a compressed-domain classifier.
- Compare their classification accuracy and processing speed.
- Determine the impact of DCT frequency selection on performance.
- Discuss the trade-off between computational efficiency and classification accuracy.

## 7. Reference



