# Flower Image Classification Project

## Introduction

This repository contains the code for a flower image classification project, building upon the Udacity's AI Programming
with Python Nanodegree project (https://github.com/udacity/aipnd-project). It implements a VGG16-based classifier with
data augmentation and achieves an accuracy of 90% on the validation set and 87.5% on the testing set. The project also
includes command-line applications for training and prediction.

![inference example](assets/inference_example.png)

## Key Features

- Leverages the VGG16 architecture for image classification.
- Employs data augmentation techniques (rotation, resizing, flipping) to improve model robustness.
- Achieves high accuracy on the flower image dataset (90% validation, 87.5% testing).
- Provides command-line applications for training (`train.py`) and prediction (`predict.py`).
- Includes a Jupyter notebook for interactive exploration and development.

## Technical Details

### Data Augmentation:

- Rotates images randomly by up to 30 degrees.
- Resizes images to a fixed size (e.g., 224x224).
- Horizontally flips images.

### Model Architecture:

- Utilizes the VGG16 convolutional neural network pre-trained on ImageNet.
- Freezes the pre-trained layers to avoid overfitting and speed up training.
- Adds a custom classifier head with fully connected layers specific to the flower dataset.


## Usage

1. **Clone this repository:**
    ```bash
    git clone https://github.com/akiva-skolnik/flower-image-classification.git
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
   https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz

4. **Train the model:**
    ```bash
    python train.py /path/to/dataset --save_dir checkpoints/ --architecture vgg16 --learning_rate 0.001 --hidden_units 512 256 --epochs 10 --gpu
    ```

5. **Make predictions:**

    ```bash
    python predict.py /path/to/image.jpg --checkpoint_path checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
    ```

Feel free to contribute to this project by creating pull requests with improvements or additional features.
