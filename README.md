# Image_Inpainting_Using_GANs

## Overview

This project aims at implementing a <a href="https://arxiv.org/abs/1806.03589">DeepFillV2</a> based model for inpainting images using the CelebA dataset. 

## Dataset

Images from the CelebA dataset available <a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">here</a>.  The images are normalized and resized to 256x256.  For this project, only SQUARE masks of size 128x128 are used and are placed in random locations.<br><br>

The dataset is then split into training and validation (9:1 split) with a batch size of 2 for training (due to memory constraints). 

## Model

The discriminator is the same Spectral Markovian Discriminator proposed in the DeepFillV2 paper.  However, the discriminator takes in only the predicted image as input and not the concatenation of the mask and image as done in the paper.<br><br>

The generator used is the same as proposed in the paper, except for the contextual attention layer being replaced by a self attention layer as done <a href="https://github.com/avalonstrel/GatedConvolution_pytorch">here</a>.  

## Training

The main difference between the original implementation and this project is in the way the model is trained.  Along with the usual hinge loss for the generator, a reconstruction loss is added (L1 Loss) since it gave better results than without. The weights for reconstruction and hinge loss are 1 and 0.01 respectively.  Also as mentioned, the discriminator only takes in the predicted image as input.  The model was trained for a maximum of _ iterations.
