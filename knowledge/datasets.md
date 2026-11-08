# Dataset A

## What the model was trained on

The app serves one model family: the Attention U-Net trained in the Dataset A notebook (Dataset_A_Final_Code). Dataset A is a set of retinal fundus images paired with binary vessel masks, loaded in the notebook with a class called RetinaDataset. The notebook does not state where the images come from, so the original source, camera and population are not documented here.

## Size and split

How many images: Dataset A has 800 image and mask pairs. The notebook splits them with random_split and seed 42 into 640 training, 80 validation and 80 test images (80% / 10% / 10%). Training used batch size 8.

## Folder name versus notebook letter

Inside the notebook the data folder is called "Dataset D", which does not match the notebook letter. The app always uses the notebook letter, so "Dataset A" in the app means the model from notebook A.

## Other datasets

The original project also trained models on three much smaller datasets (B, C and D, with 100, 45 and 28 images). Those models are not part of this app and the app cannot run them. Questions about them cannot be answered from these notes.

## What the test set size means

With 80 test images, Dataset A gives the steadiest numbers of the project. Small differences between variants (for example 0.005 Dice) are still close to noise and should not be over-interpreted.
