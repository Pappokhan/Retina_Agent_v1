# Measurements shown by the app

## Segmentation numbers

Vessel area is the share of the 512 x 512 image covered by the mask. Mean confidence is the average sigmoid probability inside the mask. Uncertain pixels is the share of the image where the probability is between 0.3 and 0.7; a high value means the model is undecided on many pixels. Model time is the forward pass only, measured on the server; it depends on the hardware and is much lower on a GPU.

## Image quality cues

These are simple acquisition cues measured on the greyscale image, not a clinical gradeability score. Brightness and contrast are the mean and standard deviation of pixel values inside the field of view, scaled 0 to 1. Sharpness is the variance of the Laplacian inside the field of view: lower values mean a blurrier image. Field of view percent is the share of the image that is not black (the circular fundus area). Overexposed percent is the share of nearly white pixels, which can hide vessels near the optic disc.

## Regions

Region numbers split the 512 x 512 image into top_left, top_right, bottom_left and bottom_right quarters plus a center square. They are image coordinates, not anatomical regions, because the app does not know which eye or which way the image is oriented. For each region the app reports vessel percent and uncertain percent. The center region often contains the optic disc and the densest vessels.

## Agreement numbers

Pairwise Dice between two variant masks is 2 x overlap / (size of mask 1 + size of mask 2); 1.0 means identical. Disagreement percent is the share of pixels where exactly one of the two masks is set.
