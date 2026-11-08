# Model card: Attention U-Net (Dataset A)

## Architecture

All four checkpoints use the same architecture: a U-Net with additive attention gates on the skip connections (Oktay et al. style). It has 13,746,861 trainable parameters, takes a 3-channel image and outputs one logit per pixel for binary vessel segmentation.

The encoder has an initial double convolution and four down blocks (max-pool then double convolution) with widths 64, 128, 256, 512 and 512. Each double convolution is Conv3x3, BatchNorm, ReLU, Dropout2d(0.1), Conv3x3, BatchNorm, ReLU, Dropout2d(0.1). The decoder has four up blocks (up1 to up4) that use bilinear upsampling, an attention gate on the skip features, concatenation and a double convolution. The output layer is a 1x1 convolution from 64 channels to 1. Dropout is switched off at inference, except in the MC dropout explanation.

## Attention gates

Each decoder block has one attention gate. The gate combines the upsampled decoder features (gating signal) with the encoder skip features, and produces a coefficient psi between 0 and 1 for every pixel. The skip features are multiplied by psi before they are passed on, so psi shows which areas the model lets through. up1 works at 64 x 64 (coarse), up2 at 128, up3 at 256 and up4 at 512 x 512 (fine).

## Input and output

Input is resized to 512 x 512 (aspect ratio is not preserved), converted to a tensor and normalised with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). The logits go through a sigmoid and are binarised at 0.5 by default. The app's threshold slider changes only this last step.

## Training setup

How the model was trained: loss is Dice plus binary cross-entropy. Optimizer is AdamW with weight decay 1e-5 and mixed precision. The learning rate follows a 10-epoch warm-up and a cosine decay. Gradient clipping is 0.5. Augmentation is random horizontal flip, vertical flip and rotation up to 15 degrees. Training runs for up to 200 epochs with early stopping (patience 20) on validation Dice. Batch size 8, seed 42.

## The four checkpoint variants

Best validation (best_model_AttentionUNet.pth) holds the weights from the epoch with the highest validation Dice. Final epoch (standard_attention_unet.pth) holds the weights at the end of training; this is the model the notebook evaluated on the test set. FedAvg and FedProx come from the federated runs described in federated_setup.md.

## Reported test results

Final-epoch model on the 80 Dataset A test images: Dice 0.9295, IoU 0.8682, sensitivity 0.9261, specificity 0.9948, AUC-ROC 0.9923, HD95 4.43 pixels. Best validation Dice was 0.9355 at epoch 156; early stopping stopped training at epoch 176.

## How the metrics were computed

Dice, IoU, sensitivity, specificity and AUC-ROC are computed over all test pixels pooled together, not averaged per image. HD95 is computed per image on the 512 x 512 masks and then averaged, so it is in pixels. Pixel accuracy is dominated by background and is not a useful headline number.
