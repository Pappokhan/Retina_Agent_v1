# Explainability (XAI) maps

All four maps are computed from the real model on the server; nothing is simulated. In every map dark means low and bright yellow means high.

## Attention gate map

Shows the psi coefficients of the model's four attention gates (up1 to up4), captured during the normal forward pass, so it costs nothing extra. The combined map is the mean of the four gates resized to 512 x 512; each gate can also be shown alone. "Attention on vessels" is the share of a gate's total attention that falls on predicted vessel pixels. The focus ratio divides that share by the vessel area: a ratio above 1 means the gate concentrates on vessels more than their area alone explains, a ratio near 1 means it attends everywhere about equally. Mid-resolution gates (up2, up3) usually focus most on vessels; the coarse gate up1 and the fine gate up4 are more diffuse.

## Grad-CAM (HiResCAM)

Shows which pixels of a chosen layer most increased the vessel score. It uses the HiResCAM form: the element-wise product of the layer's gradient and activation, summed over channels and passed through ReLU. Unlike classic Grad-CAM, which averages gradients over the whole image, this stays faithful to a local target. By default the target is the sum of logits over all predicted vessel pixels. Clicking the image sets a circular region: the target becomes the predicted vessels inside that region, or the whole region if it holds no predicted vessel. "Evidence inside region" is the share of the map's total that falls inside the region; a high value means the decision for that area was driven by local features. The layer menu goes from the coarse bottleneck (down4, 32 px) through decoder 1 to 3 (64, 128 and 256 px); finer layers give sharper maps. It needs one backward pass, so it takes a few seconds on CPU.

## Uncertainty (entropy)

Binary entropy of the predicted probability, in bits: 0 when the model is certain (probability near 0 or 1) and 1 when the probability is 0.5. It is computed from the single prediction, so it is free. It is usually highest along vessel edges, at thin vessel ends and where vessels cross. "Pixels with entropy above 0.5" is a quick measure of how much of the image is undecided.

## Monte-Carlo dropout

Runs the model several times (4, 8 or 16 passes) with only its Dropout2d layers switched on (BatchNorm stays in inference mode) and shows the per-pixel standard deviation. Bright pixels change between passes, which indicates model uncertainty rather than image ambiguity alone. "Dice, MC mean vs single pass" compares the averaged prediction with the normal one; a value close to 1 means the prediction is stable. It costs one full forward pass per run.

## How to use the maps together

A trustworthy-looking result has attention and Grad-CAM evidence concentrated on vessels, low entropy away from vessel edges and low MC variance. Bright uncertainty or MC variance over large areas, or attention spread over the background, suggests an unfamiliar image (see domain shift in limitations.md) or a threshold worth adjusting. These maps explain model behaviour; they do not measure accuracy.
