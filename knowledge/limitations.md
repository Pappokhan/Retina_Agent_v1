# Limitations and safe use

## Research use only

This app is research software. It is not a medical device, it has not been validated for clinical use, and it must not be used to diagnose, screen for or treat any condition. Segmentation output, explanation maps and the assistant's descriptions are not medical advice. Anyone with a health concern should see a clinician.

## What the numbers mean

Vessel area, mean confidence and uncertain pixels are computed from the model's own output. They describe the mask, not its correctness. The app has no ground truth for an uploaded image, so it cannot measure accuracy. The accuracy figures in model_card.md come from the notebook's held-out test split only.

## Visual quality check

The visual check in a report is done by a vision-language model that looks at the image, the overlay and the explanation maps. It is a sanity check (mask on background or on the optic-disc rim, obvious misses, broken thin vessels, over-segmentation). It does not measure accuracy, it can be wrong, and it has not been validated against Dice scores.

## Explanations are not proof

Attention maps, Grad-CAM and uncertainty maps show how the model behaved on this image. A bright explanation map on a vessel does not prove the vessel is real, and a confident prediction can still be wrong. Use the maps to find areas worth checking, not to certify a result.

## Domain shift

The model was trained on Dataset A only. Images from a different camera, field of view, resolution or population can give poor masks, and the model does not warn you. Warning signs: low sharpness, low contrast, a small field of view, many uncertain pixels or high MC dropout variance. The input is squashed to 512 x 512 without preserving aspect ratio, exactly as in training.

## Live mode at reduced resolution

The live camera can run at 256 to 448 px for speed. The model was trained at 512 px, so masks at smaller sizes are approximations: thin vessels are more likely to be missed. Use Capture frame for a full-resolution analysis.

## Metric caveats

The clDice value printed in the notebook (0.5172 for Dataset A) comes from an erosion-based approximation, not from skeletonisation, so it is not comparable with published clDice numbers. The notebook reports test metrics for the final-epoch model, not the best-validation checkpoint.

## Privacy

Uploaded images are stored on the server only temporarily and are deleted after a few hours. The explanation features (report, ask, explain map, compare) and the chat assistant send the image, the overlay and explanation maps to the configured model provider (by default Google Gemini). On a free tier the provider may use submitted content to improve its products and people may review it. Do not upload patient data unless your data agreement and consent allow it. The live camera, segmentation and the XAI maps themselves run on the server only and call no model API.
