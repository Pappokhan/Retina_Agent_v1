# Using the app

## Basic flow

Choose a variant (Best, Final, FedAvg or FedProx), upload a retinal image and choose Segment image. The viewer shows the original with predicted vessels in cyan. Cyan is used because red overlays disappear against red and orange retinal images. The view buttons switch between Overlay, Wipe (slide between original and overlay), Probability (heatmap of the sigmoid output) and Original.

## Threshold slider

The threshold decides which probability counts as vessel (default 0.5, as in training). Lowering it adds faint, thin vessels but also more false positives; raising it keeps only confident pixels. The mask updates instantly in the browser and the numbers follow a moment later from the server. The stored prediction is reused, so the model does not run again.

## Compare variants

Compare variants runs every installed variant on the same image and shows each overlay plus pairwise Dice agreement (1.0 means identical masks) and disagreement percent (share of pixels where exactly one mask is set). High agreement does not mean the masks are correct; all variants were trained on the same data.

## Explanation views (XAI)

The XAI buttons show Attention, Grad-CAM, Uncertainty and MC dropout maps, with their numbers below the image. "Explain this map in words" asks the vision model to describe the current map. See xai.md for how to read each one.

## Assistant

Report writes a structured description with a visual check, regions, what the model focused on, uncertainty, caveats and next steps. Ask answers free-form questions and remembers the last few questions for follow-ups; "Start a new conversation" clears that memory. Explain differences describes where the variants disagree. The chat agent can run segmentation, comparison and explanations itself and answer questions about the model from these notes. Answers in English or Bangla. Answers that use these notes list their sources.
