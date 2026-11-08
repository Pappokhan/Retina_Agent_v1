# Live camera (realtime mode)

## How it works

The browser plays the camera video at its full frame rate. Frames are sent to the server one at a time: a new frame is sent only when the previous result has come back, so frames never queue up and the mask always belongs to a recent frame. The server returns a transparent mask layer that is drawn over the live video. Nothing is saved and no language model is called, which keeps it fast. The browser only allows camera access on localhost or over HTTPS.

## Resolution and speed

The live input size can be 256, 320, 384, 448 or 512 px. The U-Net is fully convolutional, so it runs at any of these sizes, but it was trained at 512 px, so smaller sizes are faster approximations. Auto starts at 256 px on a CPU (512 px on a GPU) and changes the size from the measured latency: it steps down when a frame takes longer than about 650 ms and up when it takes less than about 180 ms. The overlay shows model frames per second, latency and the current size. As an example, on the development CPU one frame took about 0.35 to 0.45 s at 256 px and about 1.2 to 1.8 s at 512 px; a GPU is many times faster.

## Live explanation

The live explanation menu can add an uncertainty (entropy) or attention-gate layer under the mask. Both come from the same forward pass, so they add almost no time.

## Capture frame

Capture frame for full analysis stops the camera, uploads the current frame at 512 px and runs the normal segmentation, so all XAI views, the report and Ask can be used on it.
