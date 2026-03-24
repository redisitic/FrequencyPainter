
# FrequencyPainter

Paint and watermark images in the frequency domain using FFT and IFFT. Every brush stroke edits the magnitude spectrum directly. The spatial image reconstructs in real time via inverse FFT with the original phase preserved.

----------

## What it does
Most image editors work on pixels. FrequencyPainter works on **frequencies**, the mathematical layer underneath pixels where brightness gradients, edges and textures actually live. Painting on the FFT magnitude and watching the spatial image warp in response gives you a fundamentally different kind of control over an image.

This also makes it a practical tool for **frequency-domain steganography**: hiding information in parts of the spectrum that are perceptually invisible and statistically hard to detect.

----------

## Steganography

Hiding data in pixel values (LSB substitution) is detectable in seconds. The pixel histogram develops a measurable statistical anomaly. The frequency domain offers better hiding spots:

-   **DC component** (center pixel after fftshift): controls global average brightness. Tiny perturbations here are invisible and survive most processing pipelines.
-   **Low-frequency ring**: broad colour and luminance gradients. Small edits are perceptually imperceptible and robust to mild compression.
-   **Phase spectrum**: the magnitude image looks completely unchanged. Phase carries all spatial structure and is almost never inspected.

FrequencyPainter lets you paint directly onto these regions, making it easy to experiment with frequency-domain embedding by hand.

----------

## Features

-   Paint on the FFT magnitude or the original image, both views stay in sync
-   RGBA watermark stamping with per-pixel alpha compositing
-   Resize brush and watermark stamp with F + scroll
-   Undo / redo up to 50 strokes (Ctrl+Z / Ctrl+Y)
-   Save either image (PNG, JPEG, BMP or TIFF)
-   Live histograms and DC / peak frequency stats

----------

## Installation

```bash
git clone https://github.com/your-username/FrequencyPainter.git
cd FrequencyPainter
pip install -r requirements.txt
python main.py
```

`tkinter` is required and ships with most Python installs. On Linux: `sudo apt install python3-tk`.