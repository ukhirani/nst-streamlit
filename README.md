# 🖼️ Neural Style Transfer App

This is a deep learning-based application that performs neural style transfer — combining the content of one image with the artistic style of another to generate a new stylized image. The project uses a pretrained VGG19 model and is built using PyTorch and Streamlit for an interactive and user-friendly experience.

## About the App

The application allows users to:

* Upload a **content image** (the structure/image you want to preserve).
* Upload a **style image** (the artistic style you want to apply).
* Adjust key hyperparameters such as **content weight**, **style weight**, and **total variation loss weight**.
* Choose the **optimizer** to be used during the stylization process (Adam or L-BFGS).
* Visualize the **real-time progress** of the stylization process directly in the browser.
* View and download the final stylized image once the process completes.

## How It Works

1. The content and style images are first resized and normalized.
2. Features are extracted from a pretrained VGG19 model.
3. A new image is initialized (from content, style, or random noise).
4. Using gradient-based optimization, the pixels of this image are iteratively updated to minimize:

   * **Content loss** (difference between content features of generated and content images).
   * **Style loss** (difference between Gram matrices of style features).
   * **Total variation loss** (encourages spatial smoothness).
5. The result is a new image that captures the content of the first image and the texture/style of the second.

## Features

* **Interactive Streamlit Interface**: Upload images, tune parameters, and track progress in real time.
* **Optimizer Support**: Choose between Adam and L-BFGS based on speed or quality preference.
* **Real-Time Visualization**: View the stylized output evolving across iterations.
* **Robust Error Handling**: Handles edge cases like invalid inputs or unstable loss values.
* **Downloadable Output**: Final image is available to preview and download directly from the UI.

## Future Enhancements

* Add support for **style transfer with color preservation**.
* Include multiple style blending.
* Enable image style transfer on mobile with a lightweight backend.
* Integrate GPU acceleration on cloud for faster processing.

## Acknowledgements

This project was implemented as part of an academic exploration of computer vision and deep learning techniques, specifically convolutional neural networks and style representation via Gram matrices. Inspired by the original Neural Style Transfer algorithm proposed by Gatys et al.

