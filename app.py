import streamlit as st
import os
import time
from neural_style_transfer import neural_style_transfer
import utils.utils as utils
import torch
import numpy as np
from PIL import Image

st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("Neural Style Transfer")

# Directory setup
default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')

# Ensure directories exist
os.makedirs(content_images_dir, exist_ok=True)
os.makedirs(style_images_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Get available images with error handling
try:
    content_images = [f for f in sorted(os.listdir(content_images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    style_images = [f for f in sorted(os.listdir(style_images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
except Exception as e:
    st.error(f"Error loading image directories: {e}")
    content_images = []
    style_images = []

if not content_images:
    st.error("No content images found. Please add images to the content-images directory.")
    st.stop()

if not style_images:
    st.error("No style images found. Please add images to the style-images directory.")
    st.stop()

# Sidebar for configuration
st.sidebar.title("Settings")

# Image selection
selected_content = st.sidebar.selectbox(
    "Select Content Image",
    content_images,
    index=content_images.index('figures.jpg') if 'figures.jpg' in content_images else 0
)

selected_style = st.sidebar.selectbox(
    "Select Style Image",
    style_images,
    index=style_images.index('vg_starry_night.jpg') if 'vg_starry_night.jpg' in style_images else 0
)

# Configuration with much more conservative defaults for stability
config = {
    'content_img_name': selected_content,
    'style_img_name': selected_style,
    'height': 400,
    'content_weight': 1e5,
    'style_weight': 1e3,  # Much lower for numerical stability
    'tv_weight': 1e-1,    # Lower for stability
    'optimizer': 'adam',  # Adam is more stable than L-BFGS
    'model': 'vgg19',
    'init_method': 'content',
    'saving_freq': -1,
    'content_images_dir': content_images_dir,
    'style_images_dir': style_images_dir,
    'output_img_dir': output_img_dir,
    'img_format': (4, '.jpg')  # Fixed format tuple
}

# Display current settings
st.sidebar.title("Current Settings")
st.sidebar.write(f"Model: {config['model'].upper()}")
st.sidebar.write(f"Optimizer: {config['optimizer'].upper()}")
st.sidebar.write(f"Image Height: {config['height']}")
st.sidebar.write(f"Content Weight: {config['content_weight']:.1e}")
st.sidebar.write(f"Style Weight: {config['style_weight']:.1e}")
st.sidebar.write(f"TV Weight: {config['tv_weight']:.1e}")

# Display selected images
col1, col2 = st.columns(2)

try:
    with col1:
        st.subheader("Content Image")
        content_img_path = os.path.join(content_images_dir, config['content_img_name'])
        if os.path.exists(content_img_path):
            st.image(content_img_path, use_container_width=True)
        else:
            st.error("Content image not found")

    with col2:
        st.subheader("Style Image")
        style_img_path = os.path.join(style_images_dir, config['style_img_name'])
        if os.path.exists(style_img_path):
            st.image(style_img_path, use_container_width=True)
        else:
            st.error("Style image not found")
except Exception as e:
    st.error(f"Error displaying images: {e}")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'should_stop' not in st.session_state:
    st.session_state.should_stop = False

# Start/Stop buttons
col1, col2 = st.columns([1, 1])
with col1:
    start_button = st.button("Start Neural Style Transfer", disabled=st.session_state.processing)
with col2:
    if st.button("Stop Processing", disabled=not st.session_state.processing):
        st.session_state.should_stop = True

# Processing logic
if start_button and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.should_stop = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    output_placeholder = st.empty()
    
    # Create output directory
    out_dir_name = 'combined_' + os.path.splitext(config['content_img_name'])[0] + '_' + os.path.splitext(config['style_img_name'])[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    
    def progress_callback(iteration, max_iterations, img_tensor):
        """Fixed progress callback with proper error handling"""
        try:
            # Ensure progress stays within [0, 1] range
            progress = min(float(iteration) / float(max_iterations), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing iteration {iteration}/{max_iterations}")
            
            # Display current result with proper tensor handling
            if img_tensor is not None:
                with torch.no_grad():
                    # Convert tensor to displayable image
                    img_copy = img_tensor.clone().detach()
                    
                    # Ensure proper tensor format
                    if img_copy.dim() == 4:
                        img_copy = img_copy.squeeze(0)  # Remove batch dimension
                    
                    # Convert to numpy and handle denormalization
                    img_np = img_copy.cpu().numpy()
                    
                    # Transpose from CHW to HWC
                    if img_np.shape[0] == 3:  # If channels first
                        img_np = np.transpose(img_np, (1, 2, 0))
                    
                    # Add ImageNet mean back and clip values
                    mean = np.array([0.485, 0.456, 0.406]) * 255
                    img_np = img_np + mean
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    # Display the image
                    output_placeholder.image(img_np, caption=f"Iteration {iteration}", use_container_width=True)
            
            # Check if processing should stop
            return not st.session_state.should_stop
            
        except Exception as e:
            st.error(f"Error in progress callback: {e}")
            return True  # Continue processing even if display fails
    
    # Run neural style transfer
    try:
        with st.spinner("Processing..."):
            result_path = neural_style_transfer(config, progress_callback=progress_callback)
        
        if st.session_state.should_stop:
            st.warning("Processing was stopped by user.")
        elif result_path and os.path.isdir(result_path):
            st.success("Style transfer completed!")
            
            # Find and display the final result
            try:
                output_files = sorted([f for f in os.listdir(result_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if output_files:
                    final_image_path = os.path.join(result_path, output_files[-1])
                    
                    # Display final result
                    st.subheader("Final Result")
                    st.image(final_image_path, caption="Final Result", use_container_width=True)
                    
                    # Download button
                    with open(final_image_path, "rb") as file:
                        st.download_button(
                            label="Download Result",
                            data=file.read(),
                            file_name=f"style_transfer_{os.path.basename(final_image_path)}",
                            mime="image/jpeg"
                        )
                else:
                    st.warning("No output images found in result directory.")
            except Exception as e:
                st.error(f"Error accessing result files: {e}")
        else:
            st.error("Style transfer did not complete successfully.")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
        
    finally:
        # Reset processing state
        st.session_state.processing = False
        st.session_state.should_stop = False

# Add some helpful information
st.markdown("---")
st.markdown("### Tips for Better Results:")
st.markdown("""
- **Content Weight**: Higher values preserve more content structure
- **Style Weight**: Higher values apply more style transfer
- **TV Weight**: Higher values create smoother results but may reduce detail
- **L-BFGS** optimizer generally produces better quality but is slower
- **Adam** optimizer is faster but may need more iterations
""")

# Display system information
st.markdown("---")
st.markdown("### System Information:")
device = "CUDA" if torch.cuda.is_available() else "CPU"
st.write(f"**Device**: {device}")
if torch.cuda.is_available():
    st.write(f"**GPU**: {torch.cuda.get_device_name(0)}")
st.write(f"**PyTorch Version**: {torch.__version__}")