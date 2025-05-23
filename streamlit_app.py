import streamlit as st
from PIL import Image
import numpy as np
import os
import tempfile
import shutil
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import torch first as it's a heavy dependency
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    logger.error(f"Error importing PyTorch: {e}")
    raise

# Try to import OpenCV
try:
    import cv2
    logger.info(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    logger.error(f"Error importing OpenCV: {e}")
    raise

# Try to import NST module
try:
    from NST import neural_style_transfer, prepare_img, load_image
    logger.info("Successfully imported NST module")
except Exception as e:
    logger.error(f"Error importing NST module: {e}")
    raise

# Set page config
st.set_page_config(
    page_title="Neural Style Transfer by Umang Hirani",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #9d50bb, #6e48aa);
        color: white;
    }
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #6e48aa, #9d50bb) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# Neural Style Transfer")
st.markdown("###### _by Umang Hirani_")


# Create temporary directory for processing
@st.cache_resource
def create_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), 'nst_temp')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

temp_dir = create_temp_dir()

# Function to save uploaded file to temp directory
def save_uploaded_file(uploaded_file, filename):
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    logger.info("Starting Streamlit app")
    
    # Create a temporary directory for file operations
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check available memory
        if hasattr(os, 'sysconf'):
            if 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
                page_size = os.sysconf('SC_PAGE_SIZE')
                phys_pages = os.sysconf('SC_PHYS_PAGES')
                total_memory = (page_size * phys_pages) / (1024 ** 3)  # Convert to GB
                logger.info(f"Total system memory: {total_memory:.2f} GB")
        
        # Check disk space
        total, used, free = shutil.disk_usage("/")
        logger.info(f"Disk space - Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB")
        
        # Main layout
        st.title("Neural Style Transfer")
        st.markdown("###### _by Umang Hirani_")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### Content Image")
            content_image = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'], key='content_image')
            if content_image is not None:
                try:
                    image = Image.open(content_image)
                    st.image(image, use_container_width=True, caption="Content Image")
                    logger.info(f"Content image loaded successfully. Size: {image.size}, Mode: {image.mode}")
                except Exception as e:
                    logger.error(f"Error loading content image: {e}")
                    st.error(f"Error loading content image: {e}")

        with col2:
            st.markdown("### Style Image")
            style_image = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'], key='style_image')
            if style_image is not None:
                try:
                    image = Image.open(style_image)
                    st.image(image, use_container_width=True, caption="Style Image")
                    logger.info(f"Style image loaded successfully. Size: {image.size}, Mode: {image.mode}")
                except Exception as e:
                    logger.error(f"Error loading style image: {e}")
                    st.error(f"Error loading style image: {e}")

        with col3:
            st.markdown("### Parameters")
            
            # Model parameters
            height = st.slider("Image Height", 256, 1024, 400, 32, 
                            help="Height of the output image (higher = more detail but slower)")
            
            content_weight = st.slider("Content Weight", 1000.0, 1000000.0, 100000.0, 1000.0,
                                    help="How much to prioritize content preservation")
            
            style_weight = st.slider("Style Weight", 1000.0, 1000000.0, 30000.0, 1000.0,
                                    help="How much to prioritize style transfer")
            
            tv_weight = st.slider("Total Variation Weight", 0.0, 1000.0, 1.0, 0.1,
                                help="Controls the smoothness of the output")
            
            # Add a button to trigger style transfer
            if st.button("Apply Style Transfer", key="style_transfer_btn"):
                if content_image is None or style_image is None:
                    st.error("Please upload both content and style images")
                else:
                    try:
                        st.write("Saving uploaded files...")
                        # Save uploaded files
                        content_path = save_uploaded_file(content_image, "content.jpg")
                        style_path = save_uploaded_file(style_image, "style.jpg")
                        st.write(f"Content image saved to: {content_path}")
                        st.write(f"Style image saved to: {style_path}")
                        
                        # Create output directory
                        output_dir = os.path.join(temp_dir, "output")
                        os.makedirs(output_dir, exist_ok=True)
                        st.write(f"Output directory: {output_dir}")
                        
                        # Prepare config
                        config = {
                            'content_img_name': "content.jpg",
                            'style_img_name': "style.jpg",
                            'content_images_dir': temp_dir,
                            'style_images_dir': temp_dir,
                            'output_img_dir': output_dir,
                            'img_format': (4, '.jpg'),
                            'height': height,
                            'content_weight': content_weight,
                            'style_weight': style_weight,
                            'tv_weight': tv_weight
                        }
                        
                        st.write("Starting style transfer...")
                        st.json({"config": config})  # Show the config being used
                        
                        # Run style transfer with progress
                        with st.spinner("Running style transfer... This may take a few minutes..."):
                            import time
                            start_time = time.time()
                            results_path = neural_style_transfer(config)
                            end_time = time.time()
                            st.write(f"Style transfer completed in {end_time - start_time:.2f} seconds")
                            st.write(f"Results saved to: {results_path}")
                            
                            # Display result
                            if results_path and os.path.exists(results_path):
                                # Find the latest output image
                                output_files = [f for f in os.listdir(results_path) if f.endswith('.jpg')]
                                if output_files:
                                    output_files.sort()
                                    latest_output = os.path.join(results_path, output_files[-1])
                                    output_image = Image.open(latest_output)
                                    
                                    # Display in the output column
                                    with col3:
                                        st.markdown("### Output Image")
                                        st.image(output_image, use_container_width=True, caption="Stylized Output")
                                        
                                        # Add download button
                                        with open(latest_output, "rb") as file:
                                            st.download_button(
                                                label="Download Image",
                                                data=file,
                                                file_name=f"stylized_{os.path.basename(content_image.name)}",
                                                mime="image/jpg"
                                            )
                            else:
                                st.error("Style transfer completed but no output was generated.")
                            
                    except Exception as e:
                        import traceback
                        logger.error(f"Error during style transfer: {e}")
                        logger.error(traceback.format_exc())
                        st.error("An error occurred during style transfer:")
                        st.code(traceback.format_exc())
                        st.error(f"Error details: {str(e)}")
            
            # Display output image if it exists
            if 'output_image' in locals():
                st.markdown("### Output Image")
                st.image(output_image, use_container_width=True, caption="Stylized Output")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An unexpected error occurred: {e}")
        st.code(traceback.format_exc())
    finally:
        # Add some spacing  
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Add a footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
                <p>Neural Style Transfer App | Created with ❤️ by Umang Hirani</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



