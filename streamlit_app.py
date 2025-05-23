import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import time
import asyncio
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Fix for asyncio event loop on Windows
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

# Progress callback function
def update_progress(progress, message):
    """Update the progress bar and status message."""
    if 'progress_bar' in st.session_state and 'status_text' in st.session_state:
        st.session_state.progress_bar.progress(progress)
        st.session_state.status_text.text(message)
    return

def main():
    # Initialize temp_dir at module level
    global temp_dir
    
    try:
        # Set page config
        st.set_page_config(
            page_title="Neural Style Transfer",
            page_icon="🎨",
            layout="wide"
        )
        
        # Set title and description
        st.markdown("# Neural Style Transfer")
        st.markdown("###### _by Umang Hirani_")
        
        # Initialize temp directory
        try:
            temp_dir = create_temp_dir()
            logger.info(f"Using temp directory: {temp_dir}")
            
            # Create output directory
            output_dir = Path(temp_dir) / 'output'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Verify we can write to the temp directory
            test_file = output_dir / '.test_write'
            test_file.touch()
            test_file.unlink(missing_ok=True)
            
            # Store in session state for Streamlit
            st.session_state.temp_dir = temp_dir
            
        except Exception as e:
            logger.error(f"Failed to initialize temporary directory: {e}")
            st.error(f"Failed to initialize temporary storage. Please check permissions and try again.")
            st.stop()
        
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
        st.error(f"Failed to initialize the application: {str(e)}")
        st.stop()

    # Initialize session state for progress tracking
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        st.session_state.progress = 0
        st.session_state.status = "Ready"
    
    # Add custom CSS for better styling
    st.markdown(""" 
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .status-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .file-info {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

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

@st.cache_resource
def create_temp_dir() -> str:
    """Create and return a temporary directory for storing uploaded files."""
    try:
        temp_dir = Path(tempfile.gettempdir()) / 'nst_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temporary directory: {temp_dir}")
        return str(temp_dir)
    except Exception as e:
        logger.error(f"Error creating temp directory: {e}")
        raise

def save_uploaded_file(uploaded_file, save_dir: str) -> Optional[str]:
    """
    Save an uploaded file to the specified directory with a unique filename.
    
    Args:
        uploaded_file: The file uploaded via Streamlit's file_uploader
        save_dir: Directory to save the file to
        
    Returns:
        str: Path to the saved file, or None if there was an error
    """
    try:
        logger.info(f"Attempting to save uploaded file: {uploaded_file.name if uploaded_file else 'None'}")
        
        # Validate input
        if uploaded_file is None:
            raise ValueError("No file was uploaded")
            
        # Ensure save_dir exists
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        filename = Path(uploaded_file.name).name  # Get basename to prevent path traversal
        if not filename:
            raise ValueError("Invalid filename")
        
        # Create unique filename if needed
        file_path = save_path / filename
        counter = 1
        base, ext = os.path.splitext(filename)
        
        while file_path.exists():
            file_path = save_path / f"{base}_{counter}{ext}"
            counter += 1
            if counter > 100:  # Prevent infinite loops
                raise RuntimeError("Too many duplicate filenames")
        
        # Save the file
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise RuntimeError(f"Failed to save file: {e}")
        
        # Verify file was saved
        if not file_path.exists():
            raise RuntimeError("File was not saved correctly")
            
        file_path = str(file_path.resolve())
        logger.info(f"Successfully saved file to: {file_path}")
        st.toast(f"Saved: {filename}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error in save_uploaded_file: {e}", exc_info=True)
        st.error(f"Error saving file: {e}")
        return None

def main():
    # Main application code will go here
    try:
        # Create main columns
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
                if 'content_image' not in locals() or 'style_image' not in locals() or content_image is None or style_image is None:
                    st.error("Please upload both content and style images")
                else:
                    try:
                        with st.spinner("Processing images..."):
                            # Create output directory
                            output_dir = os.path.join(temp_dir, "output")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Save uploaded files
                            content_path = save_uploaded_file(content_image, temp_dir)
                            style_path = save_uploaded_file(style_image, temp_dir)
                            
                            if not content_path or not style_path:
                                raise Exception("Failed to save one or more uploaded files")
                            
                            # Prepare config with actual filenames
                            config = {
                                'content_img_name': os.path.basename(content_path),
                                'style_img_name': os.path.basename(style_path),
                                'content_images_dir': os.path.dirname(content_path),
                                'style_images_dir': os.path.dirname(style_path),
                                'output_img_dir': output_dir,
                                'img_format': (4, '.jpg'),
                                'height': height,
                                'content_weight': content_weight,
                                'style_weight': style_weight,
                                'tv_weight': tv_weight
                            }
                            
                            logger.info(f"Using config: {config}")
                            
                            # Initialize progress bar and status
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Store in session state for the callback
                            st.session_state.progress_bar = progress_bar
                            st.session_state.status_text = status_text
                            
                            # Run style transfer
                            status_text.text("Starting style transfer...")
                            
                            def progress_callback(progress, message):
                                progress_bar.progress(progress)
                                status_text.text(message)
                            
                            try:
                                start_time = time.time()
                                results_path = neural_style_transfer(config, progress_callback=progress_callback)
                                end_time = time.time()
                                
                                progress_bar.progress(100)
                                status_text.text(f"Style transfer completed in {end_time - start_time:.2f} seconds")
                                
                                # Display result
                                if results_path and os.path.exists(results_path):
                                    output_files = [f for f in os.listdir(results_path) if f.endswith('.jpg')]
                                    if output_files:
                                        output_files.sort()
                                        latest_output = os.path.join(results_path, output_files[-1])
                                        output_image = Image.open(latest_output)
                                        
                                        # Display in the output column
                                        col3.markdown("### Output Image")
                                        col3.image(output_image, use_column_width=True, caption="Stylized Output")
                                        
                                        # Add download button
                                        with open(latest_output, "rb") as file:
                                            col3.download_button(
                                                label="Download Image",
                                                data=file,
                                                file_name=f"stylized_{os.path.basename(content_path)}",
                                                mime="image/jpg"
                                            )
                                else:
                                    st.error("Style transfer completed but no output was generated.")
                                    
                            except Exception as e:
                                logger.error(f"Error during style transfer: {e}")
                                logger.error(traceback.format_exc())
                                st.error("An error occurred during style transfer:")
                                st.code(traceback.format_exc())
                                st.error(f"Error details: {str(e)}")
                            
                    except Exception as e:
                        logger.error(f"Error processing images: {e}")
                        logger.error(traceback.format_exc())
                        st.error(f"An error occurred while processing images: {str(e)}")
                        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An unexpected error occurred: {str(e)}")
        st.code(traceback.format_exc())
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            <p>Neural Style Transfer App | Created with ❤️ by Umang Hirani</p>
        </div>
    """, unsafe_allow_html=True)

def run_style_transfer(config, progress_callback=None):
    """Run the style transfer with progress updates."""
    try:
        if progress_callback:
            progress_callback(0, "Starting style transfer...")
            
        start_time = time.time()
        results_path = neural_style_transfer(config, progress_callback=progress_callback)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Style transfer completed in {duration:.2f} seconds")
        
        if progress_callback:
            progress_callback(100, f"Completed in {duration:.1f} seconds")
            
        return results_path
        
    except Exception as e:
        logger.error(f"Error during style transfer: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        raise

def main_wrapper():
    """Wrapper around main to catch and log all exceptions."""
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in application: {e}", exc_info=True)
        st.error("A critical error occurred. Please check the logs for more details.")
        st.stop()

def setup_signal_handlers():
    """Set up signal handlers if running in a main thread and not in Streamlit Cloud."""
    try:
        import threading
        if (threading.current_thread() is threading.main_thread() and 
            'STREAMLIT_SERVER_RUNNING' not in os.environ and 
            os.name != 'nt'):
            
            import signal
            def handle_sigint(signum, frame):
                logger.info("Received interrupt signal. Exiting...")
                sys.exit(0)
                
            signal.signal(signal.SIGINT, handle_sigint)
            logger.info("Signal handlers set up")
            return True
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")
    return False

if __name__ == "__main__":
    # Set up signal handlers if possible
    setup_signal_handlers()
    
    # Run the application
    try:
        main_wrapper()
    except Exception as e:
        logger.critical(f"Fatal error in application: {e}", exc_info=True)
        sys.exit(1)
