import streamlit as st
from PIL import Image
import numpy as np
import os
import tempfile
import shutil
from NST import neural_style_transfer, prepare_img, load_image
import torch
import cv2

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

# Main layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### Content Image")
    content_image = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'], key='content_image')
    if content_image is not None:
        image = Image.open(content_image)
        st.image(image, use_container_width=True, caption="Content Image")

with col2:
    st.markdown("### Style Image")
    style_image = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'], key='style_image')
    if style_image is not None:
        image = Image.open(style_image)
        st.image(image, use_container_width=True, caption="Style Image")

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
            with st.spinner("Applying style transfer... This may take a few minutes..."):
                try:
                    # Save uploaded files
                    content_path = save_uploaded_file(content_image, "content.jpg")
                    style_path = save_uploaded_file(style_image, "style.jpg")
                    
                    # Create output directory
                    output_dir = os.path.join(temp_dir, "output")
                    os.makedirs(output_dir, exist_ok=True)
                    
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
                    
                    # Run style transfer with progress
                    with st.spinner("Running style transfer... This may take a few minutes..."):
                        results_path = neural_style_transfer(config)
                    
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
                                    btn = st.download_button(
                                        label="Download Image",
                                        data=file,
                                        file_name=f"stylized_{os.path.basename(content_image.name)}",
                                        mime="image/jpg"
                                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    # Display output image if it exists
    if 'output_image' in locals():
        st.markdown("### Output Image")
        st.image(output_image, use_container_width=True, caption="Stylized Output")

# Add some spacing  
st.markdown("<br><br>", unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        <p>Neural Style Transfer App | Created with ❤️ by Umang Hirani</p>
    </div>
""", unsafe_allow_html=True)



