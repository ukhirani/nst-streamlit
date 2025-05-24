import os
import io
import torch
import streamlit as st
from PIL import Image
import numpy as np
from NST import neural_style_transfer, load_image, save_image, prepare_img
from torch.optim import LBFGS, Adam
from torch.autograd import Variable
from PIL import Image
import io

# Constants
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]
SUPPORTED_OPTIMIZERS = ['adam', 'lbfgs']
NUM_ITERATIONS = {
    'adam': 1000,
    'lbfgs': 100
}

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data/content-images', 'data/style-images', 'data/output-images']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def process_uploaded_image(uploaded_file, save_dir):
    """Process an uploaded file and save it"""
    if uploaded_file is not None:
        # Read and convert to RGB
        img = Image.open(uploaded_file).convert('RGB')
        
        # Save the image
        save_path = os.path.join(save_dir, uploaded_file.name)
        img.save(save_path)
        
        return save_path, img
    return None, None

def main():
    st.set_page_config(page_title='Neural Style Transfer', layout='wide')
    st.title('Neural Style Transfer')
    st.write('Transform your images using Neural Style Transfer!')

    # Setup directories
    content_dir, style_dir, output_dir = setup_directories()

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        content_file = st.file_uploader('Upload Content Image', type=['png', 'jpg', 'jpeg'])
        if content_file:
            content_image = Image.open(content_file).convert('RGB')
            st.image(content_image, caption='Content Image', use_container_width=True)
            content_path = os.path.join('data/content-images', content_file.name)
            content_image.save(content_path)
    
    with col2:
        style_file = st.file_uploader('Upload Style Image', type=['png', 'jpg', 'jpeg'])
        if style_file:
            style_image = Image.open(style_file).convert('RGB')
            st.image(style_image, caption='Style Image', use_container_width=True)
            style_path = os.path.join('data/style-images', style_file.name)
            style_image.save(style_path)

    # Advanced parameters in sidebar
    st.sidebar.title('Parameters')
    
    # Basic parameters
    height = st.sidebar.slider('Image Height', 100, 512, 400, help='Higher values give better quality but take longer')
    
    # Advanced parameters in expander
    with st.sidebar.expander('Advanced Settings'):
        optimizer = st.selectbox('Optimizer', SUPPORTED_OPTIMIZERS, 
                               help='LBFGS usually gives better results but is slower')
        init_method = st.selectbox('Initialization Method', 
                                 ['content', 'style', 'random'],
                                 help='How to initialize the optimization process')
        
        # Adjusted parameter ranges for better stability
        content_weight = st.slider('Content Weight', 1e3, 1e5, 1e4, 
                                 format='%e', help='Higher values preserve more content')
        style_weight = st.slider('Style Weight', 1e0, 1e4, 1e3, 
                                format='%e', help='Higher values apply more style')
        tv_weight = st.slider('Total Variation Weight', 0.1, 2.0, 1.0,
                            help='Controls image smoothness')
        
        # Add learning rate control for Adam
        if optimizer == 'adam':
            learning_rate = st.slider('Learning Rate', 1e-3, 1e0, 1e-1, 
                                    format='%e', help='Step size for optimization')

    # Process button
    if st.button('Apply Style Transfer', type='primary', use_container_width=True):
        if not content_file or not style_file:
            st.error('Please upload both content and style images first!')
            return

        # Show progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_image = st.empty()
        
        # Configuration
        config = {
            'content_img_name': content_file.name,
            'style_img_name': style_file.name,
            'height': height,
            'content_weight': content_weight,
            'style_weight': style_weight,
            'tv_weight': tv_weight,
            'optimizer': optimizer,
            'init_method': init_method,
            'content_images_dir': 'data/content-images',
            'style_images_dir': 'data/style-images',
            'output_img_dir': 'data/output-images',
            'img_format': (4, '.jpg')
        }
        
        try:
            with st.spinner('Initializing Neural Style Transfer...'):
                # Device setup
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f'Using device: {device}')
                
                # Load images
                content_img = prepare_img(os.path.join(config['content_images_dir'], content_file.name), 
                                        height, device)
                style_img = prepare_img(os.path.join(config['style_images_dir'], style_file.name), 
                                       height, device)
                
                # Setup NST config
                config = {
                    'content_img_path': content_path,
                    'style_img_path': style_path,
                    'height': height,
                    'content_weight': content_weight,
                    'style_weight': style_weight,
                    'tv_weight': tv_weight,
                    'optimizer': optimizer_name,
                    'init_method': init_method,
                    'saving_freq': 10,
                    'device': device
                }
                
                if optimizer_name == 'adam':
                    config['lr'] = learning_rate
                
                # Run style transfer
                output_images = neural_style_transfer(config)
                
                # Display progress
                for i, img in enumerate(output_images):
                    if i % 10 == 0:  # Update every 10 iterations
                        progress = i / NUM_ITERATIONS[optimizer_name]
                        progress_bar.progress(progress)
                        output_image.image(img, caption=f'Output - Iteration {i}', use_container_width=True)
                
                # Get final result (last image)
                final_img = output_images[-1]
                st.success('Style transfer completed successfully!')
                
                # Save result
                out_dir = os.path.join('data/output-images', 
                                      f'combined_{content_file.name.split(".")[0]}_{style_file.name.split(".")[0]}')
                os.makedirs(out_dir, exist_ok=True)
                
                final_path = os.path.join(out_dir, 'final.jpg')
                save_image(final_img, final_path)
                
                # Show download button
                with open(final_path, 'rb') as f:
                    st.download_button(
                        label='Download Result',
                        data=f.read(),
                        file_name='style_transfer_result.jpg',
                        mime='image/jpeg',
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f'An error occurred during style transfer: {str(e)}')
            raise e

if __name__ == '__main__':
    main()