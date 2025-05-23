import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import os
from models.definitions.vgg19 import Vgg19
from torch.optim import LBFGS
from torch.autograd import Variable
from PIL import Image
import io

# Constants
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Helper functions
def load_image(uploaded_file, target_shape=None):
    if uploaded_file is None:
        return None
    
    # Read image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    
    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img, target_shape, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    img = transform(img).to(device).unsqueeze(0)
    return img

def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]
    
    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)
    
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)
    
    tv_loss = total_variation(optimizing_img)
    
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss
    
    return total_loss, content_loss, style_loss, tv_loss

def process_image(optimizing_img):
    with torch.no_grad():
        out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
        out_img = np.moveaxis(out_img, 0, 2)
        out_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        out_img = np.clip(out_img, 0, 255).astype('uint8')
        return out_img

def main():
    st.title('Neural Style Transfer')
    st.write('Upload your content and style images to create a unique artistic composition!')

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        content_file = st.file_uploader('Upload Content Image', type=['png', 'jpg', 'jpeg'])
        if content_file:
            st.image(content_file, caption='Content Image', use_container_width=True)
    
    with col2:
        style_file = st.file_uploader('Upload Style Image', type=['png', 'jpg', 'jpeg'])
        if style_file:
            st.image(style_file, caption='Style Image', use_container_width=True)

    # Parameters
    st.sidebar.header('Parameters')
    height = st.sidebar.slider('Image Height', 100, 512, 400)
    content_weight = st.sidebar.slider('Content Weight', 1000.0, 1000000.0, 100000.0)
    style_weight = st.sidebar.slider('Style Weight', 1000.0, 1000000.0, 30000.0)
    tv_weight = st.sidebar.slider('Total Variation Weight', 0.0, 10.0, 1.0)
    num_iterations = st.sidebar.slider('Number of Iterations', 100, 1000, 300)

    # Process button
    if st.button('Apply Style Transfer') and content_file and style_file:
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_image = st.empty()
        
        # Configuration
        config = {
            'height': height,
            'content_weight': content_weight,
            'style_weight': style_weight,
            'tv_weight': tv_weight
        }
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f'Using device: {device}')
        
        # Load and prepare images
        content_img = load_image(content_file, config['height'])
        style_img = load_image(style_file, config['height'])
        
        content_img_tensor = prepare_img(content_img, config['height'], device)
        style_img_tensor = prepare_img(style_img, config['height'], device)
        
        # Initialize model
        model = Vgg19(requires_grad=False, show_progress=False)
        model = model.to(device).eval()
        
        # Get content and style features
        content_features = model(content_img_tensor)
        style_features = model(style_img_tensor)
        
        target_content_representation = content_features[model.content_feature_maps_index].squeeze(axis=0)
        target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_features) if cnt in model.style_feature_maps_indices]
        target_representations = [target_content_representation, target_style_representation]
        
        # Initialize optimizing image
        optimizing_img = Variable(content_img_tensor.clone(), requires_grad=True)
        
        # Setup optimizer
        optimizer = LBFGS([optimizing_img])
        
        # Optimization loop
        n_iter = [0]
        
        while n_iter[0] <= num_iterations:
            def closure():
                optimizer.zero_grad()
                total_loss, content_loss, style_loss, tv_loss = build_loss(
                    model, optimizing_img, target_representations,
                    model.content_feature_maps_index, model.style_feature_maps_indices,
                    config
                )
                
                if total_loss.requires_grad:
                    total_loss.backward()
                
                with torch.no_grad():
                    if n_iter[0] % 10 == 0 or n_iter[0] == num_iterations:
                        progress = n_iter[0] / num_iterations
                        progress_bar.progress(progress)
                        
                        status_text.text(
                            f'Iteration: {n_iter[0]}/{num_iterations}, '
                            f'Total Loss: {total_loss.item():.2f}, '
                            f'Content Loss: {content_loss.item():.2f}, '
                            f'Style Loss: {style_loss.item():.2f}, '
                            f'TV Loss: {tv_loss.item():.2f}'
                        )
                        
                        # Update output image
                        current_img = process_image(optimizing_img)
                        output_image.image(current_img, caption=f'Output - Iteration {n_iter[0]}', use_container_width=True)
                
                n_iter[0] += 1
                return total_loss
            
            optimizer.step(closure)
            
            if n_iter[0] >= num_iterations:
                break
        
        # Final output
        final_img = process_image(optimizing_img)
        st.success('Style transfer completed!')
        
        # Download button
        buf = io.BytesIO()
        Image.fromarray(final_img).save(buf, format='PNG')
        st.download_button(
            label='Download Result',
            data=buf.getvalue(),
            file_name='style_transfer_result.png',
            mime='image/png'
        )

if __name__ == '__main__':
    main()