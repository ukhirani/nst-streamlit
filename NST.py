import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from models.definitions.vgg19 import Vgg19
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path, target_shape=None):
    """Load an image from disk with error handling and validation."""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image file not found: {img_path}')
            
        # Read image
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'Failed to load image: {img_path}. The file may be corrupted or in an unsupported format.')
            
        # Convert color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if target shape is provided
        if target_shape is not None:
            if isinstance(target_shape, int):
                current_height, current_width = img.shape[0], img.shape[1]
                if current_height == 0 or current_width == 0:
                    raise ValueError(f'Invalid image dimensions: {current_width}x{current_height}')
                    
                ratio = target_shape / current_height
                target_width = int(current_width * ratio)
                
                if target_width <= 0 or target_shape <= 0:
                    raise ValueError(f'Invalid target dimensions: {target_width}x{target_shape}')
                    
                img = cv2.resize(img, (target_width, target_shape), interpolation=cv2.INTER_CUBIC)
            else:
                img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert to float32 and normalize
        img = img.astype(np.float32)
        img /= 255.0
        
        return img
        
    except Exception as e:
        logger.error(f'Error loading image {img_path}: {str(e)}')
        raise

def prepare_img(img_path, target_shape, device):
    """Prepare image for processing with validation."""
    try:
        logger.info(f'Preparing image: {img_path} (target height: {target_shape})')
        
        # Load and validate image
        img = load_image(img_path, target_shape=target_shape)
        
        if img is None or img.size == 0:
            raise ValueError('Loaded image is empty')
            
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)
        img_tensor = img_tensor.unsqueeze(0)
        
        logger.info(f'Image prepared successfully. Shape: {img_tensor.shape}, Device: {img_tensor.device}')
        return img_tensor
        
    except Exception as e:
        logger.error(f'Error preparing image {img_path}: {str(e)}')
        raise

def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])                   # convert RGB to BGR while writing

def generate_out_img_name(config):
    '''
    Generate a name for the output image.
    Example: 'c1-s1.jpg'
    where c1: content_img_name, and
          s1: style_img_name.
    '''
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    suffix = f'{config["img_format"][1]}'
    return prefix + suffix

def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations):
    '''
    Save the generated image.
    If saving_freq == -1, only the final output image will be saved.
    Else, intermediate images can be saved too.
    '''
    saving_freq = -1
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    if img_id == num_of_iterations-1 :
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
    

def prepare_model(device):
    '''
    Load VGG19 model into local cache.
    '''
    # Suppress deprecation warnings for torchvision models
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message='The parameter "pretrained" is deprecated')
    warnings.filterwarnings("ignore", category=UserWarning, message='Arguments other than a weight enum')
    
    # Initialize model with weights
    model = Vgg19(requires_grad=False, show_progress=True)
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

def gram_matrix(x, should_normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

def total_variation(y):
    '''
    Calculate total variation.
    '''
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Calculate content_loss, style_loss, and total_variation_loss.
    '''
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

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Performs a step in the tuning loop.
    (We are tuning only the pixels, not the weights.)
    '''
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def neural_style_transfer(config, progress_callback=None):
    '''
    The main Neural Style Transfer method.
    
    Args:
        config (dict): Configuration dictionary containing paths and parameters
        progress_callback (callable, optional): Callback function for progress updates
        
    Returns:
        str: Path to the directory containing the output images
    '''
    logger.info("Starting neural style transfer...")
    logger.info(f"Configuration: {config}")
    
    if progress_callback:
        progress_callback(0, "Initializing style transfer...")
    
    try:
        # Prepare paths
        content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
        style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
        out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
        dump_path = os.path.join(config['output_img_dir'], out_dir_name)
        
        print(f"Content image path: {content_img_path}")
        print(f"Style image path: {style_img_path}")
        print(f"Output directory: {dump_path}")
        
        # Create output directory
        os.makedirs(dump_path, exist_ok=True)
        print(f"Created output directory: {os.path.exists(dump_path)}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Prepare images
        print("Preparing content and style images...")
        content_img = prepare_img(content_img_path, config['height'], device)
        style_img = prepare_img(style_img_path, config['height'], device)
        print(f"Content image shape: {content_img.shape}")
        print(f"Style image shape: {style_img.shape}")
        
        # Initialize with content image
        init_img = content_img
        optimizing_img = Variable(init_img, requires_grad=True)
        
        # Prepare model
        print("Preparing VGG19 model...")
        neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
        print(f'Using VGG19 in the optimization procedure.')
        
        # Get feature maps
        print("Extracting feature maps...")
        content_img_set_of_feature_maps = neural_net(content_img)
        style_img_set_of_feature_maps = neural_net(style_img)
        
        # Get target representations
        target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
        target_representations = [target_content_representation, target_style_representation]
        
        # Set up optimization
        num_of_iterations = 1000
        print(f"Starting optimization with {num_of_iterations} iterations...")
        
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
                
            # Calculate losses
            total_loss, content_loss, style_loss, tv_loss = build_loss(
                neural_net, 
                optimizing_img, 
                target_representations, 
                content_feature_maps_index_name[0], 
                style_feature_maps_indices_names[0], 
                config
            )
            
            # Backpropagate
            if total_loss.requires_grad:
                total_loss.backward()
                
            # Log progress
            if cnt % 10 == 0:  # Print every 10 iterations
                print(f'Iteration: {cnt:03}, Total Loss: {total_loss.item():.2f}, ' 
                      f'Content: {config["content_weight"] * content_loss.item():.2f}, '
                      f'Style: {config["style_weight"] * style_loss.item():.2f}, '
                      f'TV: {config["tv_weight"] * tv_loss.item():.2f}')
                
                # Save intermediate results
                save_and_maybe_display(
                    optimizing_img, 
                    dump_path, 
                    config, 
                    cnt, 
                    num_of_iterations
                )
                
            cnt += 1
            return total_loss
            
        # Run optimization
        print("Starting optimization...")
        optimizer.step(closure)
        print("Optimization completed!")
        
        # Save final result
        save_and_maybe_display(
            optimizing_img, 
            dump_path, 
            config, 
            num_of_iterations-1,  # Save as final iteration
            num_of_iterations
        )
        
        print(f"Style transfer completed successfully! Results saved to: {dump_path}")
        return dump_path
        
    except Exception as e:
        import traceback
        print("Error in neural_style_transfer:")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    # Default configuration for when this file is run directly
    PATH = ''
    CONTENT_IMAGE = 'c1.jpg'
    STYLE_IMAGE = 's1.jpg'

    default_resource_dir = os.path.join(PATH, 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')

    optimization_config = {
        'content_img_name': CONTENT_IMAGE,
        'style_img_name': STYLE_IMAGE,
        'height': 400,
        'content_weight': 100000.0,
        'style_weight': 30000.0,
        'tv_weight': 1.0,
        'content_images_dir': content_images_dir,
        'style_images_dir': style_images_dir,
        'output_img_dir': output_img_dir,
        'img_format': img_format
    }

    results_path = neural_style_transfer(optimization_config)
