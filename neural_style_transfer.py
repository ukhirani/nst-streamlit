import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results
from utils.utils import IMAGENET_MEAN_255

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config, progress_callback=None):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add error handling for image loading
    try:
        content_img = utils.prepare_img(content_img_path, config['height'], device)
        style_img = utils.prepare_img(style_img_path, config['height'], device)
    except Exception as e:
        print(f"Error loading images: {e}")
        return None

    if config['init_method'] == 'random':
        # Reduce noise variance to prevent instability
        gaussian_noise_img = np.random.normal(loc=0, scale=50., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img.clone()
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # Reduced iteration counts to prevent instability
    num_of_iterations = {
        "lbfgs": 1000,  # Reduced from 2000
        "adam": 500,    # Reduced from 1000
    }

    #
    # Start of optimization procedure
    #
    if config['optimizer'] == 'adam':
        # Reduced learning rate for stability
        optimizer = Adam((optimizing_img,), lr=1e0)  # Reduced from 5e0
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            
            # Check for nan/inf losses and abort if found
            if (torch.isnan(total_loss) or torch.isinf(total_loss) or
                torch.isnan(content_loss) or torch.isinf(content_loss) or
                torch.isnan(style_loss) or torch.isinf(style_loss) or
                torch.isnan(tv_loss) or torch.isinf(tv_loss)):
                print(f"[WARNING] NaN or Inf detected in losses at iteration {cnt}, aborting optimization.")
                break
                
            with torch.no_grad():
                # Clamp image values to valid range
                mean = torch.tensor(IMAGENET_MEAN_255, device=optimizing_img.device).view(1, 3, 1, 1)
                optimizing_img.data = torch.clamp(optimizing_img.data, -mean, 255 - mean)
                
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                
                # Save intermediate results
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
                
                # Call progress callback with proper error handling
                if progress_callback:
                    try:
                        should_continue = progress_callback(cnt, num_of_iterations[config['optimizer']], optimizing_img)
                        if should_continue is False:
                            print("Optimization stopped by user.")
                            break
                    except Exception as callback_error:
                        print(f"Error in progress callback: {callback_error}")
                        # Continue optimization even if callback fails
                        
    elif config['optimizer'] == 'lbfgs':
        # Reduce max_iter for stability
        optimizer = LBFGS((optimizing_img,), max_iter=20, line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
                
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            
            if total_loss.requires_grad:
                total_loss.backward()
                
            # Check for nan/inf losses
            if (torch.isnan(total_loss) or torch.isinf(total_loss) or
                torch.isnan(content_loss) or torch.isinf(content_loss) or
                torch.isnan(style_loss) or torch.isinf(style_loss) or
                torch.isnan(tv_loss) or torch.isinf(tv_loss)):
                print(f"[WARNING] NaN or Inf detected in losses at iteration {cnt}, aborting optimization.")
                return total_loss
                
            with torch.no_grad():
                # Clamp image values
                mean = torch.tensor(IMAGENET_MEAN_255, device=optimizing_img.device).view(1, 3, 1, 1)
                optimizing_img.data = torch.clamp(optimizing_img.data, -mean, 255 - mean)
                
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                
                # Save intermediate results
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
                
                # Call progress callback with proper error handling
                if progress_callback:
                    try:
                        should_continue = progress_callback(cnt, num_of_iterations[config['optimizer']], optimizing_img)
                        if should_continue is False:
                            print("Optimization stopped by user.")
                            return total_loss
                    except Exception as callback_error:
                        print(f"Error in progress callback: {callback_error}")

            cnt += 1
            return total_loss

        # Run L-BFGS for specified number of iterations
        for _ in range(num_of_iterations[config['optimizer']]):
            optimizer.step(closure)
            if cnt >= num_of_iterations[config['optimizer']]:
                break

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='golden_gate.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e6)  # Reduced from 3e5
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs'], default='lbfgs')  # Only L-BFGS is supported
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    # Ensure optimizer is always L-BFGS regardless of input
    optimization_config['optimizer'] = 'lbfgs'
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)