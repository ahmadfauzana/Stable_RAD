import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as Func

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_anomaly_score(original, reconstructed):
    """Calculate the mean squared error between the original and reconstructed images."""
    return torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])

def compute_anomaly_map(original, reconstructed):
    """Calculate the absolute difference between original and reconstructed images and reduce to a single channel."""
    anomaly_map = torch.abs(original - reconstructed)
    anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)
    anomaly_map = F.interpolate(anomaly_map, size=original.shape[2:], mode='bilinear', align_corners=False)
    return anomaly_map

def highlight_anomaly(original, reconstructed, anomaly_map, threshold=0.1):
    """Highlight anomalous regions in the reconstructed images based on the anomaly map."""
    # Ensure that all tensors have the same number of channels (3 channels for RGB images)
    if original.shape[1] != 3:
        original = original.repeat(1, 3, 1, 1)  # Repeat grayscale to RGB if needed
    if reconstructed.shape[1] != 3:
        reconstructed = reconstructed.repeat(1, 3, 1, 1)  # Ensure 3 channels in reconstruction
    
    mask = anomaly_map > threshold  # Threshold the difference for significant anomalies
    mask = mask.float()
    
    # Ensure the mask has the correct number of channels for RGB (3 channels)
    if mask.shape[1] == 1:  # If it's a single-channel mask, replicate to 3 channels
        anomaly_highlight = torch.cat([mask, mask, mask], dim=1)
    else:
        anomaly_highlight = mask  # Use the mask if it already has 3 channels
    
    # Overlay the highlighted anomaly on the reconstructed image
    highlighted_anomaly = original * (1 - anomaly_highlight) + reconstructed * anomaly_highlight
    return highlighted_anomaly

def post_process_reconstruction(recon_image, anomaly_map):
    """
    Post-process the reconstructed image by blending the anomalous regions.
    A basic denoising technique can be applied here.
    """
    threshold = 0.5  # Define a threshold to consider an area anomalous
    mask = (anomaly_map > threshold).float()
    
    # Apply a basic blurring filter to smooth anomalous regions
    blurred_recon = Func.gaussian_blur(recon_image, kernel_size=5)
    
    # Blend original and blurred images, keeping the normal parts intact
    recon_image = recon_image * (1 - mask) + blurred_recon * mask
    return recon_image

def loss_function(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()

    # Ensure that `a` and `b` are tensors
    assert isinstance(a, torch.Tensor), f"Expected tensor, got {type(a)}"
    assert isinstance(b, torch.Tensor), f"Expected tensor, got {type(b)}"

    # Compute the loss
    loss = sum(
        0.1 * mse_loss(a_item, b_item) + torch.mean(1 - cos_loss(a_item.view(a_item.shape[0], -1), b_item.view(b_item.shape[0], -1)))
        for a_item, b_item in zip(a, b)
    )
    return loss

def create_directory_structure(output_dir, phase, ckpt_file, item_list):
    """
    Creates the directory structure and returns the path to store the visualization.
    
    Args:
        output_dir (str): The main directory for storing output.
        phase (str): Can be 'train' or 'test', indicating the phase.
        item_list (list): List of item names for which subfolders will be created.
        epoch (int): The current epoch number.

    Returns:
        dict: A dictionary with paths where images for each item should be saved.
    """
    # Create the main output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the subfolder for phase (train/test)
    phase_dir = os.path.join(output_dir, phase)
    if not os.path.exists(phase_dir):
        os.makedirs(phase_dir)

    if os.path.exists(ckpt_file):
        ckpt_dir = os.path.join(phase_dir, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        ckpt_dir = os.path.join(phase_dir, "zero")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    
    output_dirs = {}
    for item in item_list:
        item_dir = os.path.join(ckpt_dir, item)
        if not os.path.exists(item_dir):
            os.makedirs(item_dir)
        
        output_dirs[item] = item_dir
    
    return output_dirs

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor using the given mean and std.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def get_args():
    parser = argparse.ArgumentParser(description="Anomaly Detection Configuration")
    
    # CUDA settings
    parser.add_argument('--cuda_device', type=str, default='0', help="CUDA device id")
    parser.add_argument('--allow_kmp_duplication', type=bool, default=True, help="Allow KMP duplication (for parallelism issues)")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--image_size', type=int, default=256, help="Size of the input image")
    parser.add_argument('--threshold', type=float, default=0.1, help="Number of threshold")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha number")
    
    # Normalization
    parser.add_argument('--mean', nargs='+', type=float, default=[0.5, 0.5, 0.5], help="Mean normalization for training")
    parser.add_argument('--std', nargs='+', type=float, default=[0.5, 0.5, 0.5], help="Standard deviation normalization for training")
    
    # Paths
    parser.add_argument('--root_path', type=str, default="D:\\Fauzan\\Study PhD\\Research\\Stable_RAD\\dataset\\", help="Root path for dataset")
    parser.add_argument('--ckpt_path', type=str, default="D:\\Fauzan\\Study PhD\\Research\\Stable_RAD\\checkpoint\\", help="Checkpoint path")
    parser.add_argument('--save_path', type=str, default="D:\\Fauzan\\Study PhD\\Research\\Stable_RAD\\features\\", help="Save image features path")
    parser.add_argument('--output_path', type=str, default="D:\\Fauzan\\Study PhD\\Research\\Stable_RAD\\output\\", help="Output visualization path")
    parser.add_argument('--score_path', type=str, default="D:\\Fauzan\\Study PhD\\Research\\Stable_RAD\\", help="Path to save anomaly scores")
    
    # Phase and DMAD (train or test)
    parser.add_argument('--phase', type=str, choices=['train', 'test', 'retrieval', 'inf_train', 'inf_test', 'inf_retrieval'], default='train', help="Phase of the process: retrieval, train, or test")
    
    # Seed and item settings
    parser.add_argument('--seed', type=int, default=111, help="Random seed for reproducibility")
    parser.add_argument('--ifgeom', nargs='+', default=['screw', 'carpet', 'metal_nut'], help="Geometric anomalies to consider")
    parser.add_argument('--item_list', nargs='+', default=['bottle', 'capsule', 'cable', 'screw', 'pill', 'carpet', 'hazelnut', 'leather', 'grid', 'transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood'], help="List of items")

    args = parser.parse_args()
    return args