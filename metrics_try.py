import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import models
from skimage import filters

# Load pre-trained VGG model (features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have a pretrained VGG or similar model
vgg = models.vgg16(pretrained=True).features.to(device)
vgg.eval()

# Perceptual loss: Using VGG features for perceptual loss
def compute_perceptual_loss(original, reconstructed):
    original_features = vgg(original)
    reconstructed_features = vgg(reconstructed)
    perceptual_loss = F.mse_loss(original_features, reconstructed_features)
    return perceptual_loss

# Function to compute the anomaly map based on perceptual loss
def compute_anomaly_map_perceptual(original, reconstructed):
    perceptual_loss_map = compute_perceptual_loss(original, reconstructed)
    return perceptual_loss_map

# Apply adaptive thresholding (e.g., Otsu's method) to compute Image-wise AUROC
def adaptive_threshold(anomaly_map, method='otsu'):
    if method == 'otsu':
        threshold_value = filters.threshold_otsu(anomaly_map.cpu().numpy())
    else:
        threshold_value = anomaly_map.mean().item()
    return anomaly_map > threshold_value

# Compute AUROC (pixel-wise)
def compute_auroc(pred, ground_truth):
    return roc_auc_score(ground_truth.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

# Compute Image-wise AUROC using adaptive threshold
def compute_auroc_for_image_level(anomaly_map, ground_truth):
    thresholded_map = adaptive_threshold(anomaly_map)
    return compute_auroc(thresholded_map, ground_truth)

# Combine perceptual and GLASS anomaly detection methods
def compute_combined_anomaly_map(original, reconstructed, glass_anomaly_map):
    perceptual_loss_map = compute_anomaly_map_perceptual(original, reconstructed)
    combined_anomaly_map = (perceptual_loss_map + glass_anomaly_map) / 2  # Hybrid method
    return combined_anomaly_map

# GLASS method for anomaly detection (simplified)
def compute_glass_anomaly_map(original, reconstructed):
    # Example: Compute difference between global and local features (can be customized)
    global_features = F.adaptive_avg_pool2d(original, 1)
    local_features = F.adaptive_avg_pool2d(reconstructed, 1)
    glass_map = F.mse_loss(global_features, local_features, reduction='none')
    return glass_map

def compute_multi_scale_anomaly_map(original, reconstructed, scales=[3, 5, 7]):
    anomaly_maps = []
    for scale in scales:
        anomaly_map = compute_combined_anomaly_map(original, reconstructed, compute_glass_anomaly_map(original, reconstructed))
        anomaly_maps.append(anomaly_map)
    
    # Stack and reduce to [B, C, H, W]
    combined_anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    return combined_anomaly_map

def apply_gaussian_blur(input_tensor, kernel_size=5, sigma=1.5):
    # Ensure the input tensor has 4D shape [N, C, H, W]
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    channels = input_tensor.size(1)

    # Create the Gaussian kernel for each channel
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(input_tensor.device)

    # Apply convolution with groups equal to the number of channels
    blurred_tensor = F.conv2d(input_tensor, kernel, padding=kernel_size // 2, groups=channels)

    # Remove the batch dimension if it was added earlier
    if blurred_tensor.size(0) == 1:
        blurred_tensor = blurred_tensor.squeeze(0)

    return blurred_tensor

def gaussian_kernel(kernel_size=5, sigma=1.5, channels=1):
    # Create a 1D Gaussian kernel
    kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = torch.exp(-(kernel_1d ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize the kernel

    # Create a 2D Gaussian kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d).unsqueeze(0).unsqueeze(0)

    # Repeat the kernel for each channel
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # Shape: [channels, 1, kernel_size, kernel_size]

    return kernel_2d

# Adjusted smoothing function
def smooth_anomaly_map(anomaly_map, kernel_size=5, sigma=1.5):
    # If the anomaly map has an extra dimension, reduce it
    if anomaly_map.dim() == 5:
        anomaly_map = anomaly_map.mean(dim=0, keepdim=True)  # Aggregate across extra dimension
    
    # Ensure anomaly_map is now in shape [B, C, H, W]
    return apply_gaussian_blur(anomaly_map, kernel_size, sigma)

def apply_adaptive_threshold(anomaly_map, percentile=90):
    threshold = torch.quantile(anomaly_map, percentile / 100.0)
    return (anomaly_map > threshold).float()

def compute_anomaly_score(original, reconstructed, patch_size=16, similarity_threshold=0.85):
    anomaly_map = compute_anomaly_map_perceptual(original, reconstructed, patch_size, similarity_threshold=similarity_threshold)
    return anomaly_map.mean(dim=[1, 2, 3])  # Summing or averaging anomaly scores across all patches

def match_shapes(original, reconstructed):
    """Ensure the original and reconstructed images have the same dimensions."""
    if original.shape[2:] != reconstructed.shape[2:]:
        # Calculate padding sizes for height and width
        diff_h = original.shape[2] - reconstructed.shape[2]
        diff_w = original.shape[3] - reconstructed.shape[3]
        
        # Apply padding to match dimensions
        pad_h = max(diff_h, 0)
        pad_w = max(diff_w, 0)
        reconstructed = F.pad(reconstructed, (0, pad_w, 0, pad_h))
        
    return original, reconstructed

def find_similar_images(query_feature, reference_features, similarity_threshold=0.8):
    query_feature = query_feature.flatten().reshape(1, -1)
    reference_features_flat = reference_features.reshape(reference_features.shape[0], -1)
    similarities = cosine_similarity(query_feature, reference_features_flat)
    
    best_match = np.argmax(similarities)
    if similarities[0, best_match] < similarity_threshold:
        # Revert to using the closest match even if it’s below threshold
        return best_match
    return best_match
