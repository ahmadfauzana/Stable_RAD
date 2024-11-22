import torch
import torch.nn.functional as F
from skimage import filters

def normalize(anomaly_map):
    anomaly_map_min = anomaly_map.view(anomaly_map.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    anomaly_map_max = anomaly_map.view(anomaly_map.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
    anomaly_map = (anomaly_map - anomaly_map_min) / (anomaly_map_max - anomaly_map_min + 1e-8)
    return anomaly_map

# Apply adaptive thresholding (e.g., Otsu's method) to compute Image-wise AUROC
def adaptive_threshold(anomaly_map, method='otsu'):
    if method == 'otsu':
        threshold_value = filters.threshold_otsu(anomaly_map.cpu().numpy())
    elif method == "quantile":
        percentile = 90
        threshold_value = torch.quantile(anomaly_map, percentile / 100.0)
    else:
        threshold_value = anomaly_map.mean().item()
    return (anomaly_map > threshold_value).float()

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

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor using the given mean and std.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean