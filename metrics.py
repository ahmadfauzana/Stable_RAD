import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F

def ssim(original_patch, reconstructed_patch, C1=0.01**2, C2=0.03**2):
    """
    Calculate SSIM between two patches.
    """
    original_edge = original_patch - original_patch.mean()
    reconstructed_edge = reconstructed_patch - reconstructed_patch.mean()

    mu_x = original_patch.mean()
    mu_y = reconstructed_patch.mean()
    sigma_x = ((original_patch - mu_x) ** 2).mean()
    sigma_y = ((reconstructed_patch - mu_y) ** 2).mean()
    sigma_xy = ((original_patch - mu_x) * (reconstructed_patch - mu_y)).mean()

    edge_score = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    # Apply edge-based weighting if necessary
    edge_weight = (original_edge * reconstructed_edge).mean()
    ssim_score = edge_score * (1 + edge_weight)  # Higher score if edges align better
    return ssim_score

def compute_anomaly_map(original, reconstructed, patch_size=16, smooth_factor=3, similarity_threshold=0.8): 
    """
    Calculate SSIM-based anomaly map using a patch-based approach with shape matching.
    """
    # Ensure shape consistency
    original, reconstructed = match_shapes(original, reconstructed)
    
    anomaly_map = torch.zeros((original.shape[0], 1, original.shape[2] // patch_size, original.shape[3] // patch_size))

    original_patches = original.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    reconstructed_patches = reconstructed.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    for i in range(original_patches.shape[2]):
        for j in range(original_patches.shape[3]):
            orig_patch = original_patches[:, :, i, j, :, :]
            recon_patch = reconstructed_patches[:, :, i, j, :, :]
            ssim_score = ssim(orig_patch, recon_patch)

            # Dynamic threshold based on local SSIM statistics
            if ssim_score >= similarity_threshold:
                ssim_score = 1
            else:
                ssim_score = 0

            anomaly_map[:, :, i, j] = 1 - ssim_score

    # Apply smoothing and interpolation
    anomaly_map = F.interpolate(anomaly_map, size=original.shape[2:], mode='bilinear', align_corners=False)
    anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=3, stride=1, padding=1)  # Adaptive smoothing

    return anomaly_map

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

def compute_anomaly_score(original, reconstructed, patch_size=16, similarity_threshold=0.8):
    """
    Calculate an overall anomaly score based on the SSIM-based anomaly map.
    """
    anomaly_map = compute_anomaly_map(original, reconstructed, patch_size, similarity_threshold=similarity_threshold)
    return anomaly_map.mean(dim=[1, 2, 3])

def find_similar_images(query_feature, reference_features, similarity_threshold=0.8):
    """
    Finds the most similar feature vector in the reference_features based on cosine similarity.

    Args:
    - query_feature (numpy array): The feature vector from the test image (flattened).
    - reference_features (numpy array): A set of feature vectors (flattened) to compare against.

    Returns:
    - idx (int): Index of the most similar feature in the reference_features.
    """
    # Ensure the query feature is 2D
    query_feature = query_feature.flatten().reshape(1, -1)  # Shape becomes [1, 4096]

    # Flatten the reference features (N, C, H, W) to (N, C*H*W)
    reference_features_flat = reference_features.reshape(reference_features.shape[0], -1)  # Shape becomes [N, flattened_size]

    # Compute cosine similarity between the query feature and all reference features
    similarities = cosine_similarity(query_feature, reference_features_flat)

    # Apply threshold to filter out low-similarity matches (optional)
    valid_indices = np.where(similarities >= similarity_threshold)[1]

    # Find the index of the most similar reference feature
    if valid_indices.size > 0:
        idx = valid_indices[np.argmax(similarities[0, valid_indices])]
    else:
        idx = -1  # or any indicator for no valid match

    return idx