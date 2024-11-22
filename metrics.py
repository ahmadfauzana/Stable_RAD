import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from preprocess import adaptive_threshold

def compute_anomaly_map(inputs, reconstructions, encoder, method='absolute'):
    """
    Compute pixel-wise anomaly maps.

    Args:
        inputs (torch.Tensor): Original input images [B, C, H, W].
        reconstructions (torch.Tensor): Reconstructed images [B, C, H, W].
        method (str): Distance metric to use ('L1', 'L2', or 'SSIM').
        normalize (bool): Whether to normalize the anomaly map.

    Returns:
        torch.Tensor: Pixel-wise anomaly map [B, H, W].
    """
   
    if method == 'absolute':
        anomaly_map = compute_absolute_anomaly_map(inputs, reconstructions, encoder)
    else:
        raise ValueError(f"Unknown method: {method}")

    return anomaly_map

def compute_anomaly_score(anomaly_map, original, reconstructed, method='mean'):
    """
    Compute image-wise anomaly score from the pixel-wise anomaly map.

    Args:
        anomaly_map (torch.Tensor): Pixel-wise anomaly map [B, H, W].
        method (str): Aggregation method ('mean', 'max', or 'weighted').

    Returns:
        torch.Tensor: Image-wise anomaly scores [B].
    """
    if method == 'mean':
        # Average anomaly score
        anomaly_score = anomaly_map.mean(dim=(-1, -2))
    elif method == 'max':
        # Get the maximum along the last two dimensions (height and width)
        anomaly_score = anomaly_map.max(dim=-1)[0].max(dim=-1)[0]
    elif method == 'weighted':
        # Weighted score emphasizing high-anomaly regions
        weights = anomaly_map / (anomaly_map.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        anomaly_score = (weights * anomaly_map).sum(dim=(-1, -2))
    elif method == 'mse':
        anomaly_score = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])
    elif method == 'spatial':
        anomaly_score = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    return anomaly_score

def compute_absolute_anomaly_map(original, reconstructed, encoder, 
                                 combine_weights=(0.8, 0.2), normalize=False):
    """
    Calculate combined anomaly map using enhanced feature and pixel differences.

    Args:
        original (torch.Tensor): Original input images [batch, channels, height, width].
        reconstructed (torch.Tensor): Reconstructed images [batch, channels, height, width].
        encoder (nn.Module): Encoder to extract feature-level latent distributions.
        combine_weights (tuple): Weights for combining feature and pixel anomaly maps.
        normalize (bool): Whether to normalize anomaly maps (default: True).

    Returns:
        torch.Tensor: Optimized combined anomaly map [batch, 1, height, width].
    """
    def resize_to_match(source, target):
        """Utility to resize the source tensor to match the target's spatial dimensions."""
        return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)

    # Feature-level anomaly using cosine similarity with multi-scale features
    orig_features = F.normalize(encoder(original).latent_dist.mean, dim=1)
    recon_features = F.normalize(encoder(reconstructed).latent_dist.mean, dim=1)
    feature_anomaly_map = 1 - F.cosine_similarity(orig_features, recon_features, dim=1, eps=1e-8).unsqueeze(1)
    feature_anomaly_map = resize_to_match(feature_anomaly_map, original)

    # Pixel-level anomaly map with amplified error
    pixel_anomaly_map = torch.pow(torch.abs(original - reconstructed), 3).mean(dim=1, keepdim=True)

    # Weighted combination with improved ratios
    w1, w2 = combine_weights
    combined_anomaly_map = torch.sqrt(w1 * feature_anomaly_map ** 2 + w2 * pixel_anomaly_map ** 2)

    return combined_anomaly_map

# Compute AUROC (pixel-wise)
def compute_pixel_auroc(pred, ground_truth):
    return roc_auc_score(ground_truth.cpu().numpy().astype(int), pred.cpu().numpy())

# Compute Image-wise AUROC using adaptive threshold
def compute_image_auroc(anomaly_map, ground_truth):
    thresholded_map = adaptive_threshold(anomaly_map)
    return compute_pixel_auroc(thresholded_map, ground_truth)

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