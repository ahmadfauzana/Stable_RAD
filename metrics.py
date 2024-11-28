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
    else:
        raise ValueError(f"Unknown method: {method}")
    return anomaly_score

def compute_absolute_anomaly_map(original, reconstructed, encoder, 
                                 combine_weights=(0.7, 0.3)):
    
    def compute_feature_anomaly_map(original, reconstructed, encoder, alpha=2.0, multi_scale=False):
        """
        Compute enhanced feature-level anomaly map.

        Args:
            original (torch.Tensor): Original input images [batch, channels, height, width].
            reconstructed (torch.Tensor): Reconstructed images [batch, channels, height, width].
            encoder (nn.Module): Encoder to extract latent distributions.
            alpha (float): Scaling factor to emphasize high-magnitude anomalies.
            multi_scale (bool): Whether to use multi-scale latent feature extraction.

        Returns:
            torch.Tensor: Enhanced feature anomaly map [batch, 1, height, width].
        """
    
        def resize_to_match(source, target):
            """Resize source tensor to match target spatial dimensions."""
            return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)

        # Single-scale feature extraction
        orig_features = F.normalize(encoder(original).latent_dist.mean, dim=1)
        recon_features = F.normalize(encoder(reconstructed).latent_dist.mean, dim=1)

        # Enhanced cosine similarity (feature-level anomaly map)
        feature_anomaly_map = 1 - F.cosine_similarity(orig_features, recon_features, dim=1, eps=1e-8).unsqueeze(1)
        feature_anomaly_map = resize_to_match(feature_anomaly_map, original)  # Match spatial size

        # Apply multi-scale feature enhancement (if enabled)
        if multi_scale:
            scales = [0.5, 1.0, 2.0]  # Downsample and upsample scales
            multi_scale_maps = []
            for scale in scales:
                scaled_original = F.interpolate(original, scale_factor=scale, mode='bilinear', align_corners=False)
                scaled_reconstructed = F.interpolate(reconstructed, scale_factor=scale, mode='bilinear', align_corners=False)

                scaled_orig_features = F.normalize(encoder(scaled_original).latent_dist.mean, dim=1)
                scaled_recon_features = F.normalize(encoder(scaled_reconstructed).latent_dist.mean, dim=1)

                scaled_map = 1 - F.cosine_similarity(scaled_orig_features, scaled_recon_features, dim=1, eps=1e-8).unsqueeze(1)
                scaled_map = resize_to_match(scaled_map, original)  # Match original size
                multi_scale_maps.append(scaled_map)

            # Combine multi-scale maps (weighted sum)
            feature_anomaly_map = torch.stack(multi_scale_maps, dim=0).mean(dim=0)

        # Enhance anomaly response (emphasize high anomalies)
        feature_anomaly_map = torch.pow(feature_anomaly_map, alpha)
        
        return feature_anomaly_map  # Amplify high values

    feature_anomaly_map = compute_feature_anomaly_map(original, reconstructed, encoder, alpha=2.0, multi_scale=True)

    # Pixel-level anomaly map
    pixel_diff = torch.abs(original - reconstructed)  # Per-pixel absolute difference
    pixel_anomaly_map = pixel_diff.mean(dim=1, keepdim=True)  # Channel-wise average
    pixel_anomaly_map = (pixel_anomaly_map - pixel_anomaly_map.min()) / (pixel_anomaly_map.max() - pixel_anomaly_map.min() + 1e-8)  # Normalize

    # Weighted combination with improved ratios
    w1, w2 = combine_weights
    combined_anomaly_map = torch.sqrt(w1 * feature_anomaly_map ** 2 + w2 * pixel_anomaly_map ** 2)
    
    return combined_anomaly_map, feature_anomaly_map, pixel_anomaly_map

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