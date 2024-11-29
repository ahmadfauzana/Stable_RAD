import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from preprocess import adaptive_threshold

def compute_anomaly_map(original, reconstructed, encoder):

    feature_anomaly_map = compute_feature_anomaly_map(original, reconstructed, encoder, multi_scale=True)
    
    # Pixel-level anomaly map (optional, for combined evaluation)
    pixel_diff = torch.abs(original - reconstructed)  # Per-pixel absolute difference
    pixel_anomaly_map = pixel_diff.mean(dim=1, keepdim=True)  # Channel-wise average
    pixel_anomaly_map = (pixel_anomaly_map - pixel_anomaly_map.min()) / (pixel_anomaly_map.max() - pixel_anomaly_map.min() + 1e-8)  # Normalize
    
    return feature_anomaly_map, pixel_anomaly_map

def compute_anomaly_score(anomaly_map):
    anomaly_score = anomaly_map.mean(dim=(-1, -2))
    return anomaly_score

def compute_feature_anomaly_map(original, reconstructed, encoder, multi_scale=True):
    """
    Compute feature-level anomaly map with advanced multi-scale and hierarchical improvements.
    """
    def resize_to_match(source, target):
        """Resize source tensor to match target spatial dimensions."""
        return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)

    # Extract normalized latent features (per channel)
    orig_features = F.normalize(encoder(original).latent_dist.mean, dim=1)
    recon_features = F.normalize(encoder(reconstructed).latent_dist.mean, dim=1)

    # Compute absolute residual features (difference between original and reconstructed features)
    residual_features = torch.abs(orig_features - recon_features)
    
    # L2 norm of the residual features to get the primary feature anomaly map
    feature_anomaly_map = torch.norm(residual_features, p=2, dim=1, keepdim=True)
    feature_anomaly_map = resize_to_match(feature_anomaly_map, original)

    # If multi-scale is enabled, refine the feature anomaly map using multiple scales
    if multi_scale:
        scales = [0.25, 0.5, 1.0, 2.0, 4.0]
        scale_weights = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2], device=original.device)
        scale_weights = scale_weights.view(-1, 1, 1, 1)  # Match dimensions

        multi_scale_maps = []
        for i, scale in enumerate(scales):
            # Rescale images and recompute residuals at different scales
            scaled_original = F.interpolate(original, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_reconstructed = F.interpolate(reconstructed, scale_factor=scale, mode='bilinear', align_corners=False)

            # Extract features and compute residuals at each scale
            scaled_orig_features = F.normalize(encoder(scaled_original).latent_dist.mean, dim=1)
            scaled_recon_features = F.normalize(encoder(scaled_reconstructed).latent_dist.mean, dim=1)
            scaled_residual = torch.abs(scaled_orig_features - scaled_recon_features)

            # Compute the feature anomaly map for each scale
            scaled_map = torch.norm(scaled_residual, p=2, dim=1, keepdim=True)
            scaled_map = resize_to_match(scaled_map, original)

            # Weight the maps according to their scale importance
            weighted_map = scaled_map * scale_weights[i]
            multi_scale_maps.append(weighted_map)

        # Aggregate multi-scale anomaly maps
        feature_anomaly_map = torch.sum(torch.stack(multi_scale_maps, dim=0), dim=0)

    # Attention mechanism to highlight more prominent anomaly regions
    patch_size = 16
    attention = F.adaptive_avg_pool2d(feature_anomaly_map, (patch_size, patch_size))
    attention = F.interpolate(attention, size=feature_anomaly_map.shape[2:], mode='bilinear', align_corners=False)
    attention = torch.sigmoid(attention)
    feature_anomaly_map *= attention

    # Dynamic scaling of the feature anomaly map with more emphasis on stronger anomalies
    dynamic_alpha = feature_anomaly_map.mean() + 1.5 * feature_anomaly_map.std()
    dynamic_alpha = torch.clamp(dynamic_alpha, min=1.0, max=3.0)  # Ensure dynamic_alpha stays within reasonable bounds
    feature_anomaly_map = torch.log1p(feature_anomaly_map * dynamic_alpha)

    return feature_anomaly_map

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