import numpy as np
import matplotlib.pyplot as plt

def normalize(anomaly_map):
    if anomaly_map.ndim == 4:
        anomaly_map = anomaly_map.mean(axis=1)  # Collapse channel dimension

    # Normalize the anomaly map
    anomaly_map_min = anomaly_map.min(axis=(1, 2), keepdims=True)
    anomaly_map_max = anomaly_map.max(axis=(1, 2), keepdims=True)
    normalized_anomaly_map = ((anomaly_map - anomaly_map_min) / (anomaly_map_max - anomaly_map_min + 1e-6) * 255).astype(np.uint8)

    return normalized_anomaly_map

def visualize_reconstruction(inputs, recon_image, feature_anomaly_map, pixel_anomaly_map, gt_mask, args, save_path):
    # Convert inputs to numpy arrays and denormalize
    if inputs.ndim == 4:
        inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
        inputs = inputs * np.array(args.std) + np.array(args.mean)
    elif inputs.ndim == 3:
        inputs = np.expand_dims(inputs.cpu().numpy(), axis=0)
    else:
        raise ValueError(f"Expected inputs to be 3D or 4D, but got shape: {inputs.shape}")

    # Process reconstructed images
    recon_image = recon_image.cpu().numpy().transpose(0, 2, 3, 1)
    gt_mask = gt_mask.squeeze().cpu().numpy()

    # Process anomaly map
    feature_anomaly_map = feature_anomaly_map.cpu().numpy()
    pixel_anomaly_map = pixel_anomaly_map.cpu().numpy()
    feature_anomaly_map = normalize(feature_anomaly_map)
    pixel_anomaly_map = normalize(pixel_anomaly_map)
    
    # Visualization
    B = len(inputs)
    fig, axs = plt.subplots(B, 5, figsize=(25, 15))

    if len(inputs) == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(B):
        axs[i, 0].imshow(np.clip(inputs[i], 0, 1))
        axs[i, 0].set_title(f'Original {i}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(gt_mask, cmap='gray')
        axs[i, 1].set_title(f'Ground Truth {i}')
        axs[i, 1].axis('off')

        im = axs[i, 2].imshow(feature_anomaly_map[i], cmap='jet', vmin=0, vmax=255)
        axs[i, 2].set_title(f'Feature Anomaly Map {i}')
        axs[i, 2].axis('off')

        axs[i, 3].imshow(pixel_anomaly_map[i], cmap='jet', vmin=0, vmax=255)
        axs[i, 3].set_title(f'Pixel Anomaly Map {i}')
        axs[i, 3].axis('off')

        axs[i, 4].imshow(np.clip(recon_image[i], 0, 1))
        axs[i, 4].set_title(f'Reconstructed {i}')
        axs[i, 4].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(save_path)
    plt.close()