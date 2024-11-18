import numpy as np
import matplotlib.pyplot as plt

def visualize_reconstruction(inputs, recon_image, anomaly_map, gt_mask, args, save_path, show_anomaly_map=True):
    """Visualize original images, binary anomaly masks, reconstructed images, and highlighted anomalies, then save the figure horizontally.
    
    Parameters:
    - inputs: Original input images (batch, height, width, channels).
    - recon_image: Reconstructed images (batch, height, width, channels).
    - anomaly_map: Anomaly map (batch, height, width).
    - gt_mask: Ground truth binary mask (batch, height, width).
    - args: Arguments containing normalization parameters.
    - save_path: Path to save the visualization figure.
    - show_anomaly_map: Boolean flag to control the display of anomaly maps.
    """
    
    # Convert inputs to numpy arrays and denormalize
    if inputs.ndim == 4:
        inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # (batch, height, width, channels)
        inputs = inputs * np.array(args.std) + np.array(args.mean)  # Denormalize
    elif inputs.ndim == 3:
        inputs = np.expand_dims(inputs.cpu().numpy(), axis=0)  # Add batch dimension
    else:
        raise ValueError(f"Expected inputs to be 3D or 4D, but got shape: {inputs.shape}")

    # Process reconstructed images and ground truth masks
    recon_image = recon_image.cpu().numpy().transpose(0, 2, 3, 1)
    gt_mask = gt_mask.squeeze().cpu().numpy()

    # Normalize anomaly map to [0, 1] range for overlay
    if show_anomaly_map:
        anomaly_map = anomaly_map.cpu().numpy()
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)
        
        # Normalize the anomaly map values between 0 and 1
        anomaly_map_min = anomaly_map.min(axis=(1, 2), keepdims=True)
        anomaly_map_max = anomaly_map.max(axis=(1, 2), keepdims=True)
        anomaly_map_normalized = (anomaly_map - anomaly_map_min) / (anomaly_map_max - anomaly_map_min + 1e-6)
        
        # Convert the normalized anomaly map to a 3-channel heatmap
        anomaly_heatmap = np.zeros((*anomaly_map_normalized.shape, 3))
        for i in range(anomaly_map_normalized.shape[0]):
            anomaly_heatmap[i] = plt.cm.jet(anomaly_map_normalized[i])[:, :, :3]  # Remove alpha channel

        # Blend anomaly heatmap with the original image
        overlayed_images = 0.7 * inputs + 0.3 * anomaly_heatmap  # Adjust blending ratio as desired

    B = len(inputs)  # Batch size
    
    # Adjust layout based on whether the anomaly map is shown
    num_columns = 3 if not show_anomaly_map else 4
    fig, axs = plt.subplots(B, num_columns, figsize=(25, 15))  # Set the layout to horizontal (batch in rows, images in columns)

    if len(inputs) == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(B):
        # Original Image
        axs[i, 0].imshow(np.clip(inputs[i], 0, 1))
        axs[i, 0].set_title(f'Original {i}')
        axs[i, 0].axis('off')

        # Ground Truth
        axs[i, 1].imshow(gt_mask, cmap='gray')
        axs[i, 1].set_title(f'Ground Truth {i}')
        axs[i, 1].axis('off')

        if show_anomaly_map:
            # Overlayed Image (Original + Anomaly Map)
            im = axs[i, 2].imshow(np.clip(overlayed_images[i], 0, 1))
            axs[i, 2].set_title(f'Anomaly Overlay {i}')
            axs[i, 2].axis('off')

            # Reconstructed Image
            axs[i, 3].imshow(np.clip(recon_image[i], 0, 1))
            axs[i, 3].set_title(f'Reconstructed {i}')
            axs[i, 3].axis('off')
        else:
            # Just Reconstructed Image if no anomaly map
            axs[i, 2].imshow(np.clip(recon_image[i], 0, 1))
            axs[i, 2].set_title(f'Reconstructed {i}')
            axs[i, 2].axis('off')

    # Adjust layout to fit everything properly
    plt.tight_layout()

    # Add a color bar for the anomaly map, ensuring it's properly centered and scaled
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Save the figure
    plt.savefig(save_path)
    plt.close()