import numpy as np
import matplotlib.pyplot as plt

def visualize_reconstruction(inputs, recon_image, anomaly_map, highlighted_image, args, save_path):
    """Visualize original images, binary anomaly masks, anomaly maps, reconstructed images, and highlighted anomalies, then save the figure horizontally."""
    
    # Convert inputs to numpy arrays and denormalize
    if inputs.ndim == 4:
        inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # (batch, height, width, channels)
        inputs = inputs * np.array(args.std) + np.array(args.mean)  # Denormalize
    elif inputs.ndim == 3:
        inputs = np.expand_dims(inputs.cpu().numpy(), axis=0)  # Add batch dimension
    else:
        raise ValueError(f"Expected inputs to be 3D or 4D, but got shape: {inputs.shape}")

    # Process reconstructed images and highlighted images
    recon_image = recon_image.cpu().numpy().transpose(0, 2, 3, 1) * np.array(args.std) + np.array(args.mean)
    highlighted_image = highlighted_image.cpu().numpy().transpose(0, 2, 3, 1)

    # Process anomaly map
    anomaly_map = anomaly_map.cpu().numpy()
    if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map.squeeze(1)

    # Normalize the anomaly map between 0 and 255 for heatmap visualization
    anomaly_map_min = anomaly_map.min(axis=(1, 2), keepdims=True)
    anomaly_map_max = anomaly_map.max(axis=(1, 2), keepdims=True)
    anomaly_map_normalized = (anomaly_map - anomaly_map_min) / (anomaly_map_max - anomaly_map_min + 1e-6) * 255

    # Create a binary anomaly mask for overlay
    binary_anomaly_mask = (anomaly_map > args.threshold).astype(np.uint8)

    B = len(inputs)  # Batch size
    fig, axs = plt.subplots(B, 5, figsize=(25, 15))  # Set the layout to horizontal (batch in rows, images in columns)

    if len(inputs) == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(B):
        # Original Image
        axs[i, 0].imshow(np.clip(inputs[i], 0, 1))
        axs[i, 0].set_title(f'Original {i}')
        axs[i, 0].axis('off')

        # Binary Anomaly Mask Overlay
        overlay_image = np.copy(inputs[i])
        overlay_image[binary_anomaly_mask[i] == 1] = [1, 0, 0]  # Red overlay
        axs[i, 1].imshow(np.clip(overlay_image, 0, 1))
        axs[i, 1].set_title(f'Binary Anomaly Mask {i}')
        axs[i, 1].axis('off')

        # Anomaly Map (as heatmap)
        im = axs[i, 2].imshow(anomaly_map_normalized[i], cmap='jet', vmin=0, vmax=255)
        axs[i, 2].set_title(f'Anomaly Map {i}')
        axs[i, 2].axis('off')

        # Reconstructed Image
        axs[i, 3].imshow(np.clip(recon_image[i], 0, 1))
        axs[i, 3].set_title(f'Reconstructed {i}')
        axs[i, 3].axis('off')

        # Highlighted Anomaly (blend original and anomaly mask)
        highlighted_image = np.copy(inputs[i])
        alpha = 0.5  # Blending factor
        anomaly_areas = binary_anomaly_mask[i] == 1
        highlighted_image[anomaly_areas] = (1 - alpha) * highlighted_image[anomaly_areas] + alpha * np.array([1, 0, 0])

        axs[i, 4].imshow(np.clip(highlighted_image, 0, 1))
        axs[i, 4].set_title(f'Highlighted {i}')
        axs[i, 4].axis('off')

    # Adjust layout to fit everything properly
    plt.tight_layout()

    # Add a color bar for the anomaly map, ensuring it's properly centered and scaled
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Save the figure
    plt.savefig(save_path)
    plt.close()