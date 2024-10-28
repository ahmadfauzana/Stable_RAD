import os
import torch
import wandb
import numpy as np
from visualize import visualize_reconstruction
from sklearn.metrics import roc_auc_score
from retrieval import find_similar_images
from setup import initiate_model, test_data, load_features
from utils import denormalize, create_directory_structure, compute_anomaly_map, compute_anomaly_score, highlight_anomaly

def test(_class_, args, device):
    # Initialize WandB
    wandb.init(project="stable_rad", entity="afauzanaqil", name=f"test_{_class_}")
    
    # Log the hyperparameters to WandB
    wandb.config.update(vars(args))

    print(f"Testing on {_class_} started")
    model = initiate_model(device)

    ckpt_file = os.path.join(args.ckpt_path, f'model_{_class_}.pth')
    if os.path.exists(ckpt_file):
        print(f"Loading model from checkpoint: {ckpt_file}")
        model.load_state_dict(torch.load(ckpt_file))
    else:
        print("No checkpoint found, starting from scratch.")
    
    model.eval()

    encoder = model.encode
    decoder = model.decode
    
    all_labels = []
    anomap_scores = []
    anomaly_scores = []
    
    test_output_dirs = create_directory_structure(args.output_path, args.phase, ckpt_file, args.item_list)
    dataloader = test_data(args, _class_,)
    reference_features = load_features(_class_, args)

    with torch.no_grad():
        for i, (inputs, masks, labels, _) in enumerate(dataloader):
            torch.cuda.empty_cache()
            inputs = inputs.to(device)

            # Extract latent features from the test input using the encoder
            latents = encoder(inputs).latent_dist.mean  # Latent mean from VAE
            latents = latents.to(device)

            B, C, H, W = latents.shape

            # Flatten latent features
            latents_flat = latents.view(B, -1)  # Flatten to (B, C*H*W)
            
            # Retrieve similar features based on cosine similarity
            retrieved_features = []
            for f in latents_flat.cpu().numpy():
                idx = find_similar_images(f, reference_features)
                retrieved_features.append(reference_features[idx])

            # Convert retrieved features back to tensor and reshape
            retrieved_features = np.array(retrieved_features)  # List of arrays to a single array
            retrieved_features = torch.tensor(retrieved_features, device=device).view(B, C, H, W)  # Reshape back to original

            # Decode retrieved latent features
            recon_image = decoder(retrieved_features).sample  # Decode using Stable Diffusion decoder
            recon_image = denormalize(recon_image, args.mean, args.std)  # Denormalize reconstructed images

            # Compute anomaly map and anomaly score
            anomaly_map = compute_anomaly_map(inputs, recon_image)
            anomaly_score = compute_anomaly_score(inputs, recon_image)
            highlighted_image = highlight_anomaly(inputs, recon_image, anomaly_map)

            # Extend results
            anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))

            for j in range(inputs.size(0)):
                save_path = os.path.join(test_output_dirs[_class_], f'{_class_}_{i}{j}.png')
                visualize_reconstruction(
                    inputs[j].unsqueeze(0),  # Add batch dimension for each image
                    recon_image[j].unsqueeze(0),
                    anomaly_map[j].unsqueeze(0),
                    highlighted_image[j].unsqueeze(0),
                    args,
                    save_path
                )            
                wandb.log({"output_images": wandb.Image(save_path)})

            # Compute ROC AUC score for anomaly detection
            gt_mask = masks.cpu().numpy().astype(int)
            pred_ano_map = anomaly_map.cpu().numpy()

            for b in range(inputs.size(0)):
                if np.unique(gt_mask[b]).size > 1:
                    anomap_score = roc_auc_score(gt_mask[b].reshape(-1), pred_ano_map[b].reshape(-1))
                    anomap_scores.append(anomap_score)
            
    # Compute AUROC score
    anomap_scores = np.array(anomap_scores)
    anomaly_scores = np.array(anomaly_scores)
    all_labels = np.array(all_labels)

    mean_anomap_score = np.mean(anomap_scores) if len(anomap_scores) > 0 else None
    auroc_score = roc_auc_score(all_labels, anomaly_scores)
    
    # Log AUROC scores to WandB and MLflow
    wandb.log({"anomap_auroc": mean_anomap_score, "auroc": auroc_score})
    
    # Save the scores to a file
    with open(args.score_path, 'a') as file:
        file.write(f'{_class_} class, Zero Shot, Anomaly Map AUROC: {mean_anomap_score}, AUROC Score: {auroc_score}\n')
    
    print(f'Anomaly Map Score = {mean_anomap_score}')
    print(f'AUROC Score = {auroc_score}')
    print(f"Testing on {_class_} finished")

    # End WandB
    wandb.finish()