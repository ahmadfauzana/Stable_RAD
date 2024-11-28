import os
import torch
import wandb
import numpy as np
from visualize import visualize_reconstruction
from sklearn.metrics import roc_auc_score
from retrieval import find_similar_images
from metrics import compute_anomaly_map, compute_anomaly_score
from setup import initiate_model, test_data, load_features
from utils import create_directory_structure
from preprocess import denormalize

def test(_class_, args, device):
    wandb.init(project="stable_rad", entity="afauzanaqil", name=f"test_{_class_}")
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
    feature_anomap_scores = []
    feature_anomaly_scores = []

    test_output_dirs = create_directory_structure(args.output_path, args.phase, ckpt_file, args.item_list)
    dataloader = test_data(args, _class_)
    reference_features = load_features(_class_, args)

    with torch.no_grad():
        for i, (inputs, masks, labels, _) in enumerate(dataloader):
            torch.cuda.empty_cache()
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Extract latent features from the test input using the encoder
            latents = encoder(inputs).latent_dist.mean
            latents = latents.to(device)
            B, C, H, W = latents.shape
            latents_flat = latents.view(B, -1)
            
            # Initialize retrieved features tensor
            retrieved_features = torch.zeros((B, C, H, W), device=device)

            # Retrieve similar features
            for idx, f in enumerate(latents_flat.cpu().numpy()):
                feature_idx = find_similar_images(f, reference_features)
                if feature_idx != -1:
                    retrieved_features[idx] = torch.tensor(reference_features[feature_idx], device=device).view(C, H, W)
                else:
                    # If no match is found, keep as zeroed tensor or apply other handling if necessary
                    retrieved_features[idx] = torch.zeros((C, H, W), device=device)

            # Decode and denormalize reconstructed images
            recon_image = decoder(retrieved_features).sample
            recon_image = denormalize(recon_image, args.mean, args.std)

            # Enhanced anomaly map and score calculation
            feature_anomaly_map = compute_anomaly_map(inputs, recon_image, encoder, 'absolute')
            feature_anomaly_score = compute_anomaly_score(feature_anomaly_map, inputs, recon_image, 'mean')
            
            # Extend results
            feature_anomaly_scores.extend(feature_anomaly_score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))

            for j in range(inputs.size(0)):
                save_path = os.path.join(test_output_dirs[_class_], f'{_class_}_{i}{j}.png')
                visualize_reconstruction(
                    inputs[j].unsqueeze(0),
                    recon_image[j].unsqueeze(0),
                    feature_anomaly_map[j].unsqueeze(0),
                    masks[j].unsqueeze(0),
                    args,
                    save_path
                )            
                wandb.log({"output_images": wandb.Image(save_path)})

            # Calculate ROC AUC scores for the anomaly map and score
            gt_mask = masks.cpu().numpy().astype(int)
            feature_ano_map = feature_anomaly_map.cpu().numpy()
            
            for b in range(inputs.size(0)):
                if np.unique(gt_mask[b]).size > 1:
                    feat_anomap_score = roc_auc_score(gt_mask[b].reshape(-1), feature_ano_map[b].reshape(-1))
                    print(f"Feature-Level Anomaly Score of {b}: {feat_anomap_score}")
                    feature_anomap_scores.append(feat_anomap_score)
                    
    # Compute and log AUROC
    feature_anomap_scores = np.array(feature_anomap_scores)
    feature_anomaly_scores = np.array(feature_anomaly_scores)
    all_labels = np.array(all_labels)

    feature_wise = np.mean(feature_anomap_scores) if len(feature_anomap_scores) > 0 else None
    feature_image_wise = roc_auc_score(all_labels, feature_anomaly_scores)
    
    wandb.log({"feature-wise": feature_wise, "feature image-wise auroc": feature_image_wise})

    # Save the scores
    with open(args.score_path, 'a') as file:
        if os.path.exists(ckpt_file):
            file.write(f'{_class_} class with Checkpoint, Feature-wise AUROC: {feature_wise}, Feature-image Score: {feature_image_wise}\n')
        else:
            file.write(f'{_class_} class with Retrieval Only, Feature-wise AUROC: {feature_wise}, Feature-image Score: {feature_image_wise}\n')
            
    print(f'Feature-wise AUROC = {feature_wise}')
    print(f'Feature-Image AUROC = {feature_image_wise}')
    print(f"Testing on {_class_} finished")
    wandb.finish()