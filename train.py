import os
import tqdm
import torch
import wandb
import numpy as np
from torch.optim import AdamW
from retrieval import find_similar_images
from visualize import visualize_reconstruction
from torch.optim.lr_scheduler import CosineAnnealingLR
from setup import train_data, initiate_model, load_features
from utils import denormalize, loss_function, compute_anomaly_map, compute_anomaly_score, highlight_anomaly, create_directory_structure

def train(_class_, args, device):
    wandb.init(project="stable_rad", entity="afauzanaqil", name=f"train_{_class_}")
    wandb.config.update(vars(args))

    os.makedirs(args.ckpt_path, exist_ok=True)
    train_dataloader = train_data(args, _class_)
    print(f"Training on {_class_} started")
    model = initiate_model(device)
    
    ckpt_file = os.path.join(args.ckpt_path, f'model_{_class_}.pth')
    if os.path.exists(ckpt_file):
        print(f"Loading model from checkpoint: {ckpt_file}")
        model.load_state_dict(torch.load(ckpt_file))
    else:
        print("No checkpoint found, starting from scratch.")

    train_output_dirs = create_directory_structure(args.output_path, args.phase, ckpt_file, args.item_list)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=0.005, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load reference features if available
    reference_features = load_features(_class_, args)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_anomaly_scores = []

        for i, (images, labels) in tqdm.tqdm(enumerate(train_dataloader)):
            torch.cuda.empty_cache()
            inputs = images
            inputs = inputs.to(device)
            
            # Extract latent features using the VAE encoder
            latents = model.encode(inputs).latent_dist.mean.to(device)
            B, C, H, W = latents.shape

            # Retrieve similar features using cosine similarity
            retrieved_features = []
            for f in latents.detach().cpu().numpy():
                idx = find_similar_images(f, reference_features)
                retrieved_features.append(reference_features[idx])

            retrieved_features = np.array(retrieved_features)  # Convert list of NumPy arrays to a single NumPy array
            retrieved_features = torch.tensor(retrieved_features, device=device).view(B, C, H, W)

            # Directly decode the latent representation and retrieved features
            outputs = model.decode(retrieved_features).sample
            
            del latents
            torch.cuda.empty_cache()
            
            outputs = denormalize(outputs, args.mean, args.std)

            # Compute anomaly map and score
            anomaly_map = compute_anomaly_map(inputs, outputs)
            anomaly_score = compute_anomaly_score(inputs, outputs)
            highlighted_image = highlight_anomaly(inputs, outputs, anomaly_map)

            # Compute loss between original images and reconstructed outputs
            loss = loss_function(outputs, inputs)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_anomaly_scores.extend(anomaly_score.detach().cpu().numpy())

        # Visualize images at the end of each epoch
        save_path = os.path.join(train_output_dirs[_class_], f'{_class_}_epoch_{epoch+1}.png')
        visualize_reconstruction(inputs, outputs.detach(), anomaly_map.detach(), highlighted_image.detach(), args, save_path)

        # Print average loss and anomaly scores for the epoch
        average_loss = epoch_loss / len(train_dataloader)
        average_anomaly_score = np.mean(epoch_anomaly_scores) if epoch_anomaly_scores else 0
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {average_loss:.4f}, Anomaly Score: {average_anomaly_score:.4f}")

        with open(args.score_path, 'a') as file:
            file.write(f'{_class_} class, Epoch [{epoch+1}/{args.epochs}], Loss: {average_loss:.4f}, Anomaly Score: {average_anomaly_score:.4f}\n')

        # Log the output
        wandb.log({"loss": average_loss, "auroc": average_anomaly_score})
        wandb.log({"output_images": wandb.Image(save_path)})

        # Step the scheduler
        scheduler.step()

    # Save the model checkpoint with a valid filename
    torch.save(model.state_dict(), ckpt_file)
    print(f"Training on {_class_} finished")

    wandb.finish()