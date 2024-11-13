import os
import torch
import numpy as np
import torch.nn.functional as F
from setup import train_data, initiate_model
from sklearn.metrics.pairwise import cosine_similarity

def retrieval(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = initiate_model(device)
    model.eval()
    
    # Create a list to store features
    features_list = []

    for item in args.item_list:
        dataloader = train_data(args, item)

        # Extract features for each image in the dataset
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                # Extract latent embeddings from VAE encoder
                latents = model.encode(inputs).latent_dist.mean.to(device)  # Latent mean
                
                # Add to the feature list
                features_list.append(latents.cpu().numpy())

        # Convert the list of features to a numpy array and save
        features_db = np.vstack(features_list)
        if not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        np.save(args.save_path + 'db_features_' + item + '.npy', features_db)
        print(f"Features saved to db_features_{item}.npy")

def find_similar_images(query_feature, reference_features):
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
    reference_features_flat = reference_features.reshape(reference_features.shape[0], -1)  # Shape becomes [209, 4*32*32]

    # Compute cosine similarity between the query feature and all reference features
    similarities = cosine_similarity(query_feature, reference_features_flat)

    # Find the index of the most similar reference feature
    idx = np.argmax(similarities)

    return idx
