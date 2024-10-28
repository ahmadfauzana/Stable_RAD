import os
import numpy as np
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from dataset import train_transforms, test_transforms, MVTecDataset

def initiate_model(device):
    model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    return model

def load_features(_class_, args):
    reference_features = np.load(args.save_path + f"db_features_{_class_}.npy")
    return reference_features

def train_data(args, _class_):
    data_transform, _ = train_transforms(args.image_size, args.image_size, args.mean, args.std)
    train_path = os.path.join(args.root_path, _class_, 'train')
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    return train_dataloader

def test_data(args, _class_):
    data_transform, gt_transform = test_transforms(args.image_size, args.image_size, args.mean, args.std)
    test_path = os.path.join(args.root_path, _class_)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)
    return test_dataloader

def timer(start, end, args, item):
    total = end - start
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    with open(args.score_path, 'a') as file:
        file.write(f'{item} class - total {args.phase}ing time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds\n')