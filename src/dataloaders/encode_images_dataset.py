import os
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from src.dataloaders.transformations import ImageNormalization
from torch.utils.data import DataLoader, Dataset


class Dinov2Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_model = 'dinov2_vitb14_reg_lc'
        # self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_model)
    def center_crop_tensor(self, img_tensor, output_size):
        _, _, h, w = img_tensor.shape
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        new_h, new_w = output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img_tensor[:, :, top:top + new_h, left:left + new_w]

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        patch_size = getattr(self.dino, "patch_size", getattr(getattr(self.dino, "backbone", None), "patch_size", None))
        output_size = (patch_size * (h // patch_size), patch_size * (w // patch_size))
        x_resized = self.center_crop_tensor(x, output_size=output_size)
        latent = self.dino(x_resized)
        latent = latent.reshape(b, s, -1)
        return latent


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image).transpose((2, 0, 1))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, img_path
        keys = {'rgb': image}
        if self.transform:
            keys = self.transform(keys)

        return keys['rgb'][None,...]

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []

    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            features = model(images).cpu().numpy()

            all_features.extend(features[:,0,...])

    return np.array(all_features)

def process_images_and_save_csv(image_paths, csv_path, batch_size=64, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Enc().to(device)

    transform = transforms.Compose([
        ImageNormalization(),
    ])

    dataset = ImageDataset(image_paths, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    features = extract_features(model, dataloader, device)

    np.save(csv_path+'/'+model.backbone_model+'.npy', features)


    print(f"Features saved to {csv_path+'/'+model.backbone_model+'.npy'}")

# def process_images_and_save_csv(image_paths, csv_path):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Dinov2Enc().to(device)
#     model.eval()
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     all_features = []
#     image_names = []
#
#     for img_path in tqdm(image_paths):
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             continue
#
#         image = transform(image).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
#         image = image.to(device)
#
#         with torch.no_grad():
#             features = model(image).cpu().numpy().flatten()
#
#         all_features.append(features)
#         image_names.append(img_path)
#
#     df = pd.DataFrame(all_features)
#     # df.insert(0, "image_path", image_names)

#     df.to_csv(csv_path+'/'+model.backbone_model+'.csv', index=False)
#     print(f"Features saved to {csv_path+'/'+model.backbone_model+'.csv'}")


def read_config_folders(csv_file):

    df = pd.read_csv(csv_file)
    if "Folder Path" in df.columns:
        folder_paths = df["Folder Path"].tolist()
        file_count = df["File Count"].tolist()
        print("Folder Paths:")

        for path, n_frames in zip(folder_paths, file_count):
            camera_folders = [f for f in os.listdir(path) if "Camera" in f and "CameraBEV" not in f and os.path.isdir(os.path.join(path, f))]

            for camera in camera_folders:
                all_images = []
                for frame in range(n_frames):
                    all_images.append(path+'/'+camera+'/'+str(frame)+'.png')
                    # print(path+'/'+camera+'/'+str(frame)+'.png')

                process_images_and_save_csv(all_images, path+'/'+camera)
    else:
        print("Error: 'Folder Path' column not found in CSV file.")


if __name__=="__main_":
    # Example usage
    # mean_std = compute_mean_std("src/dataloaders/csv/config_folders.csv")
    # print(mean_std)
    all_images = read_config_folders("src/dataloaders/csv/config_folders.csv")
