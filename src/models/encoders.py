import torch
from torch import nn
from torchvision import transforms

def center_crop_tensor(img_tensor, output_size):
    """Center crop the given PyTorch tensor.

    Args:
        img_tensor (torch.Tensor): Image tensor to be cropped (C, H, W).
        output_size (tuple or int): Desired output size. If int, a square crop is made.

    Returns:
        torch.Tensor: Center-cropped image tensor.
    """
    _, h, w = img_tensor.shape
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    new_h, new_w = output_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    return img_tensor[:, top:top + new_h, left:left + new_w]

# class Dinov2Enc(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#
#     def center_crop_tensor(self, img_tensor, output_size):
#         """Center crop the given PyTorch tensor.
#
#         Args:
#             img_tensor (torch.Tensor): Image tensor to be cropped (C, H, W).
#             output_size (tuple or int): Desired output size. If int, a square crop is made.
#
#         Returns:
#             torch.Tensor: Center-cropped image tensor.
#         """
#         _,_, h, w = img_tensor.shape
#         if isinstance(output_size, int):
#             output_size = (output_size, output_size)
#
#         new_h, new_w = output_size
#         top = (h - new_h) // 2
#         left = (w - new_w) // 2
#
#         return img_tensor[:,:, top:top + new_h, left:left + new_w]
#
#     def forward(self, x):
#         b, s, c, h, w = x.shape
#
#         x = x.view(b*s, c, h, w)
#
#         output_size = (14*(h//14), 14*(w//14))
#         x_resized = self.center_crop_tensor(x, output_size=output_size)
#         latent = self.dino(x_resized)
#
#         latent = latent.reshape(b, s, -1) # (batch x [384 * 5])
#
#         return latent

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


from src.dataloaders.transformations import RecordsTransform, ImageNormalization
from torch.utils.data import DataLoader, Dataset
from src.dataloaders.dataset import AutodriveDataset
import numpy as np

if __name__ == "__main__":
    csv_file = "src/dataloaders/config_folders.csv"
    # dataset = AutodriveDataset(csv_file, seq_len=5, transform=None, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_bev', 'gnss', 'imu', 'lidar', 'radar'])
    transform = transforms.Compose([
        ImageNormalization(),
        RecordsTransform(),
    ])
    # dataset = AutodriveDataset(csv_file, seq_len=10, transform=transform, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_lb', 'rgb_rb', 'rgb_b', 'records'], use_encoded_images=False)
    dataset = AutodriveDataset(csv_file, subset='train', seq_len=1, transform=transform, sensors=['rgb_f', 'records'], use_encoded_images=False)

    dl = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Dinov2Enc().cuda()
    for batch in (dl):
        img = batch['rgb_f']
        wps = batch['wps']

        img = img.to('cuda')
        # img = img[:,:,:,:,0:450]

        output = model(img)

        print(1)



