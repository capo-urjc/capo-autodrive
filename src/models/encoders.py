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

class Dinov2Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def center_crop_tensor(self, img_tensor, output_size):
        """Center crop the given PyTorch tensor.

        Args:
            img_tensor (torch.Tensor): Image tensor to be cropped (C, H, W).
            output_size (tuple or int): Desired output size. If int, a square crop is made.

        Returns:
            torch.Tensor: Center-cropped image tensor.
        """
        _,_, h, w = img_tensor.shape
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        new_h, new_w = output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return img_tensor[:,:, top:top + new_h, left:left + new_w]

    def forward(self, x):
        b, s, c, h, w = x.shape

        x = x.view(b*s, c, h, w)

        output_size = (14*(h//14), 14*(w//14))
        x_resized = self.center_crop_tensor(x, output_size=output_size)
        latent = self.dino(x_resized)

        latent = latent.reshape(b, -1) # (batch x [384 * 5])

        return latent

