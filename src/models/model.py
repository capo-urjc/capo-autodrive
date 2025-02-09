import torch
from torch import nn
# from src.models.encoders import Dinov2Enc
from .dynamic_models import DinMod


class Model(nn.Module):

    # Initialize the parameter
    def __init__(self, latent_features, config):
        super(Model, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(1000, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)
        )

        num_wps = 3

        self.dynamic_predictor = DinMod(in_dim=256,  out_dim=num_wps, latent_features=latent_features)
        self.drop = nn.Dropout(0.25)

    # Forward pass
    def forward(self, x):

        x = self.projector(x)
        pred = self.dynamic_predictor(x)

        return pred
