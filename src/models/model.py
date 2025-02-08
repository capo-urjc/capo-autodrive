import torch
from torch import nn
from src.models.encoders import Dinov2Enc
from src.models.dynamic_models import DinMod


class Model(nn.Module):

    # Initialize the parameter
    def __init__(self, latent_features):
        super(Model, self).__init__()
        self.encoder = Dinov2Enc()
        self.projector = nn.Sequential(
            nn.Linear(384, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128)
        )
        self.dynamic_predictor = DinMod(in_dim=128,  out_dim=4, latent_features=latent_features)
        self.drop = nn.Dropout(0.25)
        # Forward pass
    def forward(self, x):
        x = self.encoder(x).detach()
        # x = self.drop(x)
        x = self.projector(x)
        pred = self.dynamic_predictor(x)
        return pred
