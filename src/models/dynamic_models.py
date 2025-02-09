import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import NCP
from ncps.torch import LTC, CfC


# Define the Dynamic Model
class DinMod(nn.Module):
    def __init__(self, in_dim,  out_dim, latent_features, dropout_rate=0.1, rnn='cfc'):
        super(DinMod, self).__init__()
        # self.fc1 = nn.Linear(in_dim, latent_features) # drone
        self.dropout = nn.Dropout(dropout_rate)
        # wiring = NCP(
        #     inter_neurons=18,
        #     command_neurons=12,
        #     motor_neurons=out_dim,
        #     sensory_fanout=6,
        #     inter_fanout=4,
        #     recurrent_command_synapses=4,
        #     motor_fanin=6,
        #     seed=2222)

        if rnn == 'ltc':
            # self.ltc_model = LTC(latent_features, wiring, batch_first=True, mixed_memory=False)
            self.ltc_model = LTC(in_dim, latent_features, batch_first=True, mixed_memory=True)
        elif rnn == 'cfc':
            # self.ltc_model = CfC(latent_features, wiring, batch_first=True, mixed_memory=True)
            self.ltc_model = CfC(in_dim, latent_features, batch_first=True, mixed_memory=False)

        self.last_state = None

        self.project_to_wps = nn.Sequential(
            nn.Linear(latent_features, latent_features//2),
            nn.SiLU(),
            nn.Linear(latent_features//2, out_dim)
            )

    def forward(self, x, state=None):

        # x = self.fc1(x)
        x, state = self.ltc_model.forward(x, hx=state, timespans=None) # qué timespan usamos?
        out = self.project_to_wps(x)

        return out