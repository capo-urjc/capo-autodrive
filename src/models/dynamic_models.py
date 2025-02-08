import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import NCP
from ncps.torch import LTC, CfC


# Define the Dynamic Model
class DinMod(nn.Module):
    def __init__(self, in_dim,  out_dim, latent_features, dropout_rate=0.1, rnn='cfc'):
        super(DinMod, self).__init__()


        self.fc1 = nn.Linear(in_dim, latent_features) # drone

        self.dropout = nn.Dropout(dropout_rate)

        wiring = NCP(
            inter_neurons=18,
            command_neurons=12,
            motor_neurons=out_dim,
            sensory_fanout=6,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=6,
            seed=2222)

        if rnn == 'ltc':
            self.ltc_model = LTC(latent_features, wiring, batch_first=True, mixed_memory=False)
        elif rnn == 'cfc':
            self.ltc_model = CfC(latent_features, wiring, batch_first=True, mixed_memory=True)

        self.last_state = None

    def forward(self, x, state=None):

        x = self.fc1(x)
        x, state = self.ltc_model.forward(x, hx=state)
        self.last_state = state

        return x