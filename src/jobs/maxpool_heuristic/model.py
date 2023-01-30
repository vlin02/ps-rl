from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch

from .space import MOVE_STATE_SIZE, POKEMON_STATE_SIZE
from lib.utils import infx

class MoveNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.nn = nn.Sequential(
            nn.Linear(MOVE_STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, move_space):
        x = self.nn(move_space)

        return x

class PokeNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.move_nn = MoveNN()

        self.nn = nn.Sequential(
            nn.Linear(POKEMON_STATE_SIZE + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
    
    def forward(self, poke_space):
        x0 = poke_space['state']

        # 4 x 32 x 256
        x1 = torch.stack([self.move_nn(move) for move in poke_space['moves']])
        x1 = torch.amax(x1, dim=0)

        x2 = torch.concat([x1, x0], dim=1)

        x3 = self.nn(x2)

        return x3
i = 0

def infxx(*args):
    if i == 50:
        infx(*args)

class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.poke_nn = PokeNN()

        self.nn =  nn.Sequential(
            nn.Linear(3 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self._logits = nn.Linear(512,num_outputs)
        self._value_branch = nn.Linear(512,1)

    def forward(self, input_dict, state, _):
        global i
        i += 1

        samples = input_dict['obs']
        
        team = samples['team']
        opp = samples['opp_active']

        x1 = torch.stack([self.poke_nn(poke) for poke in team])
        # 32 x 6 x 256
        x1 = x1.swapaxes(0, 1)
        
        active = samples['active'][..., :6].unsqueeze(2).expand(x1.shape)
        x2 = x1 * active
        x2 = x2.amax(dim=1)

        x3 = x1.amax(dim=1)

        x4 = self.poke_nn(opp)
        
        x5 = torch.cat([x2,x3,x4], dim=1)
        self._features = self.nn(x5)
        
        logits = self._logits(self._features)
        return logits, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
