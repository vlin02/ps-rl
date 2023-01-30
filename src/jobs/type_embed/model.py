from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch
from poke_env.environment import PokemonType

EMBED_SIZE = 50

class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.type_embed = nn.Embedding(len(PokemonType) + 1, EMBED_SIZE, padding_idx=0)

        self.nn = nn.Sequential(
            nn.Linear(6 * EMBED_SIZE + 4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self._logits = nn.Linear(256,num_outputs)
        self._value_branch = nn.Linear(256,1)

    def forward(self, input_dict, state, _):
        samples: torch.Tensor = input_dict['obs']
        
        # X, 6
        x1 = samples[...,4:]
        x1 = x1.int()

        # X, 6, 5
        x2 = self.type_embed(x1)
        
        # X, 30
        x2 = x2.flatten(start_dim=1)
        
        # X, 4
        x3 = samples[...,:4]

        x4 = torch.concat([x2, x3],dim=1)
        x4 = self.nn(x4)

        self._features = x4
        
        logits = self._logits(self._features)
        return logits, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
