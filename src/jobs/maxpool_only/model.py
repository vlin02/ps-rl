from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch

class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.move_nn = nn.Sequential(
            nn.Linear(6, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self._logits = nn.Linear(256,num_outputs)
        self._value_branch = nn.Linear(256,1)

    def forward(self, input_dict, state, _):
        samples = input_dict['obs']
        n_samples = samples.shape[0]
        
        x1 = samples[...,:4].unsqueeze(-1)
        x2 = samples[...,4:8].unsqueeze(-1)

        x3 = torch.eye(4)
        x3 = x3.expand((n_samples, 4, 4))

        x4 = torch.concat([x1, x2, x3],dim=-1)
        
        x4 = self.move_nn(x4)
        x4 = torch.amax(x4, dim=-2)

        self._features = x4
        
        logits = self._logits(self._features)
        return logits, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
