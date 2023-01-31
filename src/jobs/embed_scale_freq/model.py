from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch

from lib.constants import GEN8_MOVE_ID

EMBED_SIZE = 5


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.move_embed = nn.Embedding(
            len(GEN8_MOVE_ID),
            EMBED_SIZE,
            padding_idx=0,
            scale_grad_by_freq=True,
        )

        self.nn = nn.Sequential(
            nn.Linear(4 * EMBED_SIZE + 8, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self._logits = nn.Linear(256, num_outputs)
        self._value_branch = nn.Linear(256, 1)

    def forward(self, input_dict, state, _):
        samples: torch.Tensor = input_dict["obs"]

        # X, 4
        x1 = samples[..., :4]
        x1 = x1.int()

        # X, 5, 4
        x2 = self.move_embed(x1)

        # X, 20
        x2 = x2.flatten(start_dim=1)

        # X, 8
        x3 = samples[..., 4:]

        x4 = torch.concat([x2, x3], dim=1)
        x4 = self.nn(x4)

        self._features = x4

        logits = self._logits(self._features)
        return logits, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
