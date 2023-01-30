from collections import defaultdict
from poke_env.player import  ObservationType
from poke_env.environment import Gen8Battle
from gym.spaces import Space, Box
import random, string
import numpy as np
from lib.constants import GEN8_MOVE_ID

from lib.env import VsHeuristicEnv
from lib.utils import infx
from .space import BATTLE_SPACE_SIZE, get_battle_space


def id(size):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(size))


class Env(VsHeuristicEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_reward(self, _, current_battle: Gen8Battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=1, victory_value=3
        )

    def embed_battle(self, battle: Gen8Battle) -> ObservationType:
        return get_battle_space(battle)

    def describe_embedding(self) -> Space:
        return Box(-np.inf, np.inf, (BATTLE_SPACE_SIZE,))
