from pprint import pprint
from poke_env.player import  ObservationType
from poke_env.environment import Gen8Battle
from gym.spaces import Space, Box
import random, string
import numpy as np
from .space import get_battle_space, battle_space

from lib.env import VsHeuristicEnv

def id(size):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(size))

class Env(VsHeuristicEnv):

    def calc_reward(self, _, current_battle: Gen8Battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=1, victory_value=3
        )

    def embed_battle(self, battle: Gen8Battle) -> ObservationType:
        return get_battle_space(battle)

    def describe_embedding(self) -> Space:
        return battle_space