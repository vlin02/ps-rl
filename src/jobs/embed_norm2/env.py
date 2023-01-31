from collections import defaultdict
from poke_env.player import Gen8EnvSinglePlayer, ObservationType, SimpleHeuristicsPlayer
from poke_env import PlayerConfiguration
from poke_env.environment import Gen8Battle
from gym.spaces import Space, Box
import random, string
import numpy as np
from lib.constants import GEN8_MOVE_ID

from lib.env import VsHeuristicEnv
from lib.utils import infx


def id(size):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(size))


class Env(VsHeuristicEnv):
    def calc_reward(self, _, current_battle: Gen8Battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=1, victory_value=3
        )

    def embed_battle(self, battle: Gen8Battle) -> ObservationType:

        move_choices = battle.active_pokemon.moves.values()
        team = battle.team.values()
        opp_team = battle.opponent_team.values()

        move_id = np.zeros(4)
        multiplier = np.ones(4)

        for i, move in enumerate(move_choices):
            move_id[i] = GEN8_MOVE_ID[move.id]
            multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        n_fainted = sum(mon.fainted for mon in team) / 6
        n_opp_fainted = sum(mon.fainted for mon in opp_team) / 6

        has_status = battle.active_pokemon.status is not None
        opp_has_status = battle.opponent_active_pokemon.status is not None

        return np.concatenate(
            [
                move_id,
                multiplier,
                [n_fainted, n_opp_fainted, has_status, opp_has_status],
            ]
        )

    def describe_embedding(self) -> Space:
        return Box(-np.inf, np.inf, (12,))
