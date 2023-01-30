from ray.rllib.algorithms.callbacks import DefaultCallbacks
from poke_env.environment import Gen8Battle
from ray.tune.logger import UnifiedLogger
from ray.rllib.models import ModelCatalog
import pathlib
import shutil


class WinRateCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, env_index):
        env = base_env.get_sub_environments()[0]
        current_battle: Gen8Battle = env.current_battle

        assert env.current_battle.won is not None

        episode.custom_metrics["win"] = int(env.current_battle.won)

        ally_alive = 6 - sum(p.fainted for p in current_battle.team.values())
        opp_alive = 6 - sum(p.fainted for p in current_battle.opponent_team.values())
        episode.custom_metrics["alive_diff"] = ally_alive - opp_alive


def path_logger_creator(path):
    def logger_creator(config):
        return UnifiedLogger(config, str(path), loggers=None)

    return logger_creator


def find_checkpoints(path: pathlib.Path):
    checkpoints = [
        f for f in path.iterdir() if (f.is_dir() and f.name.startswith("checkpoint_"))
    ]
    checkpoints.sort(key=lambda f: int(f.name.removeprefix("checkpoint_")))

    return checkpoints


def prune_checkpoints(path: pathlib.Path):
    checkpoints = find_checkpoints(path)
    
    for checkpoint in checkpoints[:-1]:
        shutil.rmtree(checkpoint)


def apply_model(config, model):
    ModelCatalog.register_custom_model("placeholder_model", model)
    config.training(model=dict(custom_model="placeholder_model"))
