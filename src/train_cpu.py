from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from lib.ray import WinRateCallback, apply_model
from lib.runner import Runner, apply_job


def apply_base(config: AlgorithmConfig):
    (
        config.rollouts(num_rollout_workers=8)
        .callbacks(WinRateCallback)
        .resources(num_gpus=0)
    )


from jobs import (
    maxpool_heuristic,
    type_embed,
    baseline_base_power,
    embed_norm2,
    embed_scale_freq,
)


def init_config(config, job):
    return (
        apply_job(config, job)
        .rollouts(num_rollout_workers=8)
        .callbacks(WinRateCallback)
        .resources(num_gpus=0)
        .framework("torch")
    )

runner = Runner()

runner.test_job(init_config(PPOConfig(), embed_norm2))
# runner.run_job(init_config(PPOConfig(), embed_scale_freq), "embed_scale_freq", cycles=150)
