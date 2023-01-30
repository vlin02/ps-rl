from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from lib.ray import WinRateCallback, apply_model
from lib.runner import Runner


def apply_base(config: AlgorithmConfig):
    (
        config.rollouts(num_rollout_workers=8)
        .callbacks(WinRateCallback)
        .resources(num_gpus=0)
    )


from jobs import maxpool_heuristic, type_embed, baseline_base_power

runner = Runner()


def get_base_algo(conf_cls, mod) -> AlgorithmConfig:
    conf = (
        conf_cls()
        .environment(env=mod.Env)
        .rollouts(num_rollout_workers=8)
        .callbacks(WinRateCallback)
        .resources(num_gpus=0)
    )

    if mod.Model:
        apply_model(conf, mod.Model)
    return conf


def get_algo():
    return get_base_algo(PPOConfig, maxpool_heuristic).training()


runner.test_job(get_algo())
if True:
    runner.run_job(get_algo(), "maxpool_heuristic", cycles=600)
