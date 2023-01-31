
from pathlib import Path
import tempfile
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from lib.ray import path_logger_creator
from lib.runner import Runner, apply_job
from jobs import maxpool_only
from lib.utils import infx, time_execution

algo_config = apply_job(PPOConfig(), maxpool_only)

runner = Runner()

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from lib.runner import Runner, apply_job
from jobs import maxpool_only


algo_config = apply_job(PPOConfig(), maxpool_only)

algo_config.framework("torch")
algo_config.rollouts(num_rollout_workers=16)

tmp_dir = Path(tempfile.mkdtemp(dir= str('../out/tmp')))
infx(tmp_dir)
algo_config.debugging(logger_creator=path_logger_creator(tmp_dir))

algo = algo_config.build(use_copy=False)

import json

def main():
    result = None
    for _ in range(1):
        result = algo.train()

    print(json.dumps(result, default=lambda o: '<not serializable>'))


time_execution(main)