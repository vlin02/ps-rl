from pathlib import Path
import tempfile
from lib.ray import find_checkpoints, path_logger_creator, prune_checkpoints
from ray.rllib.algorithms import AlgorithmConfig

from lib.utils import TermColors, infx


class Runner:
    def __init__(self):
        self.result_dir = Path("..") / "result"
        self.tmpdir = Path("..") / "result_tmp"

    def run_job(self, conf: AlgorithmConfig, job_name, save_freq=5, cycles=120):

        job_dir = self.result_dir / job_name

        conf.framework("torch")
        conf.debugging(logger_creator=path_logger_creator(job_dir))

        algo = conf.build(use_copy=False)

        existing_checkpoints = find_checkpoints(job_dir)
        
        if len(existing_checkpoints) > 0:
            algo.restore(existing_checkpoints[-1])

        for i in range(cycles):
            result = algo.train()

            infx("episode_reward_mean:", result["episode_reward_mean"], color=TermColors.OKBLUE)
            infx("time_this_iter_s:", result["time_this_iter_s"], color=TermColors.OKBLUE)
            infx()

            if (i + 1) % save_freq == 0:
                checkpoint_dir = algo.save()
                infx(f"Checkpoint saved in directory {checkpoint_dir}", color=TermColors.OKGREEN)
                infx()

            prune_checkpoints(job_dir)

    def test_job(self, conf: AlgorithmConfig):
        dir = Path(tempfile.mkdtemp(dir= str(self.tmpdir)))
        conf.framework("torch")
        conf.training(train_batch_size=256)
        conf.rollouts(num_rollout_workers=1)
        conf.debugging(
            logger_creator=path_logger_creator(
                dir
            )
        )

        infx(dir.name)
        algo = conf.build(use_copy=False)
        algo.train()
