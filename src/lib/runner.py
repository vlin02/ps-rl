from pathlib import Path
import tempfile
from lib.ray import apply_model, find_checkpoints, path_logger_creator, prune_checkpoints
from ray.rllib.algorithms import AlgorithmConfig

from lib.utils import TermColors, infx

OUT_DIR = Path("..") / "out"
RESULT_DIR = OUT_DIR / "result"
TMP_DIR = OUT_DIR / "tmp"

class Runner:
    def __init__(self):
        self.data_dir = OUT_DIR
        self.result_dir = RESULT_DIR
        self.tmp_dir = TMP_DIR

        for dir in [self.data_dir, self.result_dir, self.tmp_dir]:
            dir.mkdir(exist_ok=True)

    def run_job(self, conf: AlgorithmConfig, job_name: str, save_freq=5, cycles=120):

        job_dir = self.result_dir / job_name

        conf.framework("torch")

        job_dir.mkdir(exist_ok=True, parents=True)
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
        job_dir = Path(tempfile.mkdtemp(dir= str(self.tmp_dir)))

        conf.framework("torch")
        conf.training(train_batch_size=256)
        conf.rollouts(num_rollout_workers=1)
        
        job_dir.mkdir(exist_ok=True, parents=True)
        conf.debugging(logger_creator=path_logger_creator(job_dir))

        infx(job_dir.absolute())
        algo = conf.build(use_copy=False)
        algo.train()


def apply_job(config, job) -> AlgorithmConfig:
    config.environment(env=job.Env)

    if job.Model:
        apply_model(config, job.Model)
    
    return config
