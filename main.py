import time
from collections import deque

import ray
import yaml

from learnable_wavelets.config import ModuleConfig
from learnable_wavelets.run import Runner
from learnable_wavelets.sa_manager import SAManager

ray.init()


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, config: dict):
        self.runner = Runner(config)

    def run(self, tree: dict) -> float | None:
        return self.runner.run(ModuleConfig.model_validate(tree))


class Manager:
    def __init__(self, config: dict):
        self.config = config
        self.trainers = []
        self.cache: dict[str, float] = {}

    def rescale(self):
        resources = ray.cluster_resources()
        gpus = int(resources.get("GPU", 0))

        for _ in range(gpus - len(self.trainers)):
            self.trainers.append(Trainer.remote(self.config))

    def ensure_trainers(self) -> None:
        self.rescale()

        while not self.trainers:
            self.log("No trainers available. Waiting...")
            time.sleep(5)
            self.rescale()

    def log(self, message: str):
        print(f"[Manager] {message}")

    def eval(self, batch: list[tuple[dict, str]]) -> list[float]:
        self.ensure_trainers()

        results = [None] * len(batch)
        work_queue = deque()

        for idx, (tree, hash_) in enumerate(batch):
            if hash_ in self.cache:
                results[idx] = self.cache[hash_]
            else:
                work_queue.append((idx, tree, hash_))

        running = {}

        idle_trainers = deque(self.trainers)

        def submit_next(trainer):
            if not work_queue:
                return False

            idx, tree, hash_ = work_queue.popleft()
            ref = trainer.run.remote(tree)
            running[ref] = (idx, hash_, trainer)
            return True

        while idle_trainers and work_queue:
            trainer = idle_trainers.popleft()
            submit_next(trainer)

        while running:
            done_refs, _ = ray.wait(list(running.keys()), num_returns=1)

            for ref in done_refs:
                idx, hash_, trainer = running.pop(ref)

                value = None
                try:
                    value = ray.get(ref)
                except Exception as e:
                    self.log(f"Trainer {trainer} failed with error: {e}")

                if value is None:
                    self.log(f"Trainer {trainer} returned None for hash {hash_}")
                    value = float("inf")

                self.cache[hash_] = value
                results[idx] = value

                submit_next(trainer)

        return [float(x) for x in results]


RUNNER_CONFIG = "configs/runner.yaml"
SA_CONFIG = "configs/simulated_annealing.yaml"

if __name__ == "__main__":
    with open(RUNNER_CONFIG) as f:
        runner_config = yaml.safe_load(f)

    with open(SA_CONFIG) as f:
        sa_config = yaml.safe_load(f)

    manager = Manager(runner_config)
    try:
        with SAManager(sa_config, objective=manager.eval) as sa_manager:
            sa_manager.start()
    except KeyboardInterrupt:
        pass
