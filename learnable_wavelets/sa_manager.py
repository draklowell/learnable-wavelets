import json
import random
import tempfile
from typing import Callable

import wandb
from learnable_wavelets.config import ModuleConfig, load_config
from learnable_wavelets.simulated_annealing import SimulatedAnnealing


class SAManager:
    def __init__(
        self, config: dict, objective: Callable[[list[tuple[dict, str]]], list[float]]
    ):
        self.config = config
        self.rng = random.Random(config.get("random_seed", 42))

        if "initial_config_path" in config:
            initial_config = load_config(config["initial_config_path"])
        else:
            initial_config = ModuleConfig.model_validate(config["initial_config"])

        self.sa = SimulatedAnnealing(
            initial_config=initial_config,
            objective=objective,
            max_score=config["max_score"],
            max_depth=config.get("max_depth", 10),
            new_wavelet_prob=config.get("new_wavelet_prob", 0.15),
            support_sizes=config.get("support_sizes"),
            initial_temperature=config.get("initial_temperature", 1.0),
            final_temperature=config.get("final_temperature", 0.01),
            cooling_rate=config.get("cooling_rate", 0.95),
            batches_per_temperature=config.get("batches_per_temperature", 100),
            batch_size=config.get("batch_size", 4),
            rng=self.rng,
            on_batch_complete=self._on_batch_complete,
        )
        self.run = None

    def __enter__(self):
        self.run = wandb.init(project=self.config["project_name"], config=self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run is not None:
            self.run.finish()

    def _on_batch_complete(self, batch_results: dict):
        if self.run is not None:
            self.run.log(batch_results)

    def start(self):
        if self.run is None:
            raise RuntimeError("Must be used within a 'with' block")

        best_config, score = self.sa.run()
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            json.dump({"config": best_config, "score": score}, tmp)
            config_artifact = wandb.Artifact("best_config", type="config")
            config_artifact.add_file(tmp.name)
            self.run.log_artifact(config_artifact)
