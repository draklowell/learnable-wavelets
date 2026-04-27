import multiprocessing as mp
import queue
from collections import deque

RUNNER_CONFIG = {
    "project_name": "lw-runs",
    "learning_rate": 0.001,
    "batch_size": 64,
    "delta": 0.00001,  # needs tuning
    "patience": 5,  # needs tuning
    "max_epochs": 10,
    "patch_size": 128,
    "val_size": 6000,
    "val_batch_size": 192,
    "device": "cuda",
    "log_interval": 50,
    "val_interval": 1000,
    "split_seed": 42,
    "dataset": "/kaggle/input/datasets/draklowell/",
    "loader_num_workers": 3,
}
SA_CONFIG = {
    "project_name": "lw-master",
    "max_score": 0.25,
    "max_depth": 6,
    "new_wavelet_prob": 0.15,
    "support_sizes": [2, 4, 6, 8, 10, 12, 14, 16],
    "random_seed": 42,
    "initial_config": {
        "wavelets": [
            {
                "name": "wv1",
                "support_size": 6,
            }
        ],
        "tree": {
            "wavelet": "wv1",
            "padding": "reflect",
            "hh": "drop",
            "hl": "drop",
            "lh": "drop",
            "ll": "keep",
        },
    },
    "initial_temperature": 1.0,
    "cooling_rate": 0.95,
    "final_temperature": 0.01,
    "batches_per_temperature": 1,
    "batch_size": 6,
}


class Manager:
    def __init__(self, config: dict, gpus: tuple[int, ...] = (0, 1)):
        self.config = config
        self.gpus = gpus
        self.cache: dict[str, float] = {}

        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()

        self.job_queues: dict[int, mp.Queue] = {}
        self.processes: dict[int, mp.Process] = {}
        self.inflight_by_worker: dict[int, tuple[int, int, str]] = {}

        self.next_job_id = 0

        for worker_id, gpu_id in enumerate(self.gpus):
            self._start_worker(worker_id, gpu_id)

    def log(self, message: str) -> None:
        print(f"[Manager] {message}", flush=True)

    def _start_worker(self, worker_id: int, gpu_id: int) -> None:
        from learnable_wavelets.mp import trainer_worker

        job_queue = self.ctx.Queue(maxsize=1)

        process = self.ctx.Process(
            target=trainer_worker,
            args=(
                worker_id,
                gpu_id,
                self.config,
                job_queue,
                self.result_queue,
            ),
            daemon=False,
        )

        process.start()

        self.job_queues[worker_id] = job_queue
        self.processes[worker_id] = process

        self.log(f"Started trainer {worker_id} on GPU {gpu_id}")

    def _restart_worker(self, worker_id: int) -> None:
        old_process = self.processes.get(worker_id)

        if old_process is not None and old_process.is_alive():
            old_process.terminate()
            old_process.join(timeout=5)

        gpu_id = self.gpus[worker_id]
        self.log(f"Restarting trainer {worker_id} on GPU {gpu_id}")
        self._start_worker(worker_id, gpu_id)

    def _new_job_id(self) -> int:
        job_id = self.next_job_id
        self.next_job_id += 1
        return job_id

    def eval(self, batch: list[tuple[dict, str]]) -> list[float]:
        results: list[float | None] = [None] * len(batch)
        work_queue = deque()

        for idx, (tree, hash_) in enumerate(batch):
            if hash_ in self.cache:
                results[idx] = self.cache[hash_]
            else:
                work_queue.append((idx, tree, hash_))

        local_jobs: dict[int, tuple[int, int, str]] = {}

        def submit_next(worker_id: int) -> bool:
            if not work_queue:
                return False

            process = self.processes[worker_id]

            if not process.is_alive():
                self._restart_worker(worker_id)

            idx, tree, hash_ = work_queue.popleft()
            job_id = self._new_job_id()

            self.job_queues[worker_id].put((job_id, idx, tree, hash_))

            local_jobs[job_id] = (worker_id, idx, hash_)
            self.inflight_by_worker[worker_id] = (job_id, idx, hash_)

            return True

        idle_workers = deque(range(len(self.gpus)))

        while idle_workers and work_queue:
            submit_next(idle_workers.popleft())

        while local_jobs:
            try:
                worker_id, job_id, idx, hash_, value, error = self.result_queue.get(
                    timeout=5
                )

            except queue.Empty:
                # Detect hard trainer crashes, e.g. CUDA OOM killing the process.
                for worker_id, process in list(self.processes.items()):
                    if process.is_alive():
                        continue

                    inflight = self.inflight_by_worker.pop(worker_id, None)

                    if inflight is None:
                        self._restart_worker(worker_id)
                        continue

                    job_id, idx, hash_ = inflight

                    if job_id in local_jobs:
                        self.log(
                            f"Trainer {worker_id} died while evaluating hash {hash_}"
                        )

                        value = float("inf")
                        self.cache[hash_] = value
                        results[idx] = value

                        local_jobs.pop(job_id, None)

                    self._restart_worker(worker_id)
                    submit_next(worker_id)

                continue

            if job_id not in local_jobs:
                # Stale result from a previous/restarted worker.
                continue

            local_jobs.pop(job_id, None)
            self.inflight_by_worker.pop(worker_id, None)

            if error is not None:
                self.log(f"Trainer {worker_id} failed on hash {hash_}:\n{error}")

            if value is None:
                self.log(f"Trainer {worker_id} returned None for hash {hash_}")
                value = float("inf")

            value = float(value)

            self.cache[hash_] = value
            results[idx] = value

            submit_next(worker_id)

        return [float(x) for x in results]

    def close(self) -> None:
        for worker_id, job_queue in self.job_queues.items():
            try:
                job_queue.put(None)
            except Exception:
                pass

        for worker_id, process in self.processes.items():
            process.join(timeout=10)

            if process.is_alive():
                self.log(f"Force terminating trainer {worker_id}")
                process.terminate()
                process.join(timeout=5)


def main() -> None:
    from learnable_wavelets.sa_manager import SAManager

    manager = Manager(RUNNER_CONFIG, gpus=(0, 1))

    try:
        with SAManager(SA_CONFIG, objective=manager.eval) as sa_manager:
            sa_manager.start()

    except KeyboardInterrupt:
        pass

    finally:
        manager.close()


if __name__ == "__main__":
    main()
