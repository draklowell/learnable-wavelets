import multiprocessing as mp
import os
import traceback


def trainer_worker(
    worker_id: int,
    gpu_id: int,
    config: dict,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import after setting CUDA_VISIBLE_DEVICES.
    from learnable_wavelets.config import ModuleConfig
    from learnable_wavelets.run import Runner

    runner = Runner(config)

    while True:
        job = job_queue.get()

        if job is None:
            break

        job_id, idx, tree, hash_ = job

        try:
            value = runner.run(ModuleConfig.model_validate(tree))

            if value is None:
                value = float("inf")
            else:
                value = float(value)

            result_queue.put((worker_id, job_id, idx, hash_, value, None))

        except Exception:
            result_queue.put(
                (
                    worker_id,
                    job_id,
                    idx,
                    hash_,
                    float("inf"),
                    traceback.format_exc(),
                )
            )
