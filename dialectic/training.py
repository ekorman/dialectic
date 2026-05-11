import random
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import extty
import torch
from extty import Example

from dialectic.distributed import barrier, is_main_process, unwrap_model
from dialectic.llm.base import BaseTransformer
from dialectic.log import log
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult

_sigterm_received = False


def _sigterm_handler(signum, frame):
    global _sigterm_received
    _sigterm_received = True
    log.info("SIGTERM received, will save checkpoint and exit after current step")


@dataclass
class StepFunctionReturn:
    n_episodes_processed: int
    metrics: dict[str, float | int]


def train_loop(
    max_episodes: int,
    save_ckpt_freq: int,
    val_freq: int,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    train_step: Callable[[int], StepFunctionReturn],
    val_fn: Callable[[Env], tuple[EvaluationResult, list[Example]]],
    val_envs: Sequence[Env],
    start_step: int = 0,
):
    global _sigterm_received
    _sigterm_received = False
    prev_handler = signal.signal(signal.SIGTERM, _sigterm_handler)

    n_episodes = 0
    step = start_step
    try:
        while n_episodes < max_episodes:
            start_time = time.perf_counter()
            step_ret = train_step(step)
            step_time = time.perf_counter() - start_time
            step += 1

            n_episodes += step_ret.n_episodes_processed

            if is_main_process() and extty.has_active_run():
                metrics = step_ret.metrics
                metrics.update({"step_time": step_time})
                extty.log(metrics, step=step)

                if step % save_ckpt_freq == 0:
                    ckpt_state = unwrap_model(net).state_dict()
                    ckpt_state["_rng_torch"] = torch.random.get_rng_state()
                    ckpt_state["_rng_python"] = random.getstate()
                    if torch.cuda.is_available():
                        ckpt_state["_rng_cuda"] = torch.cuda.get_rng_state()
                    extty.save_checkpoint(
                        step=step,
                        state_dict=ckpt_state,
                        optimizer_state_dict=opt.state_dict(),
                    )
            should_val = (
                val_freq > 0 and step % val_freq == 0 or n_episodes >= max_episodes
            )
            if is_main_process() and should_val:
                log.info(f"Running evaluation at step {step}")
                raw_net = unwrap_model(net)
                was_training = raw_net.training
                raw_net.eval()
                val_start = time.perf_counter()
                val_metrics = run_validation(val_envs=val_envs, val_fn=val_fn)
                val_time = time.perf_counter() - val_start
                if was_training:
                    raw_net.train()
                val_metrics["val/time"] = val_time
                extty.log(val_metrics, step=step)
                log.info(f"Finished evaluation at step {step}")
            barrier()

            if _sigterm_received:
                log.info(f"Saving checkpoint at step {step} before exit")
                break
    finally:
        signal.signal(signal.SIGTERM, prev_handler)

    if is_main_process() and step % save_ckpt_freq != 0 and extty.has_active_run():
        ckpt_state = unwrap_model(net).state_dict()
        ckpt_state["_rng_torch"] = torch.random.get_rng_state()
        ckpt_state["_rng_python"] = random.getstate()
        if torch.cuda.is_available():
            ckpt_state["_rng_cuda"] = torch.cuda.get_rng_state()
        extty.save_checkpoint(
            step=step,
            state_dict=ckpt_state,
            optimizer_state_dict=opt.state_dict(),
        )


@torch.no_grad()
def run_validation(
    *,
    val_envs: Sequence[Env],
    val_fn: Callable[[Env], tuple[EvaluationResult, list[Example]]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    val_means: list[float] = []

    for env in val_envs:
        env.reseed()
        label = str(env)
        result, examples = val_fn(env)

        metrics[f"val/{label}/mean"] = result.reward_mean
        metrics[f"val/{label}/std"] = result.reward_std
        for comp_name, comp_val in result.component_means.items():
            metrics[f"val/{label}/{comp_name}"] = comp_val
        if examples:
            metrics[f"val/{label}/example"] = extty.BatchExample(
                prompts=[e.prompt for e in examples],
                responses=[e.responses for e in examples],
                rewards=[e.rewards for e in examples],
            )
        val_means.append(result.reward_mean)

    if val_means:
        metrics["val/mean"] = sum(val_means) / len(val_means)

    return metrics
