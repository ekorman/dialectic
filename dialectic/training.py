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
    val_fn: Callable[[Env], tuple[EvaluationResult, list[Example]]] | None = None,
    val_envs: Sequence[Env] | None = None,
    val_dataset_fn: Callable[[], tuple[EvaluationResult, list[Example]]] | None = None,
    val_dataset_label: str = "val",
    start_step: int = 0,
    save_ckpt_steps: set[int] | None = None,
):
    """Generic training loop with two interchangeable val regimes.

    Pick exactly one of:

    - ``val_envs`` + ``val_fn``: RL-style. ``train_loop`` iterates the envs,
      calls ``env.reseed()`` before each, invokes ``val_fn(env)``, and
      namespaces metrics as ``val/<env_label>/...``. Use this when val has
      genuine env semantics (multiple held-out envs, env-driven reseed,
      cross-env aggregation).
    - ``val_dataset_fn``: dataset-style. ``train_loop`` calls it with no
      arguments once per val invocation and namespaces metrics as
      ``val/<val_dataset_label>/...``. Use this for SFT-flavored training
      where there's a single held-out dataset and the val function reads
      from a closure.

    Pass none of them and val is skipped (e.g., when ``val_freq <= 0``).
    """
    if val_fn is not None and val_dataset_fn is not None:
        raise ValueError("Pass either (val_fn + val_envs) or val_dataset_fn, not both")
    if val_fn is not None and val_envs is None:
        raise ValueError("val_fn requires val_envs to be provided")

    def _should_save_ckpt(s: int) -> bool:
        # An explicit step schedule, when given, fully overrides the periodic
        # ``save_ckpt_freq`` cadence.
        if save_ckpt_steps is not None:
            return s in save_ckpt_steps
        return s % save_ckpt_freq == 0

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

                if _should_save_ckpt(step):
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
            has_val = val_fn is not None or val_dataset_fn is not None
            if is_main_process() and should_val and has_val:
                log.info(f"Running evaluation at step {step}")
                raw_net = unwrap_model(net)
                was_training = raw_net.training
                raw_net.eval()
                val_start = time.perf_counter()
                if val_dataset_fn is not None:
                    val_metrics = run_dataset_validation(
                        val_dataset_fn=val_dataset_fn, label=val_dataset_label
                    )
                else:
                    assert val_fn is not None and val_envs is not None
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

    if is_main_process() and not _should_save_ckpt(step) and extty.has_active_run():
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
    """Multi-env (RL-style) validation. Iterates envs, reseeds each, calls
    ``val_fn`` per env, and aggregates with per-env metric namespaces."""
    metrics: dict[str, Any] = {}
    val_means: list[float] = []

    for env in val_envs:
        env.reseed()
        label = str(env)
        result, examples = val_fn(env)
        _populate_metrics(metrics, label=label, result=result, examples=examples)
        val_means.append(result.reward_mean)

    if val_means:
        metrics["val/mean"] = sum(val_means) / len(val_means)

    return metrics


@torch.no_grad()
def run_dataset_validation(
    *,
    val_dataset_fn: Callable[[], tuple[EvaluationResult, list[Example]]],
    label: str,
) -> dict[str, Any]:
    """Single-call (SFT-style) validation. Invokes ``val_dataset_fn`` once
    and namespaces its metrics under ``val/<label>/...``."""
    metrics: dict[str, Any] = {}
    result, examples = val_dataset_fn()
    _populate_metrics(metrics, label=label, result=result, examples=examples)
    metrics["val/mean"] = result.reward_mean
    return metrics


def _populate_metrics(
    metrics: dict[str, Any],
    *,
    label: str,
    result: EvaluationResult,
    examples: list[Example],
) -> None:
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
