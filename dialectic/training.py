import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import extty
import torch
from extty import Example

from dialectic.llm.base import BaseTransformer
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult


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
):
    n_episodes = 0
    step = 0
    while n_episodes < max_episodes:
        start_time = time.perf_counter()
        step_ret = train_step(step)
        step_time = time.perf_counter() - start_time
        step += 1

        n_episodes += step_ret.n_episodes_processed

        if extty.has_active_run():
            metrics = step_ret.metrics
            metrics.update({"step_time": step_time})
            extty.log(metrics, step=step)

            if step % save_ckpt_freq == 0:
                extty.save_checkpoint(
                    step=step,
                    state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
            if val_freq > 0 and step % val_freq == 0 or n_episodes >= max_episodes:
                was_training = net.training
                net.eval()
                val_metrics = run_validation(val_envs=val_envs, val_fn=val_fn)
                if was_training:
                    net.train()
                extty.log(val_metrics, step=step)

    if step % save_ckpt_freq != 0 and extty.has_active_run():
        extty.save_checkpoint(
            step=step,
            state_dict=net.state_dict(),
            optimizer_state_dict=opt.state_dict(),
        )


@torch.no_grad()
def run_validation(
    *,
    val_envs: Sequence[Env],
    val_fn: Callable[[Env], tuple[EvaluationResult, list[Example]]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    reward_means: list[float] = []

    for env in val_envs:
        env.reseed()
        label = str(env)
        result, examples = val_fn(env)

        metrics[f"val/{label}/reward_mean"] = result.reward_mean
        metrics[f"val/{label}/reward_std"] = result.reward_std
        for comp_name, comp_val in result.component_means.items():
            metrics[f"val/{label}/reward/{comp_name}"] = comp_val
        if examples:
            metrics[f"val/{label}/example"] = extty.BatchExample(
                prompts=[e.prompt for e in examples],
                responses=[e.responses for e in examples],
                rewards=[e.rewards for e in examples],
            )
        reward_means.append(result.reward_mean)

    if reward_means:
        metrics["val/reward_mean"] = sum(reward_means) / len(reward_means)

    return metrics
