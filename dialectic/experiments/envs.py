import json
from typing import Callable

import extty

from dialectic.llm.templates import Message
from dialectic.log import log
from dialectic.rl.env import Countdown, MathState
from dialectic.rl.types import EnvResponse


def get_state_to_str(
    *,
    format_messages: Callable[[list[Message], bool], str],
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
):
    def _state_to_str(data: Countdown | MathState) -> str:
        msgs = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        ret = format_messages(msgs, True)
        if assistant_prefill:
            ret += assistant_prefill
        return ret

    return _state_to_str


def load_countdown_dataset_artifacts(
    artifact_names: list[str],
    prompt_template: str,
) -> list[tuple[EnvResponse[Countdown], dict]]:
    """Load countdown problems from one or more dataset artifacts.

    Returns list of (env_response, extra_fields) tuples, where extra_fields
    contains passthrough fields like 'split' and 'equation'.
    """
    all_problems: list[tuple[EnvResponse[Countdown], dict]] = []
    for name in artifact_names:
        data = extty.load_artifact(name, cache=True)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")
        count = 0
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            numbers = entry["numbers"]
            target = entry["target"]
            prompt = prompt_template.format(numbers=numbers, target=target)
            extra: dict = {}
            for key in ("split", "equation"):
                if key in entry:
                    extra[key] = entry[key]
            all_problems.append(
                (
                    EnvResponse(
                        is_done=True,
                        data=Countdown(
                            prompt=prompt,
                            numbers=numbers,
                            target=target,
                            solution=None,
                        ),
                    ),
                    extra,
                )
            )
            count += 1
        log.info(f"Loaded {count} problems from artifact '{name}'")
    log.info(
        f"Total: {len(all_problems)} problems from {len(artifact_names)} artifact(s)"
    )
    return all_problems
