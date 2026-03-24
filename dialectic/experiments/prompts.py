from dataclasses import dataclass

REASONING_TAG = "reasoning"


# system prompt from Soft Tokens Hard Truths paper (but using answer tags instead of boxed)
STHT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves"
    " it. The assistant first shows the complete reasoning process step by step, then provides the final"
    " answer in <answer></answer> tags. The assistant must always follow the format: 'User: [question] Assistant:"
    " [detailed reasoning] The final answer is: <answer>[answer]</answer>.'"
)
SIMPLE_SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked a question to solve you first show your complete "
    "reasoning process step by step and then provide the user with the answer in the specified format."
)

ENV_PROMPT_WITH_REASONING_TAGS = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number exactly once. "
    f"Show your reasoning in <{REASONING_TAG}></{REASONING_TAG}> tags."
    " Put your final equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
ENV_PROMPT_WITHOUT_REASONING_TAGS = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number exactly once. Put your final"
    " equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>. "
)
ENV_PROMPT_WITH_SCRATCH_TAGS = """Solve the Countdown problem by combining numbers to reach the target.

Rules:
- Combine two numbers at a time using +, -, *, /
- Use <SCRATCH> tags for intermediate calculations
- Numbers from scratch results can be used in later steps
- Give your final answer in <answer> tags using all original numbers

Example:
Numbers: [5, 3, 2], Target: 13
<SCRATCH> 5 + 2 = 7 </SCRATCH>
<SCRATCH> 7 + 3 = 10 </SCRATCH>
<SCRATCH> 5 * 3 = 15 </SCRATCH>
<SCRATCH> 15 - 2 = 13 </SCRATCH>
<answer> 5 * 3 - 2 </answer>

Now solve:
Numbers: {numbers}, Target: {target}"""

ENV_PROMPT_COUNTDOWN_MINIMAL = (
    "Combine numbers to reach the target using +, -, *, /.\n\n"
    "Numbers: {numbers}, Target: {target}"
)


@dataclass
class PromptCollection:
    system_prompt: str | None
    env_prompt: str
    assistant_prefill: str | None


MAZE_INTERNAL_REASONING_PROMPT = PromptCollection(
    system_prompt=None,
    env_prompt=(
        "Navigate the maze from Start to Goal. "
        "Each line shows a cell and the directions you can move from it.\n\n"
        "{maze}"
    ),
    assistant_prefill=None,
)

COUNTDOWN_INTERNAL_REASONING_PROMPT = PromptCollection(
    system_prompt=None,
    env_prompt=(
        "Using the numbers {numbers}, reach the target {target}. "
        "Show each computation step."
    ),
    assistant_prefill=None,
)

PROMPT_COLLECTIONS: dict[str, list[PromptCollection]] = {
    "countdown": [
        PromptCollection(
            system_prompt=None,
            env_prompt=ENV_PROMPT_WITH_REASONING_TAGS,
            assistant_prefill=f"Let me solve this step by step\n<{REASONING_TAG}>",
        ),
        PromptCollection(
            system_prompt=STHT_SYSTEM_PROMPT,
            env_prompt=ENV_PROMPT_WITHOUT_REASONING_TAGS,
            assistant_prefill=None,
        ),
        PromptCollection(
            system_prompt=SIMPLE_SYSTEM_PROMPT,
            env_prompt=ENV_PROMPT_WITHOUT_REASONING_TAGS,
            assistant_prefill="Let me solve this step by step.",
        ),
        PromptCollection(
            system_prompt=SIMPLE_SYSTEM_PROMPT,
            env_prompt=ENV_PROMPT_WITH_SCRATCH_TAGS,
            assistant_prefill="Let me solve this step by step.",
        ),
    ],
    "maze": [
        PromptCollection(
            system_prompt=None,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                f"Show your reasoning in <{REASONING_TAG}></{REASONING_TAG}> tags. "
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill=f"Let me solve this step by step\n<{REASONING_TAG}>",
        ),
        PromptCollection(
            system_prompt=STHT_SYSTEM_PROMPT,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill=None,
        ),
        PromptCollection(
            system_prompt=SIMPLE_SYSTEM_PROMPT,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill="Let me solve this step by step.",
        ),
    ],
    "math": [
        PromptCollection(
            system_prompt="Solve the math problem. Respond with only the numerical answer.",
            env_prompt="",
            assistant_prefill=None,
        ),
    ],
}
