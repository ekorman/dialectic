import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    name: str
    arguments: str | dict[str, Any]

    @property
    def function(self) -> "ToolCall":
        return self


@dataclass
class ToolCallWrapper:
    function: ToolCall


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall | ToolCallWrapper] | None = None
    reasoning_content: str | None = None


@dataclass
class Tool:
    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


def get_input_text_from_messages(
    messages: list[Message],
    add_generation_prompt: bool,
    tools: list[Tool | dict[str, Any]] | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """
    Convert messages to the Qwen chat template format.

    Parameters
    ----------
    messages : list[Message]
        List of Message objects representing the conversation.
    add_generation_prompt : bool
        Whether to add the assistant generation prompt at the end.
    tools : list[Tool | dict[str, Any]] | None, optional
        List of tool definitions to include in the system prompt.
    enable_thinking : bool | None, optional
        Controls thinking mode. None uses default behavior, False explicitly
        disables thinking by adding an empty think block.

    Returns
    -------
    str
        The formatted chat template string.
    """
    result = ""

    if not messages:
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
            if enable_thinking is False:
                result += "<think>\n\n</think>\n\n"
        return result

    if tools:
        result += "<|im_start|>system\n"
        if messages[0].role == "system":
            result += messages[0].content + "\n\n"
        result += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools:
            result += "\n"
            if isinstance(tool, Tool):
                result += json.dumps({"type": tool.type, "function": tool.function})
            else:
                result += json.dumps(tool)
        result += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
    else:
        if messages[0].role == "system":
            result += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n"

    last_query_index = len(messages) - 1
    multi_step_tool = True

    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if multi_step_tool and message.role == "user":
            content = message.content if isinstance(message.content, str) else ""
            if not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            ):
                multi_step_tool = False
                last_query_index = i

    for i, message in enumerate(messages):
        is_first = i == 0
        is_last = i == len(messages) - 1
        content = message.content if isinstance(message.content, str) else ""

        if message.role == "user" or (message.role == "system" and not is_first):
            result += f"<|im_start|>{message.role}\n{content}<|im_end|>\n"

        elif message.role == "assistant":
            reasoning_content = ""

            if isinstance(message.reasoning_content, str):
                reasoning_content = message.reasoning_content
            else:
                if "</think>" in content:
                    parts = content.split("</think>")
                    before_think = parts[0]
                    if "<think>" in before_think:
                        reasoning_content = (
                            before_think.split("<think>")[-1].lstrip("\n").rstrip("\n")
                        )
                    else:
                        reasoning_content = before_think.rstrip("\n")
                    content = parts[-1].lstrip("\n")

            if i > last_query_index:
                if is_last or (not is_last and reasoning_content):
                    result += f"<|im_start|>{message.role}\n<think>\n{reasoning_content.strip()}\n</think>\n\n{content.lstrip()}"
                else:
                    result += f"<|im_start|>{message.role}\n{content}"
            else:
                result += f"<|im_start|>{message.role}\n{content}"

            if message.tool_calls:
                for j, tool_call in enumerate(message.tool_calls):
                    if (j == 0 and content) or j > 0:
                        result += "\n"

                    actual_tool_call = tool_call
                    if hasattr(tool_call, "function") and isinstance(
                        tool_call.function, ToolCall
                    ):
                        actual_tool_call = tool_call.function

                    result += "<tool_call>\n"
                    result += f'{{"name": "{actual_tool_call.name}", "arguments": '

                    if isinstance(actual_tool_call.arguments, str):
                        result += actual_tool_call.arguments
                    else:
                        result += json.dumps(actual_tool_call.arguments)

                    result += "}\n</tool_call>"

            result += "<|im_end|>\n"

        elif message.role == "tool":
            prev_is_tool = i > 0 and messages[i - 1].role == "tool"
            next_is_tool = i < len(messages) - 1 and messages[i + 1].role == "tool"

            if is_first or not prev_is_tool:
                result += "<|im_start|>user"

            result += f"\n<tool_response>\n{content}\n</tool_response>"

            if is_last or not next_is_tool:
                result += "<|im_end|>\n"

    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
        if enable_thinking is False:
            result += "<think>\n\n</think>\n\n"

    return result
