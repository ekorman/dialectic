import os

import pytest
from tokenizers import Encoding, Tokenizer
from transformers import AutoTokenizer

from dialectic.llm.templates import (
    Message,
    ToolCall,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)


def get_auto_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_tokenizer_batch_right_pad():
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.enable_padding(pad_id=151643)
    tokens = tokenizer.encode_batch(
        ["short prompt", "this is a much longer prompt for testing"]
    )
    # short prompt has two tokens

    assert len(tokens[0]) == len(tokens[1]) == max([len(t) for t in tokens])
    assert tokens[0].ids[2:] == [151643] * (len(tokens[0].ids) - 2)
    for i in range(2):
        assert tokens[0].ids[i] != 151643


def test_tokenizer_batch_left_pad():
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.enable_padding(pad_id=151643, direction="left")
    tokens = tokenizer.encode_batch(
        ["short prompt", "this is a much longer prompt for testing"]
    )
    # short prompt has two tokens
    assert len(tokens[0]) == len(tokens[1]) == max([len(t) for t in tokens])
    assert tokens[0].ids[:-2] == [151643] * (len(tokens[0].ids) - 2)
    for i in range(2):
        assert tokens[0].ids[-i - 1] != 151643


def test_tokenizer_against_hf():
    """Test a simple user message."""
    auto_tokenizer = get_auto_tokenizer()
    tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    assert (
        text
        == get_qwen_input_text_from_messages(
            messages=[Message(**m) for m in messages],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        == "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
    )

    x: Encoding = tokenizer.encode_batch([text])
    y = auto_tokenizer([text], return_tensors="pt")

    assert y.tokens() == x[0].tokens
    assert y.input_ids.tolist()[0] == x[0].ids


def test_system_and_user_message():
    """Test system message followed by user message."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        enable_thinking=True,
    )

    assert result == expected


def test_multi_turn_conversation():
    """Test a multi-turn conversation with user and assistant."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And 3+3?"},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        enable_thinking=True,
    )

    assert result == expected


def test_enable_thinking_false():
    """Test with enable_thinking=False adds empty think block."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [{"role": "user", "content": "Hello"}]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert result == expected
    assert "<think>\n\n</think>" in result


def test_no_generation_prompt():
    """Test without generation prompt."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=False,
    )

    assert result == expected
    assert not result.endswith("<|im_start|>assistant\n")


def test_with_tools():
    """Test with tools parameter."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        tools=tools,
    )

    assert result == expected
    assert "<tools>" in result
    assert "get_weather" in result


def test_with_tools_and_system_message():
    """Test with tools and a system message."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        tools=tools,
    )

    assert result == expected
    # System message content should be included before tools
    assert "You are a weather assistant." in result
    assert "<tools>" in result


def test_assistant_with_tool_calls():
    """Test assistant message with tool calls."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    }
                }
            ],
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Convert to Message objects
    msg_objects = []
    for m in messages:
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = []
            for tc in m["tool_calls"]:
                func = tc["function"]
                tool_calls.append(
                    ToolCall(name=func["name"], arguments=func["arguments"])
                )
            msg_objects.append(
                Message(role=m["role"], content=m["content"], tool_calls=tool_calls)
            )
        else:
            msg_objects.append(Message(role=m["role"], content=m["content"]))

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
    )

    assert result == expected
    assert "<tool_call>" in result
    assert "get_weather" in result


def test_tool_response():
    """Test tool response message."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 22, "condition": "sunny"}'},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Convert to Message objects
    msg_objects = []
    for m in messages:
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = []
            for tc in m["tool_calls"]:
                func = tc["function"]
                tool_calls.append(
                    ToolCall(name=func["name"], arguments=func["arguments"])
                )
            msg_objects.append(
                Message(role=m["role"], content=m["content"], tool_calls=tool_calls)
            )
        else:
            msg_objects.append(Message(role=m["role"], content=m["content"]))

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=True,
    )

    assert result == expected
    assert "<tool_response>" in result


def test_multiple_tool_responses():
    """Test multiple consecutive tool responses."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What's the weather in Tokyo and London?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    }
                },
            ],
        },
        {"role": "tool", "content": '{"temperature": 22}'},
        {"role": "tool", "content": '{"temperature": 15}'},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Convert to Message objects
    msg_objects = []
    for m in messages:
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = []
            for tc in m["tool_calls"]:
                func = tc["function"]
                tool_calls.append(
                    ToolCall(name=func["name"], arguments=func["arguments"])
                )
            msg_objects.append(
                Message(role=m["role"], content=m["content"], tool_calls=tool_calls)
            )
        else:
            msg_objects.append(Message(role=m["role"], content=m["content"]))

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=True,
    )

    assert result == expected
    # Multiple tool responses should be grouped under one user block
    assert result.count("<tool_response>") == 2


def test_assistant_with_reasoning_content():
    """Test assistant message with explicit reasoning_content."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "Let me think... 2 plus 2 equals 4.",
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    msg_objects = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content="4",
            reasoning_content="Let me think... 2 plus 2 equals 4.",
        ),
    ]

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
    )

    assert result == expected


def test_assistant_with_think_tags_in_content():
    """Test assistant message with think tags embedded in content."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>\nLet me calculate...\n</think>\n4",
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    msg_objects = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="<think>\nLet me calculate...\n</think>\n4"),
    ]

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
    )

    assert result == expected


def test_system_message_not_first():
    """Test system message that is not the first message."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "system", "content": "New instructions."},
        {"role": "user", "content": "Continue"},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    result = get_qwen_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
    )

    assert result == expected


def test_tool_call_with_dict_arguments():
    """Test tool call where arguments is a dict instead of string."""
    auto_tokenizer = get_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                }
            ],
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    msg_objects = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="get_weather", arguments={"location": "Tokyo"})],
        ),
    ]

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
    )

    assert result == expected


def test_empty_messages():
    """Test with empty messages list."""
    result = get_qwen_input_text_from_messages(
        messages=[],
        add_generation_prompt=True,
    )
    assert result == "<|im_start|>assistant\n"


def test_empty_messages_no_generation():
    """Test with empty messages and no generation prompt."""
    result = get_qwen_input_text_from_messages(
        messages=[],
        add_generation_prompt=False,
    )
    assert result == ""


def test_full_tool_use_conversation():
    """Test a complete tool use conversation flow."""
    auto_tokenizer = get_auto_tokenizer()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 18, "condition": "cloudy"}'},
        {
            "role": "assistant",
            "content": "The weather in Paris is 18°C and cloudy.",
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )

    msg_objects = []
    for m in messages:
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = []
            for tc in m["tool_calls"]:
                func = tc["function"]
                tool_calls.append(
                    ToolCall(name=func["name"], arguments=func["arguments"])
                )
            msg_objects.append(
                Message(role=m["role"], content=m["content"], tool_calls=tool_calls)
            )
        else:
            msg_objects.append(Message(role=m["role"], content=m["content"]))

    result = get_qwen_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
        tools=tools,
    )

    assert result == expected


# ============================================================================
# Llama 3 Template Tests
# ============================================================================


def get_llama_auto_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_simple_user_message():
    """Test a simple user message."""
    auto_tokenizer = get_llama_auto_tokenizer()

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    date_string = "26 Jul 2024"

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "<|start_header_id|>assistant<|end_header_id|>" in result


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_system_and_user_message():
    """Test system message followed by user message."""
    auto_tokenizer = get_llama_auto_tokenizer()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    date_string = "26 Jul 2024"

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_multi_turn_conversation():
    """Test a multi-turn conversation with user and assistant."""
    auto_tokenizer = get_llama_auto_tokenizer()

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And 3+3?"},
    ]
    date_string = "26 Jul 2024"

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_no_generation_prompt():
    """Test without generation prompt."""
    auto_tokenizer = get_llama_auto_tokenizer()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    date_string = "26 Jul 2024"

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        date_string=date_string,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=False,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected
    assert not result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_with_tools_in_user_message():
    """Test with tools parameter (tools in user message by default)."""
    auto_tokenizer = get_llama_auto_tokenizer()

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    date_string = "26 Jul 2024"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        date_string=date_string,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        tools=tools,
        tools_in_user_message=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected
    assert "Environment: ipython" in result
    assert "get_weather" in result


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_with_tools_in_system_message():
    """Test with tools in system message."""
    auto_tokenizer = get_llama_auto_tokenizer()

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    date_string = "26 Jul 2024"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        tools_in_user_message=False,
        date_string=date_string,
    )

    result = get_llama_input_text_from_messages(
        messages=[Message(**m) for m in messages],
        add_generation_prompt=True,
        tools=tools,
        tools_in_user_message=False,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_assistant_with_tool_call():
    """Test assistant message with a single tool call."""
    auto_tokenizer = get_llama_auto_tokenizer()
    date_string = "26 Jul 2024"

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                }
            ],
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        date_string=date_string,
    )

    msg_objects = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="get_weather", arguments={"location": "Tokyo"})],
        ),
    ]

    result = get_llama_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected
    assert '"name": "get_weather"' in result
    assert '"parameters":' in result


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_tool_response():
    """Test tool response message (uses ipython role)."""
    auto_tokenizer = get_llama_auto_tokenizer()
    date_string = "26 Jul 2024"

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 22, "condition": "sunny"}'},
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )

    msg_objects = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="get_weather", arguments={"location": "Tokyo"})],
        ),
        Message(role="tool", content='{"temperature": 22, "condition": "sunny"}'),
    ]

    result = get_llama_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=True,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected
    assert "<|start_header_id|>ipython<|end_header_id|>" in result


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_full_tool_use_conversation():
    """Test a complete tool use conversation flow."""
    auto_tokenizer = get_llama_auto_tokenizer()
    date_string = "26 Jul 2024"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Paris"},
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 18, "condition": "cloudy"}'},
        {
            "role": "assistant",
            "content": "The weather in Paris is 18°C and cloudy.",
        },
    ]

    expected = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        date_string=date_string,
    )

    msg_objects = [
        Message(role="system", content="You are a helpful weather assistant."),
        Message(role="user", content="What's the weather in Paris?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="get_weather", arguments={"location": "Paris"})],
        ),
        Message(role="tool", content='{"temperature": 18, "condition": "cloudy"}'),
        Message(role="assistant", content="The weather in Paris is 18°C and cloudy."),
    ]

    result = get_llama_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
        tools=tools,
        date_string=date_string,
        add_system_date_prompt=True,
    )

    assert result == expected


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_multiple_tool_calls_raises():
    """Test that multiple tool calls raises an error (Llama limitation)."""
    import pytest

    msg_objects = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(name="get_weather", arguments={"location": "Tokyo"}),
                ToolCall(name="get_weather", arguments={"location": "London"}),
            ],
        ),
    ]

    with pytest.raises(ValueError, match="single tool-calls"):
        get_llama_input_text_from_messages(
            messages=msg_objects,
            add_generation_prompt=False,
        )


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_tools_without_user_message_raises():
    """Test that tools in user message without a user message raises an error."""
    import pytest

    messages = [Message(role="system", content="You are helpful.")]

    tools = [{"type": "function", "function": {"name": "test"}}]

    with pytest.raises(ValueError, match="first user message"):
        get_llama_input_text_from_messages(
            messages=messages,
            add_generation_prompt=True,
            tools=tools,
            tools_in_user_message=True,
        )


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_bos_token():
    """Test that output starts with BOS token."""
    result = get_llama_input_text_from_messages(
        messages=[Message(role="user", content="Hello")],
        add_generation_prompt=True,
    )

    assert result.startswith("<|begin_of_text|>")


@pytest.mark.skipif(
    os.getenv("LLAMA_ACCESS") is None, reason="`LLAMA_ACCESS` env flag not set"
)
def test_llama_date_in_system():
    """Test that date is included in system message."""
    result = get_llama_input_text_from_messages(
        messages=[Message(role="user", content="Hello")],
        add_generation_prompt=True,
        date_string="15 Jan 2025",
        add_system_date_prompt=True,
    )

    assert "Today Date: 15 Jan 2025" in result
    assert "Cutting Knowledge Date: December 2023" in result
