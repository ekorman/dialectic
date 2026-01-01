from tokenizers import Encoding, Tokenizer
from transformers import AutoTokenizer

from dialectic.llm.tokenizer import Message, ToolCall, get_input_text_from_messages


def get_auto_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_tokenizer_batch():
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.enable_padding(pad_id=151643)
    tokens = tokenizer.encode_batch(
        ["short prompt", "this is a much longer prompt for testing"]
    )
    """
    tensor([[1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]])
    """
    import pdb

    pdb.set_trace()
    assert len(tokens[0]) == len(tokens[1]) == max([len(t) for t in tokens])


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
        == get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
    )

    assert result == expected


def test_empty_messages():
    """Test with empty messages list."""
    result = get_input_text_from_messages(
        messages=[],
        add_generation_prompt=True,
    )
    assert result == "<|im_start|>assistant\n"


def test_empty_messages_no_generation():
    """Test with empty messages and no generation prompt."""
    result = get_input_text_from_messages(
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

    result = get_input_text_from_messages(
        messages=msg_objects,
        add_generation_prompt=False,
        tools=tools,
    )

    assert result == expected
