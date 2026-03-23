from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Protocol

from tokenizers import Tokenizer


class Grammar(Protocol):
    def valid_token_ids(self) -> list[int]: ...
    def advance(self, token_id: int) -> None: ...
    def is_complete(self) -> bool: ...
    def reset(self) -> None: ...


@dataclass
class GrammarSpec:
    trigger_token_ids: tuple[int, ...]
    grammar_factory: Callable[[], Grammar]
    is_terminal: bool


DIGIT_TOKEN_IDS = list(range(15, 25))  # '0'..'9' in Qwen tokenizer


class _StepState(IntEnum):
    SPACE1 = auto()
    NUM1 = auto()
    SPACE2 = auto()
    NUM2 = auto()
    SPACE3 = auto()
    RESULT = auto()
    CLOSE1 = auto()
    CLOSE2 = auto()
    CLOSE3 = auto()
    COMPLETE = auto()


class CountdownStepGrammar:
    """Enforces: <space> <digits+> <space_op> <space> <digits+> <space_eq> <space> <digits+> <space_close> SCR ATCH >

    Token-level state machine for `<number> <op> <number> = <number> </SCRATCH>`.
    """

    def __init__(
        self,
        space_op_ids: list[int],
        space_eq_id: int,
        space_close_id: int,
        close_tag_ids: tuple[int, ...],
        space_id: int = 220,
    ):
        self._space_op_ids = space_op_ids
        self._space_eq_id = space_eq_id
        self._space_close_id = space_close_id
        self._close_tag_ids = close_tag_ids
        self._space_id = space_id
        self._state = _StepState.SPACE1

        self._valid: dict[_StepState, list[int]] = {
            _StepState.SPACE1: [space_id],
            _StepState.NUM1: DIGIT_TOKEN_IDS + space_op_ids,
            _StepState.SPACE2: [space_id],
            _StepState.NUM2: DIGIT_TOKEN_IDS + [space_eq_id],
            _StepState.SPACE3: [space_id],
            _StepState.RESULT: DIGIT_TOKEN_IDS + [space_close_id],
            _StepState.CLOSE1: [close_tag_ids[0]],
            _StepState.CLOSE2: [close_tag_ids[1]],
            _StepState.CLOSE3: [close_tag_ids[2]],
        }

    def valid_token_ids(self) -> list[int]:
        return self._valid[self._state]

    def advance(self, token_id: int) -> None:
        if self._state == _StepState.SPACE1:
            self._state = _StepState.NUM1
        elif self._state == _StepState.NUM1:
            if token_id in self._space_op_ids:
                self._state = _StepState.SPACE2
        elif self._state == _StepState.SPACE2:
            self._state = _StepState.NUM2
        elif self._state == _StepState.NUM2:
            if token_id == self._space_eq_id:
                self._state = _StepState.SPACE3
        elif self._state == _StepState.SPACE3:
            self._state = _StepState.RESULT
        elif self._state == _StepState.RESULT:
            if token_id == self._space_close_id:
                self._state = _StepState.CLOSE1
        elif self._state == _StepState.CLOSE1:
            self._state = _StepState.CLOSE2
        elif self._state == _StepState.CLOSE2:
            self._state = _StepState.CLOSE3
        elif self._state == _StepState.CLOSE3:
            self._state = _StepState.COMPLETE

    def is_complete(self) -> bool:
        return self._state == _StepState.COMPLETE

    def reset(self) -> None:
        self._state = _StepState.SPACE1


class _AnswerState(IntEnum):
    BODY = auto()
    CLOSE1 = auto()
    CLOSE2 = auto()
    COMPLETE = auto()


class CountdownAnswerGrammar:
    """Allows arithmetic tokens until </answer>.

    Accepts digits, space-operators, space-equals, parentheses, spaces, then
    the </answer> closing sequence.
    """

    def __init__(
        self,
        body_token_ids: list[int],
        space_close_id: int,
        close_tag_ids: tuple[int, ...],
    ):
        self._body_ids = body_token_ids + [space_close_id]
        self._space_close_id = space_close_id
        self._close_tag_ids = close_tag_ids
        self._state = _AnswerState.BODY

        self._valid: dict[_AnswerState, list[int]] = {
            _AnswerState.BODY: self._body_ids,
            _AnswerState.CLOSE1: [close_tag_ids[0]],
            _AnswerState.CLOSE2: [close_tag_ids[1]],
        }

    def valid_token_ids(self) -> list[int]:
        return self._valid[self._state]

    def advance(self, token_id: int) -> None:
        if self._state == _AnswerState.BODY:
            if token_id == self._space_close_id:
                self._state = _AnswerState.CLOSE1
        elif self._state == _AnswerState.CLOSE1:
            self._state = _AnswerState.CLOSE2
        elif self._state == _AnswerState.CLOSE2:
            self._state = _AnswerState.COMPLETE

    def is_complete(self) -> bool:
        return self._state == _AnswerState.COMPLETE

    def reset(self) -> None:
        self._state = _AnswerState.BODY


def build_countdown_grammar_specs(tokenizer: Tokenizer) -> list[GrammarSpec]:
    """Build GrammarSpec list for countdown with <SCRATCH> and <answer> tags.

    Parameters
    ----------
    tokenizer
        Qwen tokenizer used to resolve token IDs.

    Returns
    -------
    list[GrammarSpec]
        Two specs: one for <SCRATCH> (non-terminal) and one for <answer> (terminal).
    """

    def _ids(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    scratch_trigger = tuple(_ids("<SCRATCH>"))
    answer_trigger = tuple(_ids("<answer>"))

    space_close_id = _ids(" </")[0]  # 690
    scratch_close_tag = tuple(_ids("</SCRATCH>")[1:])  # skip the '</' → (SCR, ATCH, >)
    answer_close_tag = tuple(_ids("</answer>")[1:])  # skip the '</' → (answer, >)

    space_op_ids = [_ids(" +")[0], _ids(" -")[0], _ids(" *")[0], _ids(" /")[0]]
    space_eq_id = _ids(" =")[0]
    space_id = _ids(" ")[0]
    space_paren_id = _ids(" (")[0]
    close_paren_id = _ids(")")[0]

    body_token_ids = (
        DIGIT_TOKEN_IDS
        + space_op_ids
        + [space_eq_id, space_paren_id, close_paren_id, space_id]
    )

    def make_step_grammar() -> CountdownStepGrammar:
        return CountdownStepGrammar(
            space_op_ids=space_op_ids,
            space_eq_id=space_eq_id,
            space_close_id=space_close_id,
            close_tag_ids=scratch_close_tag,
            space_id=space_id,
        )

    def make_answer_grammar() -> CountdownAnswerGrammar:
        return CountdownAnswerGrammar(
            body_token_ids=body_token_ids,
            space_close_id=space_close_id,
            close_tag_ids=answer_close_tag,
        )

    return [
        GrammarSpec(
            trigger_token_ids=scratch_trigger,
            grammar_factory=make_step_grammar,
            is_terminal=False,
        ),
        GrammarSpec(
            trigger_token_ids=answer_trigger,
            grammar_factory=make_answer_grammar,
            is_terminal=True,
        ),
    ]
