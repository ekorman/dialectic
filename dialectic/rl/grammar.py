from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Protocol

import torch
from tokenizers import Tokenizer

from dialectic.rl.env import CountdownStep, build_countdown_equation


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


def countdown_solution_to_hard_tokens(
    solution: list[CountdownStep],
    numbers: list[int],
    target: int,
    tokenizer: Tokenizer,
    max_cycles: int,
    max_tokens_per_cycle: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert an expert countdown solution to per-cycle hard token tensors.

    Parameters
    ----------
    solution
        List of CountdownStep from the environment solver.
    numbers
        Original problem numbers.
    target
        Target value.
    tokenizer
        Tokenizer for encoding.
    max_cycles
        Maximum cycles dimension.
    max_tokens_per_cycle
        Maximum tokens per cycle dimension.
    pad_token_id
        Padding token ID.

    Returns
    -------
    tuple[Tensor, Tensor, int]
        (hard_token_ids [C, T_max], hard_token_lengths [C], n_cycles)
    """
    cycle_token_ids: list[list[int]] = []

    for step in solution:
        text = f" {step.left} {step.op} {step.right} = {step.result} </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cycle_token_ids.append(list(ids))

    equation = build_countdown_equation(numbers, solution, target)
    expr = equation.rsplit(" = ", 1)[0]
    answer_text = f" {expr} </answer>"
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False).ids
    cycle_token_ids.append(list(answer_ids))

    n_cycles = len(cycle_token_ids)

    hard_token_ids = torch.full(
        (max_cycles, max_tokens_per_cycle), pad_token_id, dtype=torch.long
    )
    hard_token_lengths = torch.zeros(max_cycles, dtype=torch.long)

    for c, ids in enumerate(cycle_token_ids):
        if c >= max_cycles:
            break
        length = min(len(ids), max_tokens_per_cycle)
        hard_token_ids[c, :length] = torch.tensor(ids[:length], dtype=torch.long)
        hard_token_lengths[c] = length

    return hard_token_ids, hard_token_lengths, min(n_cycles, max_cycles)


class _CycleState(IntEnum):
    OPEN = auto()
    ROUTE = auto()
    SCR_ATCH = auto()
    SCR_GT = auto()
    STEP_SPACE1 = auto()
    STEP_NUM1 = auto()
    STEP_SPACE2 = auto()
    STEP_NUM2 = auto()
    STEP_SPACE3 = auto()
    STEP_RESULT = auto()
    STEP_CLOSE1 = auto()
    STEP_CLOSE2 = auto()
    STEP_CLOSE3 = auto()
    ANS_GT = auto()
    ANS_BODY = auto()
    ANS_CLOSE1 = auto()
    ANS_CLOSE2 = auto()
    COMPLETE = auto()


class CountdownCycleGrammar:
    """Grammar for a full cycle including opening tag with SCRATCH/answer routing.

    The model chooses at the ROUTE state whether to generate a SCRATCH step
    or an answer. ``allow_answer`` controls whether the answer route is available.
    """

    def __init__(
        self,
        *,
        open_id: int,
        scr_id: int,
        atch_id: int,
        answer_id: int,
        gt_id: int,
        space_op_ids: list[int],
        space_eq_id: int,
        space_close_id: int,
        scratch_close_tag_ids: tuple[int, ...],
        answer_close_tag_ids: tuple[int, ...],
        body_token_ids: list[int],
        space_id: int = 220,
        allow_answer: bool = True,
        force_answer: bool = False,
    ):
        self._open_id = open_id
        self._scr_id = scr_id
        self._answer_id = answer_id
        self._space_op_ids = space_op_ids
        self._space_eq_id = space_eq_id
        self._space_close_id = space_close_id
        self._scratch_close_tag_ids = scratch_close_tag_ids
        self._answer_close_tag_ids = answer_close_tag_ids
        self._space_id = space_id
        self._state = _CycleState.OPEN
        self._took_answer_route = False

        if force_answer:
            route_ids = [answer_id]
        elif allow_answer:
            route_ids = [scr_id, answer_id]
        else:
            route_ids = [scr_id]

        self._valid: dict[_CycleState, list[int]] = {
            _CycleState.OPEN: [open_id],
            _CycleState.ROUTE: route_ids,
            _CycleState.SCR_ATCH: [atch_id],
            _CycleState.SCR_GT: [gt_id],
            _CycleState.STEP_SPACE1: [space_id],
            _CycleState.STEP_NUM1: DIGIT_TOKEN_IDS + space_op_ids,
            _CycleState.STEP_SPACE2: [space_id],
            _CycleState.STEP_NUM2: DIGIT_TOKEN_IDS + [space_eq_id],
            _CycleState.STEP_SPACE3: [space_id],
            _CycleState.STEP_RESULT: DIGIT_TOKEN_IDS + [space_close_id],
            _CycleState.STEP_CLOSE1: [scratch_close_tag_ids[0]],
            _CycleState.STEP_CLOSE2: [scratch_close_tag_ids[1]],
            _CycleState.STEP_CLOSE3: [scratch_close_tag_ids[2]],
            _CycleState.ANS_GT: [gt_id],
            _CycleState.ANS_BODY: body_token_ids + [space_close_id],
            _CycleState.ANS_CLOSE1: [answer_close_tag_ids[0]],
            _CycleState.ANS_CLOSE2: [answer_close_tag_ids[1]],
        }

    def valid_token_ids(self) -> list[int]:
        return self._valid[self._state]

    def advance(self, token_id: int) -> None:
        s = self._state
        if s == _CycleState.OPEN:
            self._state = _CycleState.ROUTE
        elif s == _CycleState.ROUTE:
            if token_id == self._scr_id:
                self._state = _CycleState.SCR_ATCH
            elif token_id == self._answer_id:
                self._took_answer_route = True
                self._state = _CycleState.ANS_GT
        elif s == _CycleState.SCR_ATCH:
            self._state = _CycleState.SCR_GT
        elif s == _CycleState.SCR_GT:
            self._state = _CycleState.STEP_SPACE1
        elif s == _CycleState.STEP_SPACE1:
            self._state = _CycleState.STEP_NUM1
        elif s == _CycleState.STEP_NUM1:
            if token_id in self._space_op_ids:
                self._state = _CycleState.STEP_SPACE2
        elif s == _CycleState.STEP_SPACE2:
            self._state = _CycleState.STEP_NUM2
        elif s == _CycleState.STEP_NUM2:
            if token_id == self._space_eq_id:
                self._state = _CycleState.STEP_SPACE3
        elif s == _CycleState.STEP_SPACE3:
            self._state = _CycleState.STEP_RESULT
        elif s == _CycleState.STEP_RESULT:
            if token_id == self._space_close_id:
                self._state = _CycleState.STEP_CLOSE1
        elif s == _CycleState.STEP_CLOSE1:
            self._state = _CycleState.STEP_CLOSE2
        elif s == _CycleState.STEP_CLOSE2:
            self._state = _CycleState.STEP_CLOSE3
        elif s == _CycleState.STEP_CLOSE3:
            self._state = _CycleState.COMPLETE
        elif s == _CycleState.ANS_GT:
            self._state = _CycleState.ANS_BODY
        elif s == _CycleState.ANS_BODY:
            if token_id == self._space_close_id:
                self._state = _CycleState.ANS_CLOSE1
        elif s == _CycleState.ANS_CLOSE1:
            self._state = _CycleState.ANS_CLOSE2
        elif s == _CycleState.ANS_CLOSE2:
            self._state = _CycleState.COMPLETE

    def is_complete(self) -> bool:
        return self._state == _CycleState.COMPLETE

    def is_terminal(self) -> bool:
        return self._state == _CycleState.COMPLETE and self._took_answer_route

    def reset(self) -> None:
        self._state = _CycleState.OPEN
        self._took_answer_route = False


def build_countdown_cycle_grammar_factory(
    tokenizer: Tokenizer,
) -> Callable[[bool], CountdownCycleGrammar]:
    """Build a factory that creates CountdownCycleGrammar instances.

    Parameters
    ----------
    tokenizer
        Qwen tokenizer for resolving token IDs.

    Returns
    -------
    Callable[[bool], CountdownCycleGrammar]
        Factory taking ``allow_answer`` and returning a grammar instance.
    """

    def _ids(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    open_id = _ids("<")[0]
    scr_id = _ids("SCR")[0]
    atch_id = _ids("ATCH")[0]
    answer_id = _ids("answer")[0]
    gt_id = _ids(">")[0]
    space_close_id = _ids(" </")[0]
    scratch_close_tag = tuple(_ids("</SCRATCH>")[1:])
    answer_close_tag = tuple(_ids("</answer>")[1:])
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

    def factory(
        allow_answer: bool = True, force_answer: bool = False
    ) -> CountdownCycleGrammar:
        return CountdownCycleGrammar(
            open_id=open_id,
            scr_id=scr_id,
            atch_id=atch_id,
            answer_id=answer_id,
            gt_id=gt_id,
            space_op_ids=space_op_ids,
            space_eq_id=space_eq_id,
            space_close_id=space_close_id,
            scratch_close_tag_ids=scratch_close_tag,
            answer_close_tag_ids=answer_close_tag,
            body_token_ids=body_token_ids,
            space_id=space_id,
            allow_answer=allow_answer,
            force_answer=force_answer,
        )

    return factory
