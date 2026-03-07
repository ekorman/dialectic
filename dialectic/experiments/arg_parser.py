import argparse
import inspect
from dataclasses import MISSING, Field, dataclass, fields
from types import UnionType
from typing import Callable, Literal, Sequence, Type, TypeVar, get_args, get_origin

from dialectic.experiments.prompts import PROMPT_COLLECTIONS, PromptCollection

T = TypeVar("T")


def _arg_type(tp):
    origin = get_origin(tp)
    if origin is Literal:
        return str
    if tp == int | list[int]:
        return lambda s: [int(x) for x in s.split(",")] if "," in s else int(s)
    if origin is UnionType:
        args = tuple(a for a in get_args(tp) if a is not type(None))
        return args[0] if len(args) == 1 else tp
    return tp


def _add_dataclass_to_parser_(parser: argparse.ArgumentParser, dc: Type[T]) -> None:
    for f in fields(dc):
        arg_type = _arg_type(f.type)

        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(
                f"--{f.name.replace('_', '-')}",
                type=arg_type,
                required=f.default is MISSING,
            )


def create_subparser(
    name: str,
    subparsers: argparse._SubParsersAction,
    dcs: Sequence[Type[T]],
    include_prompt_collection_id: bool,
):
    parser: argparse.ArgumentParser = subparsers.add_parser(name)
    for dc in dcs:
        _add_dataclass_to_parser_(parser, dc)

    if include_prompt_collection_id:
        parser.add_argument("--prompt-collection-id", type=int, required=True)


def load_dc_from_arg_parser_args(dc: Type[T], args: argparse.Namespace) -> T:
    def _get_value(field: Field):
        val = getattr(args, field.name)
        if val is None:
            val = field.default
        if field.type is bool and val is MISSING:
            val = False

        assert val is not MISSING
        return val

    return dc(**{f.name: _get_value(f) for f in fields(dc)})


@dataclass
class Experiment:
    env_name: str
    fn: Callable
    include_prompt_collection_id: bool


def _build_parser(
    experiments: list[Experiment],
) -> tuple[argparse.ArgumentParser, list[inspect.Parameter]]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="env")

    for ex in experiments:
        sig = inspect.signature(ex.fn)
        parameters = [
            p
            for p in sig.parameters.values()
            if not (
                p.annotation == PromptCollection and ex.include_prompt_collection_id
            )
        ]
        create_subparser(
            name=ex.env_name,
            subparsers=subparsers,
            dcs=[p.annotation for p in parameters],
            include_prompt_collection_id=ex.include_prompt_collection_id,
        )

    return parser, parameters


def run_experiments_parser(experiments: list[Experiment]):
    parser, parameters = _build_parser(experiments)
    args = parser.parse_args()
    for ex in experiments:
        if args.env == ex.env_name:
            kwargs = {}
            for p in parameters:
                param_class = p.annotation
                kwargs[p.name] = load_dc_from_arg_parser_args(param_class, args)

            if ex.include_prompt_collection_id:
                kwargs["prompt_collection"] = PROMPT_COLLECTIONS[ex.env_name][
                    args.prompt_collection_id
                ]

            return ex.fn(**kwargs)
