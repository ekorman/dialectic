import argparse
import fnmatch
import inspect
from dataclasses import MISSING, Field, dataclass, fields
from types import UnionType
from typing import Callable, Literal, Sequence, Type, TypeVar, get_args, get_origin

import extty

from dialectic.experiments.prompts import PROMPT_COLLECTIONS, PromptCollection


def resolve_artifact_glob(pattern: str) -> list[str]:
    all_artifacts = extty.list_artifacts()
    matched = [a.name for a in all_artifacts if fnmatch.fnmatch(a.name, pattern)]
    matched.sort()
    if not matched:
        raise ValueError(f"No artifacts matched pattern: {pattern!r}")
    print(f"Resolved {pattern!r} -> {len(matched)} artifacts")
    return matched


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
    include_dataset_glob: bool = False,
):
    parser: argparse.ArgumentParser = subparsers.add_parser(name)
    for dc in dcs:
        _add_dataclass_to_parser_(parser, dc)

    if include_prompt_collection_id:
        parser.add_argument("--prompt-collection-id", type=int, required=True)
    if include_dataset_glob:
        parser.add_argument("--dataset-glob", type=str, nargs="+", required=True)


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
    env_name: str | None
    fn: Callable
    include_prompt_collection_id: bool
    include_dataset_glob: bool = False


def _build_parser(
    experiments: list[Experiment],
) -> tuple[argparse.ArgumentParser, dict[str | None, list[inspect.Parameter]]]:
    parser = argparse.ArgumentParser()
    parameters: dict[str | None, list[inspect.Parameter]] = {}

    def _should_exclude(p: inspect.Parameter, ex: Experiment) -> bool:
        if p.annotation == PromptCollection and ex.include_prompt_collection_id:
            return True
        if (
            p.annotation == list[str]
            and p.name == "dataset_artifacts"
            and ex.include_dataset_glob
        ):
            return True
        return False

    if len(experiments) == 1 and experiments[0].env_name is None:
        ex = experiments[0]
        sig = inspect.signature(ex.fn)
        parameters[None] = [
            p for p in sig.parameters.values() if not _should_exclude(p, ex)
        ]
        for dc in [p.annotation for p in parameters[None]]:
            _add_dataclass_to_parser_(parser, dc)
        if ex.include_prompt_collection_id:
            parser.add_argument("--prompt-collection-id", type=int, required=True)
        if ex.include_dataset_glob:
            parser.add_argument("--dataset-glob", type=str, nargs="+", required=True)
    else:
        subparsers = parser.add_subparsers(dest="env")
        for ex in experiments:
            assert ex.env_name is not None
            sig = inspect.signature(ex.fn)
            parameters[ex.env_name] = [
                p for p in sig.parameters.values() if not _should_exclude(p, ex)
            ]

            create_subparser(
                name=ex.env_name,
                subparsers=subparsers,
                dcs=[p.annotation for p in parameters[ex.env_name]],
                include_prompt_collection_id=ex.include_prompt_collection_id,
                include_dataset_glob=ex.include_dataset_glob,
            )

    return parser, parameters


def run_experiments_parser(experiments: list[Experiment]):
    parser, parameters = _build_parser(experiments)
    args = parser.parse_args()

    def _resolve_extra_args(ex: Experiment, kwargs: dict, env_name: str | None):
        if ex.include_prompt_collection_id:
            collection_key = env_name or ""
            kwargs["prompt_collection"] = PROMPT_COLLECTIONS[collection_key][
                args.prompt_collection_id
            ]
        if ex.include_dataset_glob:
            all_matched: list[str] = []
            for pattern in args.dataset_glob:
                all_matched.extend(resolve_artifact_glob(pattern))
            # deduplicate while preserving order
            seen: set[str] = set()
            kwargs["dataset_artifacts"] = [
                n for n in all_matched if n not in seen and not seen.add(n)
            ]

    if len(experiments) == 1 and experiments[0].env_name is None:
        ex = experiments[0]
        kwargs = {}
        for p in parameters[None]:
            kwargs[p.name] = load_dc_from_arg_parser_args(p.annotation, args)
        _resolve_extra_args(ex, kwargs, None)
        return ex.fn(**kwargs)

    for ex in experiments:
        if args.env == ex.env_name:
            kwargs = {}
            for p in parameters[ex.env_name]:
                param_class = p.annotation
                kwargs[p.name] = load_dc_from_arg_parser_args(param_class, args)
            _resolve_extra_args(ex, kwargs, ex.env_name)
            return ex.fn(**kwargs)
