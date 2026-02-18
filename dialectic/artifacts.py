import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests
from safetensors.torch import load_file

WEIGHTS_CACHE = Path(
    os.getenv("DIALECTIC_WEIGHTS_PATH", Path.home() / ".dialectic" / "weights")
)


@dataclass(frozen=True)
class Artifact:
    urls: list[str]
    filenames: list[str]


def get_artifact(artifact: Artifact) -> list[Path]:
    local_paths = []
    for url, filename in zip(artifact.urls, artifact.filenames):
        local_path = WEIGHTS_CACHE / filename
        if not local_path.exists():
            print(
                f"artifact {artifact} not found in cache, downloading to {local_path}"
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        else:
            print(f"artifact file {filename} found at {local_path}")
        local_paths.append(local_path)

    return local_paths


def map_hf_key_to_dialectic(k: str) -> str:
    if k.startswith("model."):
        return k[6:]
    return k


def load_safe_tensors(weight_files: Sequence[str | Path]) -> dict[str, Any]:
    ret = {}
    for weight_file in weight_files:
        ret.update(load_file(weight_file))

    return ret


def load_state_dict_from_artifact(
    artifact: Artifact, convert_keys: bool, tied_weights: bool
) -> dict[str, Any]:
    sd = load_safe_tensors(get_artifact(artifact))
    if convert_keys:
        sd = {map_hf_key_to_dialectic(k): v for k, v in sd.items()}
    if tied_weights:
        if "lm_head.weight" not in sd:
            sd["lm_head.weight"] = sd["embed_tokens.weight"]
        elif not (sd["lm_head.weight"] == sd["embed_tokens.weight"]).all():
            raise ValueError(
                "Expected `lm_head.weight` and `embed_tokens.weight` to be identical."
            )
    return sd
