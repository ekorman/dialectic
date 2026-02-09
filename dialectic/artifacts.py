import os
from dataclasses import dataclass
from pathlib import Path

import requests

WEIGHTS_CACHE = Path(
    os.getenv("DIALECTIC_WEIGHTS_PATH", Path.home() / ".dialectic" / "weights")
)


@dataclass(frozen=True)
class Artifact:
    url: str
    filename: str


def get_artifact(artifact: Artifact) -> Path:
    local_path = WEIGHTS_CACHE / artifact.filename
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(artifact.url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

    return local_path


def map_hf_key_to_dialectic(k: str) -> str:
    if k.startswith("model."):
        return k[6:]
    return k
