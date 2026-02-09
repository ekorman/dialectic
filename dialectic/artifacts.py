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
        print(f"artifact {artifact} not found in cache, downloading to {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(artifact.url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    else:
        print(f"artifact {artifact} found at {local_path}")

    return local_path


def map_hf_key_to_dialectic(k: str) -> str:
    if k.startswith("model."):
        return k[6:]
    return k
