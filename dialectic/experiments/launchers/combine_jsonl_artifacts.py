import os
import tempfile
from datetime import datetime

import extty

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import CombineJsonlParams
from dialectic.log import log


@extty.experiment(project="combine-jsonl-artifacts")
def combine_jsonl_artifacts(
    *,
    combine_params: CombineJsonlParams,
    dataset_artifacts: list[str],
):
    tmpfile = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".jsonl", delete=False, prefix="combined_jsonl_"
    )
    total_lines = 0
    try:
        for name in dataset_artifacts:
            data = extty.load_artifact(name)
            if not isinstance(data, bytes):
                raise ValueError(
                    f"Expected bytes from artifact {name!r}, got {type(data)}"
                )
            if not data.endswith(b"\n"):
                data = data + b"\n"
            tmpfile.write(data)
            n_lines = sum(1 for line in data.splitlines() if line.strip())
            total_lines += n_lines
            log.info(f"Appended {n_lines} lines from {name!r}")
        tmpfile.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_name = f"{combine_params.name_prefix}-{ts}"
        meta = extty.save_artifact(
            name=artifact_name,
            path=tmpfile.name,
            description=(
                f"Combined jsonl from {len(dataset_artifacts)} artifact(s), "
                f"{total_lines} total lines"
            ),
            metadata={
                "source_artifacts": list(dataset_artifacts),
                "total_lines": total_lines,
            },
        )
        log.info(f"Uploaded artifact: {meta}")
    finally:
        os.unlink(tmpfile.name)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name=None,
                fn=combine_jsonl_artifacts,
                include_prompt_collection_id=False,
                include_dataset_glob=True,
            ),
        ]
    )
