"""Preview the B4 fine-tuning settings or explicitly start a new training run."""
import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("show-config", "train"))
    args = parser.parse_args()
    root = Path.cwd()
    spec = yaml.safe_load((root / "configs/b4_boundary_finetune.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((root / spec["base_config"]).read_text(encoding="utf-8"))
    for key, value in spec["overrides"].items():
        if isinstance(value, dict):
            base[key].update(value)
        else:
            base[key] = value
    if not (root / base["initial_checkpoint"]).is_file():
        raise FileNotFoundError(base["initial_checkpoint"])
    print(yaml.safe_dump(base, sort_keys=False), flush=True)
    if args.action == "show-config":
        return

    notebook = json.loads((root / "UAV_Segmentation.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        tags = cell["metadata"].get("tags", [])
        if "training-setup" in tags and "run" not in tags:
            exec(compile("".join(cell["source"]), cell["id"], "exec"), globals())
    _, best = train_experiment(base, "full", root / "UAV_dataset", root / "outputs")
    print(f"Best checkpoint: {best.resolve()}")


if __name__ == "__main__":
    main()
