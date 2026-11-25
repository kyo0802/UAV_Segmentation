"""Compare B7 and B10 using the same five-view M7 probability ensemble."""
import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from fusion_study import notebook_definitions


GRID = range(0, 101, 10)
CONFIG = Path("configs/cross_model_m7.yaml")
ROOT = Path("UAV_dataset")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def model_probability(n, member, image, device):
    if image.size != (800, 600):
        raise ValueError(f"Expected 800x600 input, received {image.size}")
    model, views = member
    total = None
    single_view = None
    for name, settings, weight in views:
        prob = n.predict_logits(model, image, settings, device).float().softmax(1)
        if name == "F0":
            single_view = prob
        weighted = prob * weight
        total = weighted if total is None else total + weighted
    return total, single_view


def ensemble_mask(probs, b2_percent):
    if b2_percent == 0:
        score = probs["b4"]
    elif b2_percent == 100:
        score = probs["b2"]
    else:
        score = probs["b2"] * (b2_percent / 100) + probs["b4"] * ((100 - b2_percent) / 100)
    return score[0].argmax(0).byte().cpu().numpy()


def load_members(n, checkpoints, spec, split):
    members = {}
    weights = spec["methods"]["M7"]
    if len(weights) != 5 or min(weights.values()) <= 0 or not math.isclose(sum(weights.values()), 1):
        raise ValueError("Invalid five-view M7 weights")
    for variant, path in checkpoints.items():
        model, state = n.load_trained(Path(path), "cuda", "official")
        config = state["config"]
        if (state["mode"] != "full" or state["split"] != split
                or config["model"]["variant"] != variant or config["model"]["name"] != "segformer"):
            raise ValueError(f"Unexpected model or validation split in {path}")
        views = []
        for name, weight in weights.items():
            _, width, height = spec["views"][name]
            settings = n.merge(config, dict(image_size=[width, height], inference=dict(
                tile_size=None, hflip=True, vflip=False, scales=[1.0])))
            views.append((name, settings, weight))
        members[variant] = model, views
        del state
    return members


def compare_reference(observed, reference, tolerance):
    result = {}
    for key in ("image_miou", "global_miou"):
        expected = float(reference[key])
        actual = observed[key]
        result[key] = dict(expected=expected, observed=actual,
                           passed=abs(actual - expected) <= tolerance)
    return dict(status="passed" if all(v["passed"] for v in result.values()) else "failed",
                metrics=result)


def evaluate(n, folder, members, pairs, references, b4_reference, tolerance):
    metrics = {percent: n.SegmentationMetrics() for percent in GRID}
    b4_single = n.SegmentationMetrics()
    seconds = dict.fromkeys(GRID, 0.0)
    for index, pair in enumerate(pairs, 1):
        image, mask = n.read_pair(pair)
        target = np.asarray(mask)
        if target.shape != (600, 800):
            raise ValueError(f"Unexpected mask dimensions for {pair.image}")
        probs, elapsed = {}, {}
        for variant, member in members.items():
            torch.cuda.synchronize()
            start = perf_counter()
            probs[variant], single = model_probability(n, member, image, "cuda")
            torch.cuda.synchronize()
            elapsed[variant] = perf_counter() - start
            if variant == "b4":
                b4_single.add(target, single[0].argmax(0).byte().cpu().numpy(), pair.image.name)
        for percent in GRID:
            start = perf_counter()
            prediction = ensemble_mask(probs, percent)
            seconds[percent] += (perf_counter() - start
                                 + (elapsed["b2"] if percent else 0)
                                 + (elapsed["b4"] if percent < 100 else 0))
            metrics[percent].add(target, prediction, pair.image.name)
        if index % 100 == 0:
            print(f"{index}/{len(pairs)}", flush=True)
    summaries = {percent: metric.summary() for percent, metric in metrics.items()}
    rows, classes = [], []
    for percent, summary in summaries.items():
        rows.append(dict(b2_percent=percent, b4_percent=100 - percent,
                         image_miou=summary["image_miou"], global_miou=summary["global_miou"],
                         pixel_accuracy=summary["pixel_accuracy"],
                         estimated_inference_seconds=seconds[percent]))
        for label, iou in enumerate(summary["per_class_iou"]):
            classes.append(dict(b2_percent=percent, b4_percent=100-percent, label=label, iou=iou))
    rows.sort(key=lambda row: -row["image_miou"])
    write_rows(folder / "comparison.csv", [dict(rank=i, **r) for i, r in enumerate(rows, 1)])
    write_rows(folder / "classes.csv", classes)
    checks = dict(
        b2_m7=compare_reference(summaries[100], references["M7"], tolerance),
        b4_512_hflip=(compare_reference(b4_single.summary(), references["B4"], tolerance)
                      if "B4" in references else dict(status="unverified",
                      reason="The historical B4 reference belongs to a different checkpoint")),
        b4_m7=(compare_reference(summaries[0], b4_reference, tolerance) if b4_reference else
               dict(status="unverified", reason="No independent B4 five-view M7 reference supplied")),
    )
    n.write_json(folder / "endpoint_checks.json", checks)
    print(folder.resolve(), rows[0], checks, flush=True)
    if any(check["status"] == "failed" for check in checks.values()):
        raise ValueError("Endpoint mismatch. See endpoint_checks.json")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    ev = actions.add_parser("evaluate")
    ev.add_argument("--b4-m7-reference", type=Path,
                    help="Independent same-split B4 M7 JSON with image_miou and global_miou")
    sub = actions.add_parser("submit")
    sub.add_argument("--b2-percent", type=int, choices=GRID, required=True)
    for action in (ev, sub):
        action.add_argument("--b4-checkpoint", type=Path, help="Use an alternate trained B4 best.pt")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    spec = yaml.safe_load(Path(cfg["m7_config"]).read_text(encoding="utf-8"))
    reference_run = Path(cfg["reference_run"])
    split = json.loads((reference_run / "split.json").read_text(encoding="utf-8"))
    if len(split["validation"]) != 800 or len(set(split["validation"])) != 800:
        raise ValueError("Expected the fixed 800-image validation split")
    checkpoints = dict(b2=spec["checkpoints"]["B7"], b4=str(args.b4_checkpoint or cfg["b4_checkpoint"]))
    b4_reference = None
    if args.action == "evaluate":
        references = {row["method"]: row for row in read_rows(reference_run / "comparison.csv")}
        if Path(checkpoints["b4"]).resolve() != Path(cfg["b4_checkpoint"]).resolve():
            del references["B4"]
        if args.b4_m7_reference:
            b4_reference = json.loads(args.b4_m7_reference.read_text(encoding="utf-8"))
            for key in ("image_miou", "global_miou"):
                if not 0 <= float(b4_reference[key]) <= 1:
                    raise ValueError(f"Invalid reference {key}")
    else:
        if args.b2_percent == 0:
            del checkpoints["b2"]
        elif args.b2_percent == 100:
            del checkpoints["b4"]
    n = notebook_definitions()
    members = load_members(n, checkpoints, spec, split)
    folder = n.new_run(Path("outputs"), "cross_model_m7_" + args.action)
    n.write_json(folder / "config.json", dict(checkpoints=checkpoints, m7=spec["methods"]["M7"],
                 views={k: spec["views"][k][1:] for k in spec["methods"]["M7"]},
                 b2_percent=list(GRID) if args.action == "evaluate" else args.b2_percent,
                 reference_run=str(reference_run), b4_m7_reference=str(args.b4_m7_reference)
                 if b4_reference else None, hflip=True))
    if args.action == "evaluate":
        pairs = n.validation_pairs(ROOT, {"split": split})
        if sorted(p.image.name for p in pairs) != sorted(split["validation"]):
            raise ValueError("Validation image names differ from reference")
        evaluate(n, folder, members, pairs, references, b4_reference, cfg["absolute_tolerance"])
    else:
        def predict(selected, image, device):
            probs = {key: model_probability(n, member, image, device)[0]
                     for key, member in selected.items()}
            return ensemble_mask(probs, args.b2_percent)

        output = n.output_for(folder, "submissions") / "submission.csv"
        count = n.generate_submission(ROOT, members, output, "cuda", mask_fn=predict)
        print(output.resolve(), count, flush=True)


if __name__ == "__main__":
    main()
