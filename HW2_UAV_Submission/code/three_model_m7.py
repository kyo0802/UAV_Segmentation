"""Evaluate or submit probability ensembles of the fixed B2, B4 and Residual B4 checkpoints."""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from class_postprocess import read, save
from cross_model_m7 import CONFIG, GRID, ROOT, load_members, model_probability, read_rows, write_rows
from fusion_study import notebook_definitions
from segmentation_errors import components
from sliding_m7 import boundaries


MODELS = ("b2", "b4", "residual_b4")
WEIGHTS = [(a, b, 100-a-b) for a in GRID for b in GRID if a+b <= 100]
BASELINE = (30, 0, 70)


def weight_columns(weights):
    return dict(zip(("b2_percent", "b4_percent", "residual_b4_percent"), weights))


def prediction(probabilities, weights):
    score = None
    for key, percent in zip(MODELS, weights, strict=True):
        if percent:
            term = probabilities[key] if percent == 100 else probabilities[key] * (percent / 100)
            score = term if score is None else score + term
    return score[0].argmax(0).byte().cpu().numpy()


def probabilities_for(n, members, image):
    probabilities = {key: model_probability(n, member, image, "cuda")[0]
                     for key, member in members.items()}
    if any(p.dtype != torch.float32 or p.shape != (1, n.NUM_CLASSES, 600, 800)
           for p in probabilities.values()):
        raise ValueError("Expected native 800×600 FP32 probabilities")
    return probabilities


@dataclass
class Totals:
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((16, 16), dtype=np.int64))
    scores: list = field(default_factory=list)
    boundary_errors: int = 0
    recall_sum: float = 0.0

    def combine(self, other):
        return Totals(self.matrix + other.matrix, self.scores + other.scores,
                      self.boundary_errors + other.boundary_errors, self.recall_sum + other.recall_sum)

    def summary(self, n, denominators):
        boundary_pixels, components_count = denominators
        return dict(image_miou=n.mean_iou(np.array(self.scores)),
                    global_miou=n.mean_iou(n.iou_values(self.matrix)),
                    pixel_accuracy=float(self.matrix.trace() / self.matrix.sum()),
                    boundary_error_rate=self.boundary_errors / boundary_pixels if boundary_pixels else None,
                    small_component_recall=self.recall_sum / components_count if components_count else None)


def measure(n, members, pairs, split_name):
    totals = {weights: Totals() for weights in WEIGHTS}
    boundary_pixels = small_count = 0
    for index, pair in enumerate(pairs, 1):
        image, mask = n.read_pair(pair)
        target = np.asarray(mask)
        if target.shape != (600, 800):
            raise ValueError(f"Unexpected mask dimensions for {pair.image}")
        boundary, _ = boundaries(target)
        ids = components(target)
        areas = np.bincount(ids.ravel())
        valid = target != n.IGNORE_INDEX
        small = (areas <= 256) & (np.bincount(ids[valid], minlength=len(areas)) > 0)
        small_mask = small[ids] & valid
        small_weights = 1.0 / areas[ids[small_mask]]
        boundary_pixels += int(boundary.sum())
        small_count += int(small.sum())
        # Precompute GT terms shared by all 66 configurations.
        gt = target[valid]
        gt_bins = n.NUM_CLASSES * gt.astype(np.int64)
        boundary = boundary[valid]
        small_mask = small_mask[valid]
        probabilities = probabilities_for(n, members, image)
        for weights, total in totals.items():
            pred = prediction(probabilities, weights)[valid]
            cm = np.bincount(gt_bins + pred, minlength=n.NUM_CLASSES**2).reshape(total.matrix.shape)
            total.matrix += cm
            score = n.mean_iou(n.iou_values(cm))
            total.scores.append(score if score is not None else np.nan)
            correct = pred == gt
            total.boundary_errors += int(np.count_nonzero(boundary & ~correct))
            total.recall_sum += float(small_weights[correct[small_mask]].sum())
        if index % 50 == 0:
            print(f"{split_name}: {index}/{len(pairs)}", flush=True)
    return totals, (boundary_pixels, small_count)


def model_count(weights):
    return sum(percent > 0 for percent in weights)


def holdout_comparison(totals, selected):
    candidate = np.array(totals[selected].scores)
    baseline = np.array(totals[BASELINE].scores)
    delta = (candidate - baseline)[np.isfinite(candidate) & np.isfinite(baseline)]
    draws = np.random.default_rng(42).integers(len(delta), size=(10000, len(delta)))
    low, high = np.percentile(100 * delta[draws].mean(axis=1), [2.5, 97.5])
    return dict(delta_pp=float(100 * delta.mean()), ci95_low_pp=float(low), ci95_high_pp=float(high))


def evaluate(n, members, pairs, partition, reference, folder, config):
    by_name = {p.image.name: p for p in pairs}
    cal_names, hold_names = partition["calibration"], partition["holdout"]
    if ((len(cal_names), len(hold_names)) != (600, 200)
            or len(set(cal_names + hold_names)) != 800 or set(cal_names + hold_names) != by_name.keys()):
        raise ValueError("Historical calibration and holdout must partition the fixed 800-image split")
    save(folder / "split.json", partition)
    cal, cal_counts = measure(n, members, [by_name[k] for k in cal_names], "calibration")
    cal_order = sorted(WEIGHTS, key=lambda w: (-cal[w].summary(n, cal_counts)["image_miou"], model_count(w), w))
    selected = cal_order[0]
    group_selected = {k: next(w for w in cal_order if model_count(w) == k) for k in (1, 2, 3)}
    save(folder / "calibration_selection.json", dict(weights=weight_columns(selected),
         by_model_count={k: weight_columns(w) for k, w in group_selected.items()}))
    print("Calibration selection:", weight_columns(selected), flush=True)
    hold, hold_counts = measure(n, members, [by_name[k] for k in hold_names], "holdout")
    all_totals = {w: cal[w].combine(hold[w]) for w in WEIGHTS}
    all_counts = tuple(a+b for a, b in zip(cal_counts, hold_counts))
    partitions = {"calibration": (cal, cal_counts), "holdout": (hold, hold_counts),
                  "all": (all_totals, all_counts)}
    summary = {key: {w:t.summary(n, counts) for w,t in metrics.items()}
               for key,(metrics,counts) in partitions.items()}
    observed = summary["all"][BASELINE]
    checks = {}
    for key in ("image_miou", "global_miou", "pixel_accuracy"):
        expected = float(reference[key])
        checks[key] = dict(expected=expected, observed=observed[key],
                           passed=abs(expected-observed[key]) <= config["absolute_tolerance"])
    save(folder / "baseline_checks.json", checks)
    if not all(result["passed"] for result in checks.values()):
        raise ValueError("Historical baseline mismatch. See baseline_checks.json")
    order = sorted(WEIGHTS, key=lambda w: -summary["all"][w]["image_miou"])
    rows, classes = [], []
    for rank, weights in enumerate(order, 1):
        prefix = weight_columns(weights)
        row = dict(rank=rank, **prefix, models=model_count(weights), calibration_rank=cal_order.index(weights)+1)
        for split, results in summary.items():
            row.update({f"{split}_{key}":value for key,value in results[weights].items()})
            cm = partitions[split][0][weights].matrix
            for label, iou in enumerate(n.iou_values(cm)):
                tp = int(cm[label,label])
                predicted, target = int(cm[:,label].sum()), int(cm[label].sum())
                classes.append(dict(**prefix, split=split, label=label, iou=iou if np.isfinite(iou) else None,
                                    precision=tp/predicted if predicted else None,
                                    recall=tp/target if target else None))
        rows.append(row)
    write_rows(folder / "comparison.csv", rows)
    write_rows(folder / "top10.csv", rows[:10])
    write_rows(folder / "classes.csv", classes)
    comparison = holdout_comparison(hold, selected)
    save(folder / "best_weights.json", dict(weights=weight_columns(selected), selected_on="calibration",
         all_rank_1=weight_columns(order[0]), holdout_vs_baseline=comparison,
         by_model_count={k:weight_columns(w) for k,w in group_selected.items()}, baseline_checks=checks))
    lines = ["# Three-model M7 probability ensemble", "",
             "權重先用既有 600 張 calibration 選定，再到既有 200 張 holdout 比較。",
             "完整排名依 800 張的 Image mIoU。邊界為有效 GT 四鄰域轉換兩側。小區域是面積至多 256 像素的四連通 GT 區域，召回率按區域等權平均。",
             f"30:0:70 歷史基準三項指標皆通過，Image mIoU {observed['image_miou']:.6%}。", "",
             "|配置|B2:B4:Residual|Calibration|Holdout|800 張|", "|---|---|---:|---:|---:|"]
    for label, w in [("歷史基準", BASELINE), ("提交配置", selected),
                     *[(f"Calibration 最佳 {k} 模型", w) for k,w in group_selected.items()],
                     ("800 張排名第一", order[0])]:
        scores = [summary[s][w]["image_miou"] for s in partitions]
        lines.append(f"|{label}|{w[0]}:{w[1]}:{w[2]}|{scores[0]:.4%}|{scores[1]:.4%}|{scores[2]:.4%}|")
    lines.extend(["", ("提交配置相對基準的 holdout Image mIoU 差為 {delta_pp:+.3f} 個百分點，"
                      "10,000 次配對 bootstrap 95% CI 為 [{ci95_low_pp:+.3f}, {ci95_high_pp:+.3f}]。").format(**comparison),
                  "", "## 800 張排名 Top 10", "", "|排名|B2:B4:Residual|Image mIoU|", "|---:|---|---:|"])
    for rank,w in enumerate(order[:10],1):
        lines.append(f"|{rank}|{w[0]}:{w[1]}:{w[2]}|{summary['all'][w]['image_miou']:.4%}|")
    (folder / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Finished:", folder.resolve(), weight_columns(selected), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("evaluate")
    sub = actions.add_parser("submit")
    sub.add_argument("--weights", type=int, nargs=3, metavar=("B2", "B4", "RESIDUAL"), required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = cfg["three_model"]
    historical = Path(config["baseline_run"])
    record = read(historical / "config.json")
    spec = {"methods": {"M7": record["m7"]},
            "views": {k: ["B7", *record["views"][k]] for k in record["m7"]}}
    if not record["hflip"]:
        raise ValueError("Historical M7 must use HFlip")
    split = read(Path(cfg["reference_run"]) / "split.json")
    checkpoints = dict(b2=record["checkpoints"]["b2"], b4=cfg["b4_checkpoint"],
                       residual_b4=config["residual_checkpoint"])
    if Path(checkpoints["residual_b4"]).resolve() != Path(record["checkpoints"]["b4"]).resolve():
        raise ValueError("Residual checkpoint differs from the historical baseline")
    if args.action == "submit":
        weights = tuple(args.weights)
        if weights not in WEIGHTS:
            parser.error("Weights must sum to 100 and each be a multiple of 10")
        checkpoints = {k:checkpoints[k] for k,w in zip(MODELS,weights) if w}
    n = notebook_definitions()
    members = load_members(n, checkpoints, spec, split)
    for key, (model, _) in members.items():
        if hasattr(model,"refinement") != (key=="residual_b4"):
            raise ValueError(f"Unexpected architecture for {key}")
    folder = n.new_run(Path("outputs"), "three_model_m7_"+args.action)
    save(folder / "config.json", dict(checkpoints=checkpoints, m7=record["m7"], views=record["views"],
                                     hflip=True, settings=config))
    print(folder.resolve(), flush=True)
    if args.action == "evaluate":
        pairs = n.validation_pairs(ROOT, {"split": split})
        reference = next(r for r in read_rows(historical/"comparison.csv")
                         if (int(r["b2_percent"]), int(r["b4_percent"])) == (30,70))
        parts = read(Path(config["partition_run"]) / "split.json")
        evaluate(n, members, pairs, parts, reference, folder, config)
    else:
        def mask_fn(selected, image, device):
            return prediction(probabilities_for(n,selected,image), weights)

        save(folder / "weights.json", weight_columns(weights))
        output = n.output_for(folder, "submissions") / "submission.csv"
        count = n.generate_submission(ROOT,members,output,"cuda",mask_fn=mask_fn)
        if count != 1000:
            raise ValueError(f"Expected 1000 submission rows, got {count}")
        print(output.resolve(),count,flush=True)


if __name__ == "__main__":
    main()
