"""Calibration-only post-processing search on the recorded B7/B10 M7 ensemble."""
import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from cross_model_m7 import ROOT, load_members, model_probability, write_rows
from fusion_study import notebook_definitions


CONFIG = Path("configs/m7_postprocess.yaml")
CLASSES = 16


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@dataclass(frozen=True)
class Params:
    weights: tuple = (1.0,) * CLASSES
    thresholds: tuple = (0.0,) * CLASSES
    global_threshold: float = 0.0


def changed(params, axis, cls, value):
    if axis == "global_threshold":
        return replace(params, global_threshold=value)
    values = list(getattr(params, axis))
    values[cls] = value
    return replace(params, **{axis: tuple(values)})


def grid(bounds):
    first, last, step = bounds
    return [v / 100 for v in range(first, last + 1, step)]


def source(cfg):
    run = Path(cfg["fusion_run"])
    record = read(run / "config.json")
    if cfg["b2_percent"] != 40 or not record["hflip"] or len(record["m7"]) != 5:
        raise ValueError("Expected the recorded 40:60 five-view HFlip ensemble")
    # Iterate the historical M7 mapping in its recorded summation order.
    spec = {"methods": {"M7": record["m7"]},
            "views": {k: ["B7", *record["views"][k]] for k in record["m7"]}}
    split = read(Path(record["reference_run"]) / "split.json")
    return record["checkpoints"], spec, split


def fusion(n, members, image):
    b2 = model_probability(n, members["b2"], image, "cuda")[0]
    b4 = model_probability(n, members["b4"], image, "cuda")[0]
    p = (b2 * 0.4 + b4 * 0.6)[0]
    if p.dtype != torch.float32 or p.shape != (CLASSES, 600, 800):
        raise ValueError("Expected native FP32 probabilities")
    return p


def choose(p, primary, eligible, params_t, low):
    # If every class fails, the selected primary is retained.
    best_value, best_id = eligible
    confidence = p.gather(0, primary.unsqueeze(0))[0]
    fallback = (confidence < params_t[primary]) & (best_value >= 0)
    return torch.where(low, 0, torch.where(fallback, best_id, primary))


def predict(p, params):
    weights = p.new_tensor(params.weights)
    thresholds = p.new_tensor(params.thresholds)
    primary = (p * weights[:, None, None]).argmax(0)
    eligible = p.masked_fill(p < thresholds[:, None, None], -1).max(0)
    return choose(p, primary, eligible, thresholds, p.amax(0) < params.global_threshold)


def insert_class(other, value, cls):
    best, label = other
    take = (value > best) | ((value == best) & (cls < label))
    return torch.maximum(best, value), torch.where(take, cls, label)


def sweep_masks(p, params, axis, cls, values):
    # Cache the best of the 15 unchanged classes; preserve first-index argmax ties.
    weights = p.new_tensor(params.weights)
    thresholds = p.new_tensor(params.thresholds)
    scores = p * weights[:, None, None]
    allowed = p.masked_fill(p < thresholds[:, None, None], -1)
    confidence = p.amax(0)
    low = confidence < params.global_threshold
    if axis == "global_threshold":
        labels = choose(p, scores.argmax(0), allowed.max(0), thresholds, torch.zeros_like(low))
        for value in values:
            yield torch.where(confidence < value, 0, labels)
    elif axis == "weights":
        scores[cls].fill_(-1)
        other = scores.max(0)
        eligible = allowed.max(0)
        for value in values:
            _, primary = insert_class(other, p[cls] * value, cls)
            yield choose(p, primary, eligible, thresholds, low)
    else:
        primary = scores.argmax(0)
        allowed[cls].fill_(-1)
        other = allowed.max(0)
        for value in values:
            thresholds[cls] = value
            candidate = p[cls].masked_fill(p[cls] < value, -1)
            yield choose(p, primary, insert_class(other, candidate, cls), thresholds, low)


def array(labels):
    return labels.byte().cpu().numpy()


class Search:
    def __init__(self, n, samples, folder):
        self.n = n
        self.samples = samples
        self.folder = folder
        self.rows = []

    def scan(self, stage, base, axes, grids, iteration=0):
        # Axes in an independent sweep share one disk read per calibration image.
        print(f"Calibration: {stage}, pass {iteration}, {axes[0]} through {axes[-1]}", flush=True)
        totals = {a: np.zeros(len(grids[a[0]])) for a in axes}
        counts = {a: np.zeros(len(grids[a[0]]), dtype=int) for a in axes}
        for p, gt, _ in self.samples():
            for axis, cls in axes:
                key = axis, cls
                for i, mask in enumerate(sweep_masks(p, base, axis, cls, grids[axis])):
                    cm = self.n.confusion_matrix(gt, array(mask))
                    score = self.n.mean_iou(self.n.iou_values(cm))
                    if score is not None:
                        totals[key][i] += score
                        counts[key][i] += 1
        winners = {}
        for axis, cls in axes:
            key = axis, cls
            if not counts[key].all():
                raise ValueError("No valid calibration scores")
            values = grids[axis]
            current = base.global_threshold if cls is None else getattr(base, axis)[cls]
            scores = totals[key] / counts[key]
            # Keep current value on ties, then prefer the closest change.
            best = max(range(len(values)), key=lambda i: (scores[i], -abs(values[i] - current)))
            winners[key] = values[best]
            for i, v in enumerate(values):
                self.rows.append(dict(stage=stage, iteration=iteration, parameter=axis,
                                      label=cls, value=v, image_miou=scores[i],
                                      selected=i == best, base=json.dumps(asdict(base))))
        write_rows(self.folder / "calibration_search.csv", self.rows)
        return winners

    def descend(self, stage, start, axes, grids, limit):
        params = start
        for iteration in range(1, limit + 1):
            old = params
            for key in axes:
                value = self.scan(stage, params, [key], grids, iteration)[key]
                params = changed(params, *key, value)
            if params == old:
                break
        return params


def summarize(n, samples, methods):
    metrics = {k: n.SegmentationMetrics() for k in methods}
    for p, target, name in samples:
        for key, params in methods.items():
            metrics[key].add(target, array(predict(p, params)), name)
    return metrics


def class_stats(cm):
    total = cm.sum()
    for c in range(CLASSES):
        tp = int(cm[c, c])
        gt, pred = int(cm[c].sum()), int(cm[:, c].sum())
        fp, fn = pred - tp, gt - tp
        yield dict(label=c, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                   precision=tp/pred if pred else None, recall=tp/gt if gt else None,
                   fp=fp, fn=fn, gt_ratio=gt/total, predicted_ratio=pred/total,
                   bias="over" if pred > gt else "under" if gt > pred else "equal")


def report(folder, partitions, best):
    summaries, per_class, per_image = [], [], []
    for part, metrics in partitions.items():
        baseline = list(class_stats(metrics["baseline"].matrix))
        for method, metric in metrics.items():
            s = metric.summary()
            summaries.append(dict(split=part, method=method, images=len(metric.per_image),
                                  **{k:s[k] for k in ("image_miou","global_miou","pixel_accuracy")}))
            for row in class_stats(metric.matrix):
                old = baseline[row["label"]]["iou"]
                delta = None if old is None or row["iou"] is None else 100*(row["iou"]-old)
                per_class.append(dict(split=part, method=method, **row, delta_iou_pp=delta))
            per_image.extend(dict(split=part, method=method, **r) for r in metric.per_image)
    write_rows(folder/"comparison.csv", summaries)
    write_rows(folder/"classes.csv", per_class)
    write_rows(folder/"per_image.csv", per_image)
    lines = ["# M7 後處理消融", "", f"Calibration 選出的提交設定為 **{best}**。",
             "以逐張 Image mIoU 的平均值選擇參數。Holdout 僅比較已固定的方法。",
             "", "|切分|方法|Image mIoU|Global mIoU|Pixel accuracy|","|---|---|---:|---:|---:|"]
    for r in summaries:
        lines.append("|{split}|{method}|{image_miou:.6f}|{global_miou:.6f}|{pixel_accuracy:.6f}|".format(**r))
    lines += ["", "## 全部 800 張的 baseline 預測偏差", "",
              "|Class|GT %|預測 %|偏差|", "|---:|---:|---:|---|"]
    for r in per_class:
        if r["split"] == "all" and r["method"] == "baseline":
            lines.append(f"|{r['label']}|{100*r['gt_ratio']:.3f}|{100*r['predicted_ratio']:.3f}|{r['bias']}|")
    lines += ["", "## 選定方法在 holdout 的各類別 IoU 改善", "",
              "|Class|IoU 差（百分點）|","|---:|---:|"]
    for r in per_class:
        if r["split"] == "holdout" and r["method"] == best:
            delta = r["delta_iou_pp"]
            lines.append(f"|{r['label']}|{delta:+.3f}|" if delta is not None else f"|{r['label']}|N/A|")
    (folder/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def study(n, cfg, folder, members, split, cache):
    pairs = n.validation_pairs(ROOT, {"split":split})
    pairs = sorted(pairs, key=lambda pair: pair.image.name)
    nc, nh = cfg["calibration_images"], cfg["holdout_images"]
    if ((nc, nh) != (600, 200) or len(pairs) != nc+nh
            or len(set(p.image.name for p in pairs)) != nc+nh):
        raise ValueError("Expected 800 unique fixed validation samples")
    rng = np.random.default_rng(cfg["split_seed"])
    ordered = [pairs[i] for i in rng.permutation(len(pairs))]
    calibration, holdout = ordered[:nc], ordered[nc:]
    save(folder/"split.json", {k:[p.image.name for p in items] for k,items in
                               (("calibration",calibration),("holdout",holdout))})
    if shutil.disk_usage(cache).free < nc * 600 * 800 * (4*CLASSES+1) + 50_000_000:
        raise ValueError("Calibration cache requires about 17.5 GiB of free space")
    for i, pair in enumerate(calibration):
        image, target = n.read_pair(pair)
        np.save(cache/f"{i}.npy", fusion(n,members,image).cpu().numpy())
        np.save(cache/f"{i}_gt.npy", np.asarray(target, dtype=np.uint8))
        if (i + 1) % 100 == 0:
            print(f"Cached calibration probabilities: {i+1}/{nc}", flush=True)
    def samples():
        for i, pair in enumerate(calibration):
            yield (torch.from_numpy(np.load(cache/f"{i}.npy")).to("cuda"),
                   np.load(cache/f"{i}_gt.npy"), pair.image.name)

    def final_samples():
        for pair in holdout:
            image, target = n.read_pair(pair)
            yield fusion(n,members,image), np.asarray(target), pair.image.name

    finish_study(n, cfg, folder, samples, final_samples, Search(n, samples, folder))


def finish_study(n, cfg, folder, samples, final_samples, search):
    grids = {"weights":grid(cfg["weight_percent"]),
             "thresholds":[0.0, *grid(cfg["class_threshold_percent"])],
             "global_threshold":grid(cfg["global_threshold_percent"])}
    weights = [("weights", c) for c in range(CLASSES)]
    thresholds = [("thresholds", c) for c in range(CLASSES)]
    global_axis = [("global_threshold", None)]
    axes = global_axis + weights + thresholds
    baseline = Params()
    independent = search.scan("one_at_a_time", baseline, axes, grids)
    methods = {"baseline":baseline}
    for name, group in (("global", global_axis), ("weight_single", weights),
                        ("threshold_single", thresholds)):
        # Report the strongest individually tested class as its own ablation.
        candidates = [r for r in search.rows if r["selected"] and (r["parameter"],r["label"]) in group]
        r = max(candidates,key=lambda row:row["image_miou"])
        methods[name] = changed(baseline,r["parameter"],r["label"],r["value"])
    limit = cfg["max_coordinate_passes"]
    if limit < 1:
        raise ValueError("max_coordinate_passes must be positive")
    for name, group in (("scaling",weights), ("class_threshold",thresholds)):
        start = methods["weight_single" if name=="scaling" else "threshold_single"]
        methods[name] = search.descend(name, start, group, grids, limit)
    cal = summarize(n, samples(), methods)
    strongest = max(methods, key=lambda k:cal[k].summary()["image_miou"])
    methods["joint"] = search.descend("joint", methods[strongest], axes, grids, limit)
    cal.update(summarize(n, samples(), {"joint":methods["joint"]}))
    best = max(methods, key=lambda k:cal[k].summary()["image_miou"])
    save(folder/"best_params.json", dict(method=best, params=asdict(methods[best]), config=cfg,
         independent=[dict(parameter=a,label=c,value=v) for (a,c),v in independent.items()],
         methods={k:asdict(v) for k,v in methods.items()}))
    held = summarize(n, final_samples(), methods)
    combined = {}
    for key in methods:
        metric = n.SegmentationMetrics()
        metric.matrix = cal[key].matrix + held[key].matrix
        metric.per_image = cal[key].per_image + held[key].per_image
        combined[key] = metric
    report(folder, {"calibration":cal, "holdout":held, "all":combined}, best)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    ev = sub.add_parser("study")
    ev.add_argument("--cache-dir", type=Path, help="Parent directory for temporary FP32 calibration cache")
    submit = sub.add_parser("submit")
    submit.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "submit":
        artifact = read(args.params)
        cfg = artifact["config"]
    else:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    checkpoints, spec, split = source(cfg)
    n = notebook_definitions()
    members = load_members(n, checkpoints, spec, split)
    folder = n.new_run(Path("outputs"), "m7_postprocess_"+args.action)
    print(folder.resolve(), flush=True)
    if args.action == "study":
        parent = args.cache_dir or folder
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent, prefix="calibration_") as cache:
            study(n, cfg, folder, members, split, Path(cache))
    else:
        params = Params(**artifact["params"])
        def mask_fn(selected, image, device):
            return array(predict(fusion(n,selected,image), params))
        output = n.output_for(folder,"submissions")/"submission.csv"
        count = n.generate_submission(ROOT,members,output,"cuda",mask_fn=mask_fn)
        print(output.resolve(), count, flush=True)


if __name__ == "__main__":
    main()
