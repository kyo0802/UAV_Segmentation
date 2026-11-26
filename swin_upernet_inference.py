"""M7 checkpoint selection, cross-architecture evaluation and RLE export."""
from pathlib import Path

import numpy as np
import torch
import yaml

from class_postprocess import class_stats, read, save
from cross_model_m7 import GRID, ROOT, load_members, model_probability, read_rows, write_rows
from segmentation_errors import components
from sliding_m7 import Metrics, boundaries


OUTPUTS = Path("outputs")
SWIN = "swin"


def evaluation_context(spec):
    cfg = yaml.safe_load(Path(spec["historical_config"]).read_text(encoding="utf-8"))
    historical = cfg["three_model"]
    baseline_dir = Path(historical["baseline_run"])
    record = read(baseline_dir / "config.json")
    m7 = {"methods": {"M7": record["m7"]},
          "views": {k: ["B7", *record["views"][k]] for k in record["m7"]}}
    if not record["hflip"] or len(record["m7"]) != 5:
        raise ValueError("Expected historical five-scale HFlip M7")
    split = read(Path(cfg["reference_run"]) / "split.json")
    part = read(Path(historical["partition_run"]) / "split.json")
    names = part["calibration"] + part["holdout"]
    if ((len(part["calibration"]), len(part["holdout"])) != (600, 200)
            or len(set(names)) != 800 or set(names) != set(split["validation"])):
        raise ValueError("Historical calibration/holdout differs from the fixed split")
    reference = next(row for row in read_rows(baseline_dir / "comparison.csv")
                     if (int(row["b2_percent"]), int(row["b4_percent"])) == (30, 70))
    paths = {"b2": record["checkpoints"]["b2"], "residual_b4": record["checkpoints"]["b4"]}
    return m7, split, part, reference, paths


def validation_index(n, split):
    pairs = n.validation_pairs(ROOT, {"split": split})
    indexed = {p.image.name: p for p in pairs}
    if len(pairs) != 800 or indexed.keys() != set(split["validation"]):
        raise ValueError("Validation pairs do not match the fixed 800 images")
    return indexed


def load_swin(n, checkpoint, m7, split):
    model, state = n.load_trained(Path(checkpoint), "cuda", "official")
    config = state["config"]
    if (state["mode"] != "full" or state["split"] != split
            or config["model"]["name"] != "swin_upernet_residual"):
        raise ValueError("Swin checkpoint architecture or validation split is incompatible")
    views = []
    for view, weight in m7["methods"]["M7"].items():
        _, width, height = m7["views"][view]
        view_cfg = n.merge(config, dict(image_size=[width, height],
            inference=dict(tile_size=None, hflip=True, vflip=False, scales=[1.0])))
        views.append((view, view_cfg, weight))
    return model, views


def hard_mask(probability):
    if probability.dtype != torch.float32 or probability.shape != (1, 16, 600, 800):
        raise ValueError("M7 must produce native FP32 probabilities")
    return probability[0].argmax(0).byte().cpu().numpy()


@torch.inference_mode()
def evaluate(n, spec, training_run):
    """Rank the training top 3 using calibration before reading holdout predictions."""
    m7, split, part, _, _ = evaluation_context(spec)
    indexed = validation_index(n, split)
    checkpoint_dir = n.output_for(training_run, "checkpoints")
    candidates = read(training_run / "top3.json")
    if len(candidates) != 3 or not (training_run / "summary.json").is_file():
        raise ValueError("M7 selection requires three checkpoints and completed training")
    out = n.new_run(OUTPUTS, "swin_m7_evaluate")
    rows = []
    for candidate in candidates:
        checkpoint = checkpoint_dir / candidate["file"]
        member = load_swin(n, checkpoint, m7, split)
        metrics = n.SegmentationMetrics()
        for index, name in enumerate(part["calibration"], 1):
            image, target = n.read_pair(indexed[name])
            probability = model_probability(n, member, image, "cuda")[0]
            metrics.add(np.asarray(target), hard_mask(probability), name)
            if index % 100 == 0:
                print(f"M7 epoch {candidate['epoch']}: {index}/600", flush=True)
        scores = metrics.summary()
        rows.append(dict(epoch=candidate["epoch"], checkpoint=str(checkpoint.resolve()),
                         training_validation_image_miou=candidate["image_miou"],
                         m7_calibration_image_miou=scores["image_miou"],
                         m7_calibration_global_miou=scores["global_miou"]))
        del member
    rows.sort(key=lambda row: (-row["m7_calibration_image_miou"], row["epoch"]))
    write_rows(out / "top3_m7.csv", rows)
    save(out / "best_checkpoint.json", rows[0])
    print(f"M7 artifacts: {out.resolve()}", flush=True)
    return out


def fuse(probabilities, alpha):
    if alpha == 100:
        return probabilities[SWIN]
    baseline = probabilities["b2"] * 0.3 + probabilities["residual_b4"] * 0.7
    return fuse_baseline(baseline, probabilities.get(SWIN), alpha)


def fuse_baseline(baseline, swin, alpha):
    if alpha == 0:
        return baseline
    if alpha == 100:
        return swin
    return baseline * ((100 - alpha) / 100) + swin * (alpha / 100)


def measure(n, members, names, indexed, partition):
    keys = [*GRID, "residual_b4"]
    metrics = {k: Metrics(n) for k in keys}
    recall = dict.fromkeys(keys, 0.0)
    small_count = 0
    for i, name in enumerate(names, 1):
        image, mask = n.read_pair(indexed[name])
        gt = np.asarray(mask)
        regions = boundaries(gt)
        ids = components(gt)
        area = np.bincount(ids.ravel())
        small = (area <= 256) & (np.bincount(ids[gt != 255], minlength=len(area)) > 0)
        small_count += int(small.sum())
        probabilities = {k: model_probability(n, member, image, "cuda")[0] for k, member in members.items()}
        baseline = fuse(probabilities, 0)
        for k, metric in metrics.items():
            probability = (probabilities[k] if k == "residual_b4"
                           else fuse_baseline(baseline, probabilities[SWIN], k))
            pred = hard_mask(probability)
            metric.add(gt, pred, regions, name, 0)
            hits = np.bincount(ids[pred == gt], minlength=len(area))
            recall[k] += float((hits[small] / area[small]).sum())
        if i % 100 == 0:
            print(f"Ensemble {partition}: {i}/{len(names)}", flush=True)
    return metrics, recall, small_count


def record_scores(metric, recall, count):
    scores = metric.summary()
    keys = ("image_miou", "global_miou", "pixel_accuracy", "boundary_error_rate", "interior_error_rate")
    return dict(**{key: scores[key] for key in keys}, small_component_recall=recall / count if count else None)


def paired_holdout(candidate, baseline):
    differences = [a["miou"] - b["miou"]
                   for a, b in zip(candidate.metric.per_image, baseline.metric.per_image, strict=True)
                   if a["miou"] is not None and b["miou"] is not None]
    delta = 100 * np.asarray(differences)
    if not delta.size:
        raise ValueError("No valid holdout image scores")
    rng = np.random.default_rng(42)
    draws = rng.integers(delta.size, size=(10000, delta.size))
    low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
    return dict(delta_pp=float(delta.mean()), ci95_low_pp=float(low), ci95_high_pp=float(high))


@torch.inference_mode()
def ensemble(n, spec, checkpoint):
    m7, split, part, reference, paths = evaluation_context(spec)
    indexed = validation_index(n, split)
    members = load_members(n, paths, m7, split)
    members[SWIN] = load_swin(n, checkpoint, m7, split)
    out = n.new_run(OUTPUTS, "swin_m7_ensemble")
    save(out / "split.json", part)
    print(f"Ensemble artifacts: {out.resolve()}", flush=True)
    cal, cal_recall, cal_count = measure(n, members, part["calibration"], indexed, "calibration")
    order = sorted(GRID, key=lambda k: (-cal[k].summary()["image_miou"], k))
    best = order[0]
    save(out / "calibration_selection.json", dict(alpha=best / 100, checkpoint=str(Path(checkpoint).resolve())))
    hold, hold_recall, hold_count = measure(n, members, part["holdout"], indexed, "holdout")
    full = {k: cal[k].merged(n, hold[k]) for k in cal}
    full_recall = {k: cal_recall[k] + hold_recall[k] for k in cal}
    checks = {}
    baseline = full[0].summary()
    for k in ("image_miou", "global_miou", "pixel_accuracy"):
        expected = float(reference[k])
        checks[k] = dict(expected=expected, observed=baseline[k], passed=abs(baseline[k] - expected) <= 1e-6)
    save(out / "baseline_checks.json", checks)
    if not all(v["passed"] for v in checks.values()):
        raise ValueError("The alpha=0 baseline does not match the historical results")
    rows, classes = [], []
    for k in [*order, "residual_b4"]:
        header = dict(model="residual_b4" if k == "residual_b4" else "ensemble",
                      alpha=None if k == "residual_b4" else k / 100)
        row = dict(**header, calibration_rank=order.index(k)+1 if k in order else None)
        for label, stats, recalls, count in (("calibration", cal, cal_recall, cal_count),
                    ("holdout", hold, hold_recall, hold_count), ("all", full, full_recall, cal_count + hold_count)):
            row.update({f"{label}_{metric}": value for metric, value in record_scores(stats[k], recalls[k], count).items()})
            classes.extend(dict(**header, split=label, **r) for r in class_stats(stats[k].metric.matrix))
        rows.append(row)
    write_rows(out / "comparison.csv", rows)
    write_rows(out / "classes.csv", classes)
    delta = paired_holdout(hold[best], hold[0])
    selected = dict(alpha=best / 100, checkpoint=str(Path(checkpoint).resolve()), holdout_vs_baseline=delta)
    save(out / "best_ensemble.json", selected)
    lines = ["# Swin-B UPerNet residual M7 成績", "",
             f"Calibration 選出的 Swin 權重為 {best/100:.1f}。",
             ("Holdout Image mIoU 相對基準差 {delta_pp:+.3f} 個百分點，"
              "10,000 次配對 bootstrap 的 95% CI 為 [{ci95_low_pp:+.3f}, {ci95_high_pp:+.3f}]。").format(**delta),
             "Alpha=0 通過歷史 Image mIoU、Global mIoU 與 pixel accuracy 核對。",
             "Top 3 候選先用全體 800 張逐輪驗證篩選，再以 600 張 calibration 選 M7 checkpoint 和融合權重。Holdout 已參與逐輪 checkpoint 篩選。",
             "", "|配置|Calibration Image mIoU|Holdout Image mIoU|全體 Image mIoU|Global mIoU|邊界錯誤率|內部錯誤率|Small Component Recall|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = row["model"] if row["alpha"] is None else f"alpha={row['alpha']:.1f}"
        lines.append("|"+label+"|{calibration_image_miou:.4%}|{holdout_image_miou:.4%}|"
                     "{all_image_miou:.4%}|{all_global_miou:.4%}|{all_boundary_error_rate:.4%}|"
                     "{all_interior_error_rate:.4%}|{all_small_component_recall:.4%}|".format(**row))
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return out


@torch.inference_mode()
def submit(n, spec, checkpoint, alpha):
    m7, split, _, _, paths = evaluation_context(spec)
    members = {} if alpha == 100 else load_members(n, paths, m7, split)
    if alpha:
        members[SWIN] = load_swin(n, checkpoint, m7, split)
    out = n.new_run(OUTPUTS, "swin_m7_submit")
    save(out / "selection.json", dict(alpha=alpha/100, checkpoint=str(checkpoint)))

    def mask_fn(selected, image, device):
        probabilities = {k: model_probability(n, member, image, device)[0] for k, member in selected.items()}
        return hard_mask(fuse(probabilities, alpha))

    path = n.output_for(out, "submissions") / "submission.csv"
    count = n.generate_submission(ROOT, members, path, "cuda", mask_fn=mask_fn)
    if count != 1000:
        raise ValueError(f"Expected 1000 submission rows, got {count}")
    print(f"Validated RLE submission: {path.resolve()} ({count} images)", flush=True)
    return path


def finish(n, spec, training_run):
    evaluation = evaluate(n, spec, training_run)
    checkpoint = Path(read(evaluation / "best_checkpoint.json")["checkpoint"])
    fusion = ensemble(n, spec, checkpoint)
    best = read(fusion / "best_ensemble.json")
    path = submit(n, spec, checkpoint, round(best["alpha"] * 100))
    save(fusion / "artifacts.json", dict(training=str(training_run.resolve()), m7=str(evaluation.resolve()),
         ensemble=str(fusion.resolve()), submission=str(path.resolve())))
