"""Paired native-grid M7 evaluation of original and residual SegFormer-B4."""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import class_postprocess as pp
from cross_model_m7 import ROOT, load_members, model_probability, write_rows
from fusion_study import notebook_definitions
from segmentation_errors import components
from sliding_m7 import Metrics, boundaries


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(pp.CONFIG.read_text(encoding="utf-8"))
    checkpoints, spec, split = pp.source(cfg)
    n = notebook_definitions()
    members = {}
    for key, path in (("original", checkpoints["b4"]), ("residual", args.checkpoint)):
        member = load_members(n, {"b4": path}, spec, split)["b4"]
        if hasattr(member[0], "refinement") != (key == "residual"):
            raise ValueError(f"Unexpected architecture for {key}")
        members[key] = member
    pairs = n.validation_pairs(ROOT, {"split": split})
    if len(pairs) != 800:
        raise ValueError("Expected 800 fixed validation images")
    folder = n.new_run(Path("outputs"), "b4_residual_compare")
    metrics = {k: Metrics(n) for k in members}
    recall = dict.fromkeys(members, 0.)
    count = 0
    for pair in pairs:
        image, gt = n.read_pair(pair)
        gt = np.asarray(gt)
        regions = boundaries(gt)
        ids = components(gt)
        area = np.bincount(ids.ravel())
        small = (area <= 256) & (np.bincount(ids[gt != 255], minlength=len(area)) > 0)
        count += int(small.sum())
        for key, member in members.items():
            prob, _ = model_probability(n, member, image, "cuda")
            mask = pp.array(prob[0].argmax(0))
            metrics[key].add(gt, mask, regions, pair.image.name, 0)
            hits = np.bincount(ids[mask == gt], minlength=len(area))
            recall[key] += float((hits[small]/area[small]).sum())
    rows, classes = [], []
    for key, result in metrics.items():
        summary = result.metric.summary()
        model = members[key][0]
        rows.append(dict(model=key, image_miou=summary["image_miou"], global_miou=summary["global_miou"],
                         boundary_error=result.errors[0]/result.pixels[0],
                         interior_error=result.errors[1]/result.pixels[1],
                         small_component_recall=recall[key]/count if count else None,
                         small_components=count, parameters=sum(p.numel() for p in model.parameters()),
                         added_parameters=sum(p.numel() for p in model.refinement.parameters()) if key=="residual" else 0))
        classes.extend(dict(model=key,label=i,iou=v) for i,v in enumerate(summary["per_class_iou"]))
    write_rows(folder/"comparison.csv",rows)
    write_rows(folder/"classes.csv",classes)
    print(folder.resolve(),rows)


if __name__ == "__main__":
    main()
