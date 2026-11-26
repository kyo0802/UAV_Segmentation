"""Train, evaluate, ensemble and submit the UAV Swin-B UPerNet residual model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from cross_model_m7 import ROOT
from fusion_study import notebook_definitions
from swin_upernet_model import optimizer_groups


CONFIG = Path("configs/swin_upernet_residual.yaml")
OUTPUTS = Path("outputs")
ARTIFACTS = ("config.json", "preprocessor_config.json", "model.safetensors")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolved_config(n, spec):
    base = read_yaml(Path(spec["base_config"]))
    overrides = spec["overrides"]
    config = n.merge(base, overrides)
    source = spec["pretrained"]
    directory = Path(source["directory"])
    config["model"] = dict(overrides["model"], pretrained_dir=str(directory))
    config["training"].pop("learning_rate")
    config["training"].pop("patience")
    architecture = read_json(directory / "config.json")
    processor = read_json(directory / "preprocessor_config.json")
    backbone = architecture["backbone_config"]
    if (architecture["model_type"] != "upernet" or backbone["model_type"] != "swin"
            or backbone["depths"] != [2, 2, 18, 2] or backbone["embed_dim"] != 128
            or not architecture["use_auxiliary_head"] or len(architecture["id2label"]) != 150):
        raise ValueError("Expected a complete Swin-Base ADE20K UPerNet source")
    if (tuple(processor["image_mean"]) != n.MEAN or tuple(processor["image_std"]) != n.STD
            or not processor["do_normalize"] or not processor["do_rescale"]
            or not math.isclose(processor["rescale_factor"], 1 / 255) or not config["normalize"]):
        raise ValueError("Checkpoint preprocessing is incompatible with the UAV image loader")
    config["model"]["architecture"] = architecture
    config["pretrained_source"] = source
    return config


def verify_file(source):
    path = Path(source["directory"]) / "model.safetensors"
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise ValueError(f"Unexpected checksum for {path}: {actual}")


def prepare(n, spec):
    source = spec["pretrained"]
    directory = Path(source["directory"])
    if any(not (directory / name).is_file() for name in ARTIFACTS):
        from huggingface_hub import snapshot_download

        snapshot_download(source["repo"], revision=source["revision"], local_dir=directory,
                          allow_patterns=ARTIFACTS, token=False)
    verify_file(source)
    config = resolved_config(n, spec)
    n.create_model(config["model"])


def fixed_data(n, spec):
    historical = read_yaml(Path(spec["historical_config"]))
    split = read_json(Path(historical["reference_run"]) / "split.json")
    if (len(split["train"]), len(split["validation"])) != (3200, 800):
        raise ValueError("Expected the existing 3200/800 UAV split")
    names = split["train"] + split["validation"]
    n.require_official(ROOT)
    pairs = {pair.image.name: pair for pair in n.training_pairs(ROOT)}
    if len(set(names)) != 4000 or pairs.keys() != set(names):
        raise ValueError("Historical train/validation filenames differ from the dataset")
    return split, [pairs[k] for k in split["train"]], [pairs[k] for k in split["validation"]]


class AuxiliaryLoss(nn.Module):
    def __init__(self, main, weight, ignore_index):
        super().__init__()
        self.main = main
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        main, aux = output
        total = self.main(main, target)
        aux_ce = F.cross_entropy(aux.float(), target, ignore_index=self.ignore_index, reduction="sum")
        return total + self.weight * aux_ce / (target != self.ignore_index).sum().clamp_min(1)


class TrainingProgress:
    def __init__(self, loader, epoch):
        self.loader = loader
        self.dataset = loader.dataset
        self.epoch = epoch

    def __iter__(self):
        started = perf_counter()
        for step, batch in enumerate(self.loader, 1):
            yield batch
            if step % 200 == 0:
                print(f"Epoch {self.epoch:03d} | batch {step}/{len(self.loader)} | "
                      f"{perf_counter() - started:.1f} s | CUDA peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


@torch.inference_mode()
def validate_native(n, model, pairs, config, loss, device):
    model.eval()
    metrics = n.SegmentationMetrics()
    total = 0.0
    for pair in pairs:
        image, mask = n.read_pair(pair)
        target = np.asarray(mask)
        if image.size != (800, 600) or target.shape != (600, 800):
            raise ValueError(f"Unexpected native image or label size: {pair.image}")
        x = n.image_tensor(image.resize(n.resize_wh(config["image_size"]), n.BILINEAR),
                           config["normalize"])[None].to(device)
        with torch.autocast(device_type=device.type, enabled=config["training"]["mixed_precision"]):
            logits = model(x)
        logits = n.resize_logits(logits.float(), target.shape)
        total += loss(logits, torch.from_numpy(target.astype(np.int64)).to(device)[None]).item()
        metrics.add(target, logits[0].argmax(0).byte().cpu().numpy(), pair.image.name)
    return {**metrics.summary(), "val_loss": total / len(pairs)}


def train(n, spec, config, resume=None):
    split, training_pairs, validation_pairs = fixed_data(n, spec)
    settings = config["training"]
    device = n.select_device(config, "full")
    if device.type != "cuda":
        raise ValueError("Swin-B training requires CUDA")
    n.seed_everything(config["seed"])
    n.torch.set_num_threads(4)
    resumed = n.load_checkpoint(n.output_for(resume, "checkpoints") / "last.pt") if resume else None
    if resumed is not None and (resumed["config"] != config or resumed["split"] != split):
        raise ValueError("Resume requires the exact training config and split")
    if resumed is None:
        verify_file(config["pretrained_source"])
    model = n.create_model(config["model"], initialize_pretrained=resumed is None).to(device)
    loader = n.make_loader(training_pairs, config, settings["batch_size"], device, training=True)
    main_loss = n.SegmentationLoss(**config["loss"]).to(device)
    criterion = AuxiliaryLoss(main_loss, settings["auxiliary_weight"], n.IGNORE_INDEX)
    groups = optimizer_groups(model, settings)
    optimizer = torch.optim.AdamW(groups, fused=True)
    updates = math.ceil(len(loader) / settings["accumulation"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: n.learning_rate_factor(
        step, settings["warmup_epochs"] * updates, settings["epochs"] * updates))
    scaler = torch.amp.GradScaler("cuda", enabled=settings["mixed_precision"])

    run = resume or n.new_run(OUTPUTS, "swin_upernet_residual")
    checkpoints = n.output_for(run, "checkpoints")
    history, top = [], []
    if resumed is None:
        n.setup_output(run, config, training_pairs, validation_pairs, settings["batch_size"], device, {})
        n.write_json(run / "pretrained_load.json", model.pretrained_report)
        n.write_json(run / "optimizer_groups.json", [{
            "name": group["group_name"], "lr": group["initial_lr"], "weight_decay": group["weight_decay"],
            "tensor_count": len(group["params"]), "parameter_count": sum(p.numel() for p in group["params"]),
        } for group in groups])
    else:
        model.load_state_dict(resumed["model_state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        history, top = resumed["history"], resumed["top_checkpoints"]
        del resumed
    print(f"Training artifacts: {run.resolve()}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for epoch in range(len(history) + 1, settings["epochs"] + 1):
        n.seed_everything(config["seed"] + epoch)
        first_lr = optimizer.param_groups[0]["lr"]
        train_loss = n.train_epoch(model, TrainingProgress(loader, epoch), criterion, optimizer,
                                   scheduler, scaler, settings, device)
        optimizer.zero_grad(set_to_none=True)
        result = validate_native(n, model, validation_pairs, config, main_loss, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, first_step_lr=first_lr,
                            **{k: result[k] for k in ("val_loss", "image_miou", "global_miou", "pixel_accuracy")}))
        payload = dict(model_state=model.state_dict(), config=config, split=split,
                       dataset="official", mode="full", epoch=epoch)
        candidate = dict(epoch=epoch, image_miou=result["image_miou"], file=f"epoch_{epoch:03d}.pt")
        updated = sorted([*top, candidate], key=lambda row: (-row["image_miou"], row["epoch"]))[:3]
        if candidate in updated:
            n.save_checkpoint(checkpoints / candidate["file"], payload)
        n.save_checkpoint(checkpoints / "last.pt", {
            **payload, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "history": history, "top_checkpoints": updated})
        for old in top:
            if old not in updated:
                (checkpoints / old["file"]).unlink(missing_ok=True)
        top = updated
        n.log_epoch(run, history)
        n.write_json(run / "top3.json", top)
    if not top:
        raise RuntimeError("No trained checkpoints are available")
    n.save_checkpoint(checkpoints / "best.pt", n.load_checkpoint(checkpoints / top[0]["file"]))
    n.write_json(run / "summary.json", dict(epochs=len(history), best_epoch=top[0]["epoch"],
                 image_miou=top[0]["image_miou"], eval_size=[800, 600],
                 elapsed_seconds=perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare")
    commands.add_parser("show-config")
    for action in ("train", "run"):
        command = commands.add_parser(action)
        command.add_argument("--resume", type=Path, metavar="TRAINING_RUN")
    for action in ("evaluate", "finish"):
        command = commands.add_parser(action)
        command.add_argument("--training-run", type=Path, required=True)
    ensemble_cmd = commands.add_parser("ensemble")
    ensemble_cmd.add_argument("--checkpoint", type=Path, required=True)
    submit_cmd = commands.add_parser("submit")
    submit_cmd.add_argument("--checkpoint", type=Path)
    submit_cmd.add_argument("--alpha", type=float, choices=[i / 10 for i in range(11)])
    submit_cmd.add_argument("--selection", type=Path, help="best_ensemble.json from ensemble")
    args = parser.parse_args()
    spec = read_yaml(args.config)
    n = notebook_definitions()
    if args.action == "prepare":
        prepare(n, spec)
    elif args.action == "show-config":
        print(yaml.safe_dump(resolved_config(n, spec), sort_keys=False))
    elif args.action in {"train", "run"}:
        training_run = train(n, spec, resolved_config(n, spec), args.resume)
        if args.action == "run":
            from swin_upernet_inference import finish

            finish(n, spec, training_run)
    else:
        from swin_upernet_inference import ensemble, evaluate, finish, submit

        if args.action == "evaluate":
            evaluate(n, spec, args.training_run)
        elif args.action == "finish":
            finish(n, spec, args.training_run)
        elif args.action == "ensemble":
            print(ensemble(n, spec, args.checkpoint).resolve())
        elif args.selection:
            if args.checkpoint is not None or args.alpha is not None:
                parser.error("Use --selection on its own")
            selected = read_json(args.selection)
            submit(n, spec, Path(selected["checkpoint"]), round(selected["alpha"] * 100))
        else:
            if args.alpha is None or (args.alpha > 0 and args.checkpoint is None):
                parser.error("Supply --alpha and, for a positive alpha, --checkpoint")
            submit(n, spec, args.checkpoint, round(args.alpha * 100))


if __name__ == "__main__":
    main()
