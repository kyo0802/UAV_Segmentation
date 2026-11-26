"""Native-resolution sliding-window ablation for the fixed B7/B10 M7 fusion."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import class_postprocess as pp
from cross_model_m7 import ROOT, load_members, read_rows, write_rows
from fusion_study import notebook_definitions


CONFIG = Path("configs/sliding_m7.yaml")
HW = (600, 800)


@dataclass(frozen=True)
class Method:
    tile: str = "global"
    model: str = "fusion"
    alpha: float = 0.0
    post: bool = False

    def name(self):
        return f"{self.tile}_{self.model}_a{self.alpha:g}" + ("_post" if self.post else "")


GLOBAL = Method()


def timed(fn, *args):
    torch.cuda.synchronize()
    start = perf_counter()
    result = fn(*args)
    torch.cuda.synchronize()
    return result, perf_counter() - start


class Layout:
    def __init__(self, n, settings, sigma_fraction):
        self.size = size = settings["size"]
        step = int(size * (1 - settings["overlap"]))
        if not 0 < step <= size <= min(HW) or sigma_fraction <= 0:
            raise ValueError("Invalid native tile geometry")
        coord = (torch.arange(size, device="cuda", dtype=torch.float32) - (size-1)/2) / (size*sigma_fraction)
        bell = (-0.5 * coord.square()).exp()
        self.weight = bell[:, None] * bell[None, :]
        self.areas = [(slice(y,y+size), slice(x,x+size))
                      for y in n.tile_starts(HW[0],size,step)
                      for x in n.tile_starts(HW[1],size,step)]
        self.coverage = self.weight.new_zeros(HW)
        for area in self.areas:
            self.coverage[area] += self.weight
        if not torch.isfinite(self.coverage).all() or self.coverage.min().item() <= 0:
            raise ValueError("Invalid Gaussian coverage")

    def probability(self, n, member, image):
        if image.size != HW[::-1]:
            raise ValueError(f"Expected 800x600, got {image.size}")
        model, views = member
        normalize = views[0][1]["normalize"]
        native = n.image_tensor(image, normalize)[None].to("cuda")
        total = self.weight.new_zeros((n.NUM_CLASSES, *HW))
        for y, x in self.areas:
            tile = native[:, :, y, x].contiguous()
            logits = model(tile).float()
            flipped = model(tile.flip((-1,))).float().flip((-1,))
            if logits.shape[-2:] != tile.shape[-2:]:
                raise ValueError("SegFormer output must align with the native tile")
            probability = ((logits + flipped) * 0.5).softmax(1)[0]
            total[:, y, x].addcmul_(probability, self.weight)
        return total.div_(self.coverage)


def mix(a, b, share):
    return a * (1-share) + b * share


def boundaries(target):
    valid = target != 255
    b = np.zeros_like(valid)
    for axis in (0, 1):
        t, v, edge = (a.swapaxes(0, axis) for a in (target, valid, b))
        transition = v[1:] & v[:-1] & (t[1:] != t[:-1])
        edge[1:] |= transition
        edge[:-1] |= transition
    return b, valid & ~b


class Metrics:
    def __init__(self, n):
        self.metric = n.SegmentationMetrics()
        self.errors = np.zeros(2, dtype=np.int64)
        self.pixels = np.zeros(2, dtype=np.int64)
        self.seconds = 0.

    def add(self, target, mask, regions, name, seconds):
        self.metric.add(target, mask, name)
        wrong = target != mask
        for i, region in enumerate(regions):
            self.errors[i] += np.count_nonzero(wrong & region)
            self.pixels[i] += np.count_nonzero(region)
        self.seconds += seconds

    def merged(self, n, other):
        out = Metrics(n)
        out.metric.matrix = self.metric.matrix + other.metric.matrix
        out.metric.per_image = self.metric.per_image + other.metric.per_image
        out.errors = self.errors + other.errors
        out.pixels = self.pixels + other.pixels
        out.seconds = self.seconds + other.seconds
        return out

    def summary(self):
        m = self.metric.matrix
        s = self.metric.summary()
        rates = [float(err/count) if count else None for err,count in zip(self.errors,self.pixels)]
        return dict(image_miou=s["image_miou"],global_miou=s["global_miou"],
                    pixel_accuracy=s["pixel_accuracy"],boundary_error_rate=rates[0],
                    interior_error_rate=rates[1],estimated_inference_seconds=self.seconds,
                    images=len(self.metric.per_image),gt1_pred3=int(m[1,3]),gt3_pred1=int(m[3,1]),
                    gt1_pred0=int(m[1,0]),gt0_pred1=int(m[0,1]))


def classify(prob, method, params):
    labels = pp.predict(prob, params) if method.post else prob.argmax(0)
    return pp.array(labels)


def probabilities(n, members, layouts, cfg, image):
    global_p, global_s = timed(pp.fusion, n, members, image)
    yield GLOBAL, global_p, global_s
    for tile, layout in layouts.items():
        local = {}
        cost = 0.
        for model in ("b2","b4"):
            local[model], seconds = timed(layout.probability, n, members[model], image)
            cost += seconds
            yield Method(tile,model,1.), local[model], seconds
        local_p, blend_s = timed(mix, local["b2"],local["b4"],0.6)
        del local
        cost += blend_s
        for alpha in cfg["alphas"]:
            if alpha == 0:
                continue  # Identical for S1–S4.
            if alpha == 1:
                yield Method(tile,"fusion",alpha), local_p, cost
            else:
                p, seconds = timed(mix, global_p, local_p, alpha)
                yield Method(tile,"fusion",alpha), p, global_s + cost + seconds


def measure(n, members, layouts, cfg, pairs, params, post_methods=None):
    results = defaultdict(lambda: Metrics(n))
    for index, pair in enumerate(pairs,1):
        image, mask = n.read_pair(pair)
        target = np.asarray(mask)
        if target.shape != HW:
            raise ValueError(f"Invalid validation mask: {pair.image}")
        regions = boundaries(target)
        for method, p, seconds in probabilities(n,members,layouts,cfg,image):
            # Retain scalar calibration scores for frozen PP without caching/re-inferring maps.
            versions = [method]
            if post_methods is None or method in post_methods:
                versions.append(replace(method,post=True))
            for choice in versions:
                labels, decision_s = timed(classify,p,choice,params)
                results[choice].add(target,labels,regions,pair.image.name,seconds+decision_s)
        if index % 50 == 0:
            print(f"{index}/{len(pairs)}",flush=True)
    return results


def baseline_check(metrics, reference, tolerance):
    observed = metrics.summary()
    checks = {}
    for key in ("image_miou","global_miou","pixel_accuracy"):
        expected = float(reference[key])
        checks[key] = dict(expected=expected, observed=observed[key],
                           passed=abs(observed[key]-expected)<=tolerance)
    return checks


def write_results(folder, partitions, checks, sliding, chosen):
    rows, classes, images = [],[],[]
    for part, results in partitions.items():
        ordered = sorted(results.items(),key=lambda item:-item[1].metric.summary()["image_miou"])
        for rank,(method,metric) in enumerate(ordered,1):
            prefix = dict(split=part,method=method.name())
            rows.append(dict(**prefix,rank=rank,**asdict(method),**metric.summary()))
            classes.extend(dict(**prefix,**row) for row in pp.class_stats(metric.metric.matrix))
            images.extend(dict(**prefix,**row) for row in metric.metric.per_image)
    write_rows(folder/"comparison.csv",rows)
    write_rows(folder/"classes.csv",classes)
    write_rows(folder/"per_image.csv",images)
    pp.save(folder/"results.json",dict(comparison=rows,classes=classes,baseline_checks=checks))
    lines = ["# Sliding window 消融","",f"Calibration 選出的原始滑窗設定為 {sliding.name()}，提交設定為 {chosen.name()}。",
             "原始設定依 calibration Image mIoU 選取，再與固定後處理及全域基準比較。分數相同時優先保留全域基準。",
             "邊界為有效 GT 中四鄰域類別轉換兩側的像素；其餘有效像素為內部。錯誤率依各區域累計像素計算。",
             "推論秒數加總該方法需要的已同步 GPU 路徑、融合、分類與遮罩回傳時間，不含讀檔、計分與共用視窗幾何初始化。",
             "", "|切分|方法|排名|Image mIoU|Global mIoU|邊界錯誤率|內部錯誤率|秒|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("|{split}|{method}|{rank}|{image_miou:.6f}|{global_miou:.6f}|"
                     "{boundary_error_rate:.6f}|{interior_error_rate:.6f}|{estimated_inference_seconds:.1f}|".format(**r))
    held = {r["method"]:r for r in rows if r["split"]=="holdout"}
    lines.extend(["","## Holdout 改善幅度","",
                  "|設定|參考設定|Image mIoU 差（百分點）|邊界錯誤率差（百分點）|內部錯誤率差（百分點）|",
                  "|---|---|---:|---:|---:|"])
    for test, reference in ((sliding,GLOBAL),(replace(sliding,post=True),sliding),
                            (replace(sliding,post=True),replace(GLOBAL,post=True))):
        a,b = held[test.name()],held[reference.name()]
        diff = [100*(a[k]-b[k]) for k in ("image_miou","boundary_error_rate","interior_error_rate")]
        lines.append(f"|{test.name()}|{reference.name()}|{diff[0]:+.3f}|{diff[1]:+.3f}|{diff[2]:+.3f}|")
    lines.extend(["","## Alpha=0 核對",""])
    for part, values in checks.items():
        for key,v in values.items():
            lines.append(f"- {part} {key}：實測 {v['observed']:.9f}，參考 {v['expected']:.9f}，通過 {v['passed']}")
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def study(n, cfg, folder, history, model_source, fixed, split, members, layouts):
    pairs = n.validation_pairs(ROOT,{"split":split})
    index = {pair.image.name:pair for pair in pairs}
    parts = pp.read(history/"split.json")
    cal, held = parts["calibration"],parts["holdout"]
    if (len(cal),len(held)) != (600,200) or len(set(cal+held))!=800 or set(cal+held)!=set(index):
        raise ValueError("Historical partition does not match the fixed 800-image validation split")
    pp.save(folder/"split.json",parts)
    ref = {r["split"]:r for r in read_rows(history/"comparison.csv") if r["method"]=="baseline"}
    tol = cfg["baseline_absolute_tolerance"]
    calibration = measure(n,members,layouts,cfg,[index[k] for k in cal],fixed)
    checks = {"calibration":baseline_check(calibration[GLOBAL],ref["calibration"],tol)}
    def check():
        pp.save(folder/"baseline_checks.json",checks)
        if not all(v["passed"] for group in checks.values() for v in group.values()):
            raise ValueError(f"Alpha=0 reference mismatch, see {folder/'baseline_checks.json'}")
    check()
    score = lambda method:calibration[method].metric.summary()["image_miou"]
    sliding = max((m for m in calibration if not m.post and m.tile!="global"),key=score)
    finalists = [GLOBAL,sliding,replace(GLOBAL,post=True),replace(sliding,post=True)]
    chosen = max(finalists,key=score)
    selected = dict(method=asdict(chosen), raw_sliding_winner=asdict(sliding),config=cfg,
                    params=asdict(fixed), fusion_source=model_source)
    pp.save(folder/"selection.json",selected)
    calibration = {m:v for m,v in calibration.items() if not m.post or m in finalists}
    holdout = measure(n,members,layouts,cfg,[index[k] for k in held],fixed,{GLOBAL,sliding})
    combined = {m:v.merged(n,holdout[m]) for m,v in calibration.items()}
    for part,group in (("holdout",holdout),("all",combined)):
        checks[part] = baseline_check(group[GLOBAL],ref[part],tol)
    check()
    write_results(folder,dict(calibration=calibration,holdout=holdout,all=combined),checks,sliding,chosen)
    pp.save(folder/"best_config.json",selected)


def selected_probability(n,members,layouts,image,method):
    if method.tile=="global":
        return pp.fusion(n,members,image)
    layout = layouts[method.tile]
    if method.model!="fusion":
        return layout.probability(n,members[method.model],image)
    b2 = layout.probability(n,members["b2"],image)
    b4 = layout.probability(n,members["b4"],image)
    local = mix(b2,b4,0.6)
    if method.alpha==1:
        return local
    return mix(pp.fusion(n,members,image),local,method.alpha)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    actions.add_parser("study")
    submit=actions.add_parser("submit")
    submit.add_argument("--selection",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="study":
        cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        history=Path(cfg["postprocess_run"])
        saved=pp.read(history/"best_params.json")
        model_source=saved["config"]
        fixed=pp.Params(**saved["params"])
    else:
        saved=pp.read(args.selection)
        cfg=saved["config"]
        model_source=saved["fusion_source"]
        fixed=pp.Params(**saved["params"])
        method=Method(**saved["method"])
    checkpoints,spec,split=pp.source(model_source)
    if args.action=="submit" and method.model!="fusion":
        checkpoints={method.model:checkpoints[method.model]}
    n=notebook_definitions()
    members=load_members(n,checkpoints,spec,split)
    layouts={k:Layout(n,v,cfg["gaussian_sigma_fraction"]) for k,v in cfg["tiles"].items()
             if args.action=="study" or k==method.tile}
    folder=n.new_run(Path("outputs"),"sliding_m7_"+args.action)
    print(folder.resolve(),flush=True)
    if args.action=="study":
        pp.save(folder/"config.json",dict(config=cfg,checkpoints=checkpoints,m7=spec))
        study(n,cfg,folder,history,model_source,fixed,split,members,layouts)
    else:
        def mask_fn(selected,image,device):
            return classify(selected_probability(n,selected,layouts,image,method),method,fixed)
        output=n.output_for(folder,"submissions")/"submission.csv"
        count=n.generate_submission(ROOT,members,output,"cuda",mask_fn=mask_fn)
        if count!=1000:
            raise ValueError(f"Expected 1000 submission rows, got {count}")
        print(output.resolve(),count,flush=True)


if __name__=="__main__":
    main()
