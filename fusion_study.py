"""Run from the project root. Only notebook definitions are imported."""
import argparse
import ast
import csv
import json
import sys
import types
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from PIL import Image

def notebook_definitions():
    module = types.ModuleType("uav_inference")
    sys.modules[module.__name__] = module
    constants = set("PROJECT NUM_CLASSES OUTPUTS VARIANTS HEADS REDUCTIONS HEADER RLE_SYNTAX IMAGE_EXTENSIONS DEBUG_MARKER PALETTE MEAN STD BILINEAR NEAREST IGNORE_INDEX".split())
    for cell in json.loads(Path("UAV_Segmentation.ipynb").read_text(encoding="utf8"))["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            tree.body = [x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)) or isinstance(x,ast.Assign) and getattr(x.targets[0],"id","") in constants]
            exec(compile(tree,cell["id"],"exec"),module.__dict__)
    return module

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=["all","multiscale","ensemble","submit","final","final-submit","b4","b4-submit"])
    parser.add_argument("--b4-checkpoint",type=Path,help="Trained SegFormer-B4 best.pt")
    parser.add_argument("--method",help="Submission method, e.g. F4")
    args=parser.parse_args()
    final=args.action in {"final","final-submit"}
    b4=args.action in {"b4","b4-submit"}
    submit=args.action in {"submit","final-submit","b4-submit"}
    config_path=Path("configs/fusion_final.yaml" if final or b4 else "configs/fusion.yaml")
    cfg=yaml.safe_load(config_path.read_text(encoding="utf8"))
    methods=dict(cfg["methods"])
    methods.update({key:{key:1} for key in cfg["views"]})
    if b4:
        path=args.b4_checkpoint
        if path is None:
            candidates=[Path("outputs/checkpoints")/s.parent.name/"best.pt"
                        for s in Path("outputs/logs").glob("*_b10_mitb4_512_80ep_*/summary.json")]
            if len(candidates)!=1:
                parser.error("Train B10 first, or select its best.pt with --b4-checkpoint")
            path=candidates[0]
        if not path.is_file():
            parser.error(f"Missing trained B4 checkpoint: {path}")
        m7=cfg["methods"]["M7"]
        cfg["views"]={key:cfg["views"][key] for key in m7}
        cfg["views"]["MIT_B4"]=["MIT_B4",512,512]
        cfg["checkpoints"]={"B7":cfg["checkpoints"]["B7"],"MIT_B4":str(path)}
        methods={"M7":m7,"B4":{"MIT_B4":1.0}}
        for label,share in [("mix75",0.75),("mix50",0.50)]:
            methods[label]={**{key:w*share for key,w in m7.items()},"MIT_B4":1-share}
        cfg["methods"]=methods.copy()
        cfg.pop("baselines",None)
    if final and (len(methods)!=40 or len(cfg["views"])!=10):
        parser.error("Final study requires 40 configurations and 10 unique views")
    if submit:
        if args.method not in methods:
            parser.error("Select a configured --method")
        methods={args.method:methods[args.method]}
    elif args.action not in {"all","final","b4"}:
        prefix="F" if args.action=="multiscale" else "E"
        refs="ABCDH" if prefix=="F" else "AUV"
        methods={k:v for k,v in methods.items() if k.startswith(prefix) or k in refs}
    n=notebook_definitions()
    root=Path("UAV_dataset")
    folder=n.new_run(Path("outputs"),"fusion_"+args.action)
    n.write_json(folder/"config.json",dict(config=cfg,methods=methods,hflip=True,scales=[1],padding=False))
    needed=set().union(*methods.values())
    views={}
    models={}
    split=None
    for key,(name,w,h) in cfg["views"].items():
        if key not in needed:
            continue
        if name not in models:
            model,state=n.load_trained(Path(cfg["checkpoints"][name]),"cuda","official")
            if split is not None:
                assert split==state["split"]
            split=state["split"]
            assert state["config"]["model"]["variant"]==("b4" if name=="MIT_B4" else "b2")
            models[name]=model,state["config"]
            del state
        model,settings=models[name]
        settings=n.merge(settings,dict(image_size=[w,h],inference=dict(tile_size=None,hflip=True,vflip=False,scales=[1.])))
        views[key]=(model,settings)
    for weights in methods.values():
        assert min(weights.values())>0 and abs(sum(weights.values())-1)<1e-8
    if submit:
        output=n.output_for(folder,"submissions")
        members=[(*views[k],w) for k,w in methods[args.method].items()]
        count=n.generate_submission(root,members,output/"submission.csv","cuda",prediction_dir=output/"masks")
        assert count==1000
        print(output.resolve(),count,flush=True)
        return
    pairs=n.validation_pairs(root,{"split":split})
    assert len(set(split["validation"]))==len(pairs)==800
    n.write_json(folder/"split.json",split)
    metrics={k:n.SegmentationMetrics() for k in methods}
    times=dict.fromkeys(methods,0.)
    torch.cuda.reset_peak_memory_stats()
    for index,pair in enumerate(pairs):
        image,gt=n.read_pair(pair)
        assert image.size==gt.size==(800,600)
        cache={}
        duration={}
        for key,(model,settings) in views.items():
            start=perf_counter()
            cache[key]=n.predict_logits(model,image,settings,"cuda").float().softmax(1)
            torch.cuda.synchronize()
            duration[key]=perf_counter()-start
        for method,weights in methods.items():
            start=perf_counter()
            score=sum(cache[key]*w for key,w in weights.items())
            pred=score.max(1).indices[0].byte().cpu().numpy()
            times[method]+=perf_counter()-start+sum(duration[k] for k in weights)
            metrics[method].add(np.asarray(gt),pred,pair.image.name)
            del score
        del cache
        if index%200==199:
            print(index+1,flush=True)
    reference="M7" if b4 else "F4" if final else "A"
    baseline=metrics[reference].summary()["image_miou"]
    checks={}
    if final:
        for key,target in cfg["baselines"].items():
            observed=metrics[key].summary()["image_miou"]
            checks[key]=dict(expected=target,observed=observed,passed=abs(observed-target)<=0.0001)
        # Each bootstrap draw uses identical image indices for both paired methods.
        draws=np.random.default_rng(42).integers(len(pairs),size=(5000,len(pairs)),dtype=np.int32)
        ref_scores=np.array([r["miou"] for r in metrics[reference].per_image])
    elif b4:
        assert abs(baseline-0.66452)<1e-4,baseline
    else:
        assert abs(baseline-0.653394063880451)<1e-5,baseline
    rows=[]
    classes=[]
    for method,metric in metrics.items():
        s=metric.summary()
        cm=np.asarray(s["confusion_matrix"])
        hit=cm.diagonal()
        if not final and method in ("E0","F0"):
            assert np.array_equal(cm,metrics["A"].matrix) and s["image_miou"]==baseline
        rows.append(dict(method=method,image_miou=s["image_miou"],global_miou=s["global_miou"],delta_pp=100*(s["image_miou"]-baseline),inference_seconds=times[method]))
        if final:
            delta=np.array([r["miou"] for r in metric.per_image])-ref_scores
            low,high=np.percentile(100*delta[draws].mean(axis=1),[2.5,97.5])
            row=rows[-1]
            row.update(estimated_inference_seconds=row.pop("inference_seconds"),images=len(delta),
                       ci95_low_pp=low,ci95_high_pp=high,class15_iou=s["per_class_iou"][15],
                       class15_precision=hit[15]/max(1,cm[:,15].sum()),class15_recall=hit[15]/max(1,cm[15].sum()))
        for k in range(16):
            classes.append(dict(method=method,label=k,iou=s["per_class_iou"][k],precision=hit[k]/max(1,cm[:,k].sum()),recall=hit[k]/max(1,cm[k].sum()),false_positives=int(cm[:,k].sum()-hit[k])))
    rows.sort(key=lambda r:-r["image_miou"])
    for filename,data in [("comparison",rows),("classes",classes)]:
        with (folder/(filename+".csv")).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    peak=torch.cuda.max_memory_allocated()
    n.write_json(folder/"results.json",dict(comparison=rows,classes=classes,shared_peak_gpu_bytes=peak,baseline_checks=checks))
    if final:
        lines=["# 最終多尺度融合驗證", "", f"驗證影像 {len(pairs)} 張。整輪 PyTorch GPU 配置峰值 {peak/2**30:.2f} GiB。",
               "推論秒數為共用前向時間加上各設定的融合時間估計，不含讀檔與計分。差值及配對 bootstrap 95% CI 均為相對 F4 的百分點（5,000 次，seed 42）。",""]
        for key,check in checks.items():
            status="通過" if check["passed"] else "警告：差異超過 0.0001"
            lines.append(f"- {key} {status}，實測 {check['observed']:.6f}，參考 {check['expected']:.5f}")
            if not check["passed"]:
                print(lines[-1],flush=True)
        lines.extend(["","|排名|設定|Image mIoU|Global mIoU|差值 pp|95% CI pp|類 15 IoU|估計秒數|",
                      "|---:|---|---:|---:|---:|---|---:|---:|"])
        for rank,r in enumerate(rows,1):
            lines.append(f"|{rank}|{r['method']}|{r['image_miou']:.5f}|{r['global_miou']:.5f}|{r['delta_pp']:+.3f}|"
                         f"{r['ci95_low_pp']:+.3f} 至 {r['ci95_high_pp']:+.3f}|{r['class15_iou']:.5f}|{r['estimated_inference_seconds']:.1f}|")
        (folder/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(folder.resolve(),rows[0],flush=True)

if __name__=="__main__":
    main()
