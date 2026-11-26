"""Run from the UAV project root. Writes only to a new error-analysis run."""
import csv
import heapq
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from fusion_study import notebook_definitions

MODELS = ("B7", "M7")
SIZES = ("small", "medium", "large")


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a / b) if b else None


def distribution(values):
    x = [v for v in values if v is not None and np.isfinite(v)]
    if not x:
        return [None] * 3
    return np.percentile(x, [25, 50, 75]).tolist()


def components(gt):
    """Four-connected same-label runs. Assign compact numeric region IDs."""
    starts = np.ones(gt.shape, bool)
    starts[:, 1:] = gt[:, 1:] != gt[:, :-1]
    runs = starts.cumsum().reshape(gt.shape) - 1
    parent = list(range(int(runs.max()) + 1))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    overlap = starts[1:] | starts[:-1]
    overlap &= gt[1:] == gt[:-1]
    for a, b in zip(runs[1:][overlap], runs[:-1][overlap]):
        parent[root(int(a))] = root(int(b))
    _, compact = np.unique([root(i) for i in range(len(parent))], return_inverse=True)
    return compact[runs]


def corrected_matrix(cm, k):
    # Give every pixel involving k its true label, leaving other errors intact.
    oracle = cm.copy()
    errors_to_k = cm[:, k].copy()
    errors_to_k[k] = 0
    oracle[:, k] = 0
    oracle[k, :] = 0
    oracle[k, k] = cm[k].sum()
    oracle[np.diag_indices(len(cm))] += errors_to_k
    return oracle


def predictions(n, model, settings, fusion, image):
    base = total = None
    for view, weight in fusion["methods"]["M7"].items():
        name, w, h = fusion["views"][view]
        assert name == "B7"
        cfg = n.merge(settings, {"image_size": [w, h], "inference": {
            "tile_size": None, "hflip": True, "vflip": False, "scales": [1.]}})
        # predict_logits evaluates the unflipped view first. Keep only its mask.
        def forward(x):
            nonlocal base
            logits = model(x)
            if (w,h) == (512,512) and base is None:
                base = n.resize_logits(logits.float(), (image.height,image.width)).argmax(1)[0].byte().cpu().numpy()
            return logits
        prob = n.predict_logits(forward, image, cfg, "cuda").float().softmax(1)
        term = prob * weight
        total = term if total is None else total + term
    assert base is not None
    return base, total.argmax(1)[0].byte().cpu().numpy()


@torch.inference_mode()
def main():
    n = notebook_definitions()
    fusion = yaml.safe_load(Path("configs/fusion_final.yaml").read_text())
    model, state = n.load_trained(Path(fusion["checkpoints"]["B7"]), "cuda", "official")
    data = Path("UAV_dataset")
    pairs = n.validation_pairs(data, state)
    assert len(pairs) == len(set(state["split"]["validation"])) == 800
    folder = n.new_run(Path("outputs/error_analysis"), "b7_m7")
    n.write_json(folder/"config.json", dict(checkpoint=fusion["checkpoints"]["B7"],epoch=state["epoch"],
                 views=fusion["views"],weights=fusion["methods"]["M7"],split=state["split"]["validation"]))
    settings = state["config"]
    del state
    metrics = [n.SegmentationMetrics() for _ in MODELS]
    spatial = {}
    images, details, groups = [], [], []
    worst = {name: [] for name in MODELS}
    for pair in pairs:
        image, mask = n.read_pair(pair)
        gt = np.asarray(mask)
        assert image.size == mask.size == (800,600) and gt.max()<16
        base, m7 = predictions(n,model,settings,fusion,image)
        bad0,bad1 = base!=gt,m7!=gt
        boundary = np.asarray(Image.fromarray(n.boundary_pixels(gt).astype("uint8")).filter(ImageFilter.MaxFilter(5)))>0
        comp = components(gt)
        area = np.bincount(comp.ravel())
        interior = np.bincount(comp.ravel(),weights=(~boundary).ravel(),minlength=len(area))
        bins = np.digitize(area,[257,4097])
        gray = np.asarray(image.convert("L"),dtype=float)
        lap = 4*gray[1:-1,1:-1]-gray[:-2,1:-1]-gray[2:,1:-1]-gray[1:-1,:-2]-gray[1:-1,2:]
        freq = np.bincount(gt.ravel(),minlength=16)
        row = dict(image=pair.image.name,brightness=gray.mean(),contrast=gray.std(),
                   saturation=np.asarray(image.convert("HSV"))[:,:,1].mean(),sharpness=lap.var(),
                   class_count=np.count_nonzero(freq),dominant_class=int(freq.argmax()),
                   small_area_share=area[bins==0].sum()/gt.size,small_components=int((bins==0).sum()),
                   resolved=int((bad0 & ~bad1).sum()),introduced=int((~bad0 & bad1).sum()),persistent=int((bad0 & bad1).sum()))
        for j,(name,pred) in enumerate(zip(MODELS,(base,m7))):
            cm,score = metrics[j].add(gt,pred,pair.image.name)
            row[name] = score
            hits = np.bincount(comp.ravel(),weights=(pred==gt).ravel(),minlength=len(area))
            for label,sel in [(v,bins==i) for i,v in enumerate(SIZES)]+[("thin",interior==0),("has_interior",interior>0)]:
                if sel.any():
                    groups.append(dict(image=row["image"],model=name,group=label,components=int(sel.sum()),
                        missed=int((hits[sel]==0).sum()),recall_sum=(hits[sel]/area[sel]).sum(),
                        pixels=int(area[sel].sum()),correct=int(hits[sel].sum())))
            for label,sel in (("boundary",boundary),("interior",~boundary)):
                key=(name,label)
                spatial[key]=spatial.get(key,0)+n.confusion_matrix(gt[sel],pred[sel])
            for k,iou in enumerate(n.iou_values(cm)):
                tp=int(cm[k,k]); fp=int(cm[:,k].sum()-tp); fn=int(freq[k]-tp)
                oracle=n.mean_iou(n.iou_values(corrected_matrix(cm,k)))
                details.append(dict(image=row["image"],model=name,label=k,gt=int(freq[k]),tp=tp,fp=fp,fn=fn,
                    iou=float(iou) if np.isfinite(iou) else None,image_miou=score,oracle_gain_pp=100*(oracle-score)))
        images.append(row)
        for name in MODELS:
            entry=(-row[name],row["image"],base,m7)
            heap=worst[name]
            if len(heap)<50: heapq.heappush(heap,entry)
            elif entry[:2]>heap[0][:2]: heapq.heapreplace(heap,entry)
    table(folder/"images.csv",images)
    table(folder/"image_classes.csv",details)
    table(folder/"component_image_groups.csv",groups)
    report(n,folder,metrics,images,details,groups,spatial)
    worst_dir=folder/"worst50";worst_dir.mkdir()
    ranked=sorted(images,key=lambda r:r["M7"])[:50]
    table(folder/"worst50_M7.csv",ranked)
    table(folder/"worst50_B7.csv",sorted(images,key=lambda r:r["B7"])[:50])
    selected={entry[1]:entry[2:] for heap in worst.values() for entry in heap}
    for name,(base,m7) in selected.items():
        rgb=Image.open(data/"train/imgs"/name).convert("RGB")
        gt=np.asarray(Image.open(data/"train/masks"/name))
        changes=(base!=gt).astype(int)*2+(m7!=gt)
        fig,axes=plt.subplots(1,5,figsize=(17,3.5))
        colors=np.array([[0,0,0],[220,50,50],[40,190,80],[220,170,45]],np.uint8)
        for ax,pic,title in zip(axes,[rgb,n.PALETTE[gt],n.PALETTE[base],n.PALETTE[m7],colors[changes]],
                                [name,"GT","B7","M7","green=fixed / red=new / amber=persistent"]):
            ax.imshow(pic);ax.set_title(title,fontsize=9);ax.axis("off")
        save(fig,worst_dir/f"{Path(name).stem}.png")
    print(folder.resolve())


def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def report(n,folder,metrics,images,details,groups,spatial):
    classes=[]; regions=[]; objects=[]
    summary={}
    for name,metric in zip(MODELS,metrics):
        summary[name]=metric.summary()
        cm=metric.matrix
        norm=cm/cm.sum(1,keepdims=True).clip(min=1)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,a,label in zip(axes,[cm,norm],["raw","row_normalized"]):
            np.savetxt(folder/f"{name}_confusion_{label}.csv",a,delimiter=",",fmt="%.8g" if label!="raw" else "%d")
            fig.colorbar(ax.imshow(a),ax=ax);ax.set(title=label,xlabel="Predicted",ylabel="GT")
        save(fig,folder/f"{name}_confusion.png")
        for k in range(16):
            d=[r for r in details if r["model"]==name and r["label"]==k]
            present=[r for r in d if r["gt"]];absent=[r for r in d if not r["gt"]]
            tp=int(cm[k,k]);fn=int(cm[k].sum()-tp);fp=int(cm[:,k].sum()-tp)
            other=sorted((j for j in range(16) if j!=k and cm[k,j]),key=lambda j:-cm[k,j])[:3]
            r=dict(model=name,label=k,tp=tp,fp=fp,fn=fn,iou=ratio(tp,tp+fp+fn),precision=ratio(tp,tp+fp),
                   recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),gt_share=cm[k].sum()/cm.sum(),
                   pred_share=cm[:,k].sum()/cm.sum(),present_images=len(present),presence_rate=len(present)/len(d),
                   affected=sum(bool(v["fp"]+v["fn"]) for v in d),absent_fp_images=sum(v["fp"]>0 for v in absent),
                   absent_fp_pixels=sum(v["fp"] for v in absent),oracle_gain_pp=np.mean([v["oracle_gain_pp"] for v in d]),
                   top_confusions=" / ".join(f"{j} ({cm[k,j]} px)" for j in other))
            for key,values in [('class_iou',[x['iou'] for x in d]),('present_miou',[x['image_miou'] for x in present]),('absent_miou',[x['image_miou'] for x in absent])]:
                for q,val in zip((25,50,75),distribution(values)):r[f"{key}_q{q}"]=val
            classes.append(r)
        for label in ("boundary","interior"):
            a=spatial[name,label]
            regions.append(dict(model=name,region=label,pixels=int(a.sum()),errors=int(a.sum()-a.trace()),
                                error_rate=ratio(a.sum()-a.trace(),a.sum()),global_miou=n.mean_iou(n.iou_values(a))))
        for label in (*SIZES,"thin","has_interior"):
            d=[r for r in groups if r["model"]==name and r["group"]==label]
            count=sum(r["components"] for r in d)
            objects.append(dict(model=name,group=label,count=count,miss_rate=ratio(sum(r["missed"] for r in d),count),
                component_mean_recall=ratio(sum(r["recall_sum"] for r in d),count),
                pixel_recall=ratio(sum(r["correct"] for r in d),sum(r["pixels"] for r in d))))
    table(folder/"classes.csv",classes);table(folder/"regions.csv",regions);table(folder/"components.csv",objects)
    summary["classes"]=classes;summary["regions"]=regions;summary["components"]=objects
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for name in MODELS:
        d=[r for r in classes if r["model"]==name]
        for ax,key in zip(axes.flat[:3],["iou","precision","recall"]):
            ax.plot(range(16),[r[key] for r in d],marker=".",label=name);ax.set(title=key,xlabel="Class")
        axes[1,0].scatter([r['gt_share'] for r in d],[r['iou'] for r in d],label=name)
        axes[1,1].scatter([r['recall'] for r in d],[r['precision'] for r in d],label=name)
        axes[1,2].plot([r['group'] for r in objects if r['model']==name],[r['component_mean_recall'] for r in objects if r['model']==name],marker=".",label=name)
    axes[1,0].set(xlabel="GT pixel share",ylabel="IoU");axes[1,1].set(xlabel="Recall",ylabel="Precision")
    axes[1,2].set_title("Component mean recall")
    for ax in axes.flat:ax.legend()
    save(fig,folder/"class_and_size_comparison.png")
    associations=[]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axes.flat,["brightness","contrast","saturation","sharpness","class_count","small_area_share"]):
        x=np.array([r[key] for r in images])
        for name in MODELS:
            y=np.array([r[name] for r in images])
            ax.scatter(x,y,s=3,alpha=.25,label=name)
            associations.append(dict(property=key,model=name,pearson_r=float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None))
        ax.set(xlabel=key,ylabel="Image mIoU");ax.legend()
    save(fig,folder/"image_properties.png");table(folder/"associations.csv",associations)
    (folder/"results.json").write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False),encoding="utf8")
    priorities=sorted((r for r in classes if r["model"]=="M7"),key=lambda r:-r["oracle_gain_pp"])
    lines=["# B7 與 M7 錯誤分析","",f"原尺寸逐圖 mIoU：B7 {summary['B7']['image_miou']:.5f}，M7 {summary['M7']['image_miou']:.5f}。",
           "逐張影像將某類的 FP、FN 改回真值，再以原指標重新計分；下表依 800 張平均可恢復的百分點排序。這是單類理想修正的診斷上限，彼此不可加總，也無法直接預測 Kaggle 增幅。","",
           "|優先類別|上限（百分點）|受影響影像|預測量|主要錯分去向（像素）|出現影像比例|","|---:|---:|---:|---|---|---:|"]
    for r in priorities:
        bias="低估" if r["fn"]>r["fp"] else "高估" if r["fp"]>r["fn"] else "相等"
        lines.append(f"|{r['label']}|{r['oracle_gain_pp']:.3f}|{r['affected']}|{bias}|{r['top_confusions']}|{r['presence_rate']:.1%}|")
    total=lambda key:sum(x[key] for x in images)
    lines+=["",f"融合修正 {total('resolved'):,} 像素、新增 {total('introduced'):,} 個錯誤，兩組都錯 {total('persistent'):,} 像素。",
            "四連通 GT 區域按原圖面積分成小（≤256）、中（257–4096）、大（>4096）。零正確像素算完全漏掉。邊界帶為 GT 四鄰域交界再以 5×5 方窗擴張；整個連通區都位於帶內列為 thin，包含細長及微小區域。"]
    reg=[r for r in regions if r['model']=='M7']
    for r in reg:lines.append(f"- {r['region']}：錯分率 {r['error_rate']:.2%}，共 {r['errors']:,} 個錯分像素。")
    for r in objects:
        if r['model']=='M7' and r['count']:lines.append(f"- {r['group']}：完全漏掉 {r['miss_rate']:.2%}，各區域等權平均 recall {r['component_mean_recall']:.2%}。")
    photo=[r for r in associations if r["model"]=="M7" and r["property"] in ("brightness","contrast","saturation","sharpness") and r["pearson_r"] is not None]
    if photo:
        strongest=max(photo,key=lambda r:abs(r["pearson_r"]))
        lines.append(f"影像外觀中線性關聯絕對值最大的是 {strongest['property']}，Pearson r={strongest['pearson_r']:.3f}。")
    rare=min(priorities,key=lambda r:r["presence_rate"])
    lines.append(f"驗證影像中最少出現的是類 {rare['label']}（{rare['present_images']}/800 張），其 oracle 上限為 {rare['oracle_gain_pp']:.3f} 個百分點。")
    top=priorities[0]
    lines+=["","## 下一個對照實驗",f"先檢查類 {top['label']} 的標註、最常混淆類別與最差樣本。該類 FP={top['fp']:,}、FN={top['fn']:,}，缺席圖誤報 {top['absent_fp_pixels']:,} 像素。"]
    lines.append("此類漏判較多，可優先單獨測試小幅增加該類 CE 權重，或 Tversky 提高 FN 代價。" if top['fn']>top['fp'] else "此類誤報較多，優先檢查誤報場景並抽取對應負例，或單獨測試 Tversky 提高 FP 代價。")
    lines+=["若優先類別在訓練集也少見，可比較 class-aware sampling；若主要卡在小區域，可比較保留上下文的 class-aware cropping。驗證頻率只提示資料覆蓋，不能證明訓練資料不足。",
            "若交界錯誤率顯著高於內部，檢查標註對齊後再比較 Boundary Loss。Focal Loss 需先檢查反覆錯分是否為有效難例，而非標註雜訊。",
            "亮度、灰階標準差、HSV 飽和度和四鄰域 Laplacian 變異數與分數的關聯僅供探索。應控制類別組成後，再決定是否測試 photometric augmentation 或影像前處理。"]
    (folder/"report.md").write_text("\n\n".join(lines[:2])+"\n"+"\n".join(lines[2:]),encoding="utf8")


if __name__=="__main__":
    main()
