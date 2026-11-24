"""Targeted analysis of six directed M7 confusions. Requires scipy."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from segmentation_errors import notebook_definitions, predictions, table, save

PAIRS = [(1,3),(3,1),(1,0),(0,1),(1,5),(5,1)]
COLORS = np.array([[255,40,40],[255,155,20],[0,210,255],[70,100,255],[245,50,210],[80,245,75]],np.uint8)
WIDTHS = (1,2,4,8)
EDGES = [0,1,2,4,8,16,32,64,np.inf]


def pair_edge(gt,a,b):
    edge=np.zeros(gt.shape,bool)
    for left,right in [(np.s_[:,1:],np.s_[:,:-1]),(np.s_[1:,:],np.s_[:-1,:])]:
        adjacent=((gt[left]==a)&(gt[right]==b))|((gt[left]==b)&(gt[right]==a))
        edge[left]|=adjacent
        edge[right]|=adjacent
    return edge


def visualize(folder,name,rgb,gt,pred,palette,band):
    raw=np.asarray(rgb)
    overlay=raw.copy()
    spatial=raw.copy()
    any_error=np.zeros(gt.shape,bool)
    for (a,b),color in zip(PAIRS,COLORS):
        err=(gt==a)&(pred==b)
        overlay[err]=color
        any_error|=err
    spatial[any_error&band]=[255,160,0]
    spatial[any_error&~band]=[255,0,180]
    fig,axes=plt.subplots(2,5,figsize=(19,7))
    pics=[raw,palette[gt],palette[pred],overlay,spatial]
    for ax,pic,title in zip(axes[0],pics,["RGB","GT","M7","1>3 red, 3>1 orange, 1>0 cyan, 0>1 blue, 1>5 pink, 5>1 green","orange boundary / pink interior"]):
        ax.imshow(pic);ax.set_title(title,fontsize=8)
    # Largest connected error region; identical crop in each panel.
    labels,count=ndi.label(any_error)
    if count:
        sizes=np.bincount(labels.ravel()); k=1+sizes[1:].argmax()
        ys,xs=np.where(labels==k); y,x=ys[len(ys)//2],xs[len(xs)//2]
    else:y,x=300,400
    y,x=int(np.clip(y-48,0,504)),int(np.clip(x-48,0,704))
    for ax,pic in zip(axes[1],pics):
        ax.imshow(pic[y:y+96,x:x+96],interpolation="nearest")
    # Overlay exact GT contours on the RGB enlargement for alignment review.
    native_edge=np.zeros(gt.shape,bool)
    native_edge[:,1:] |= gt[:,1:] != gt[:,:-1]
    native_edge[1:] |= gt[1:] != gt[:-1]
    rgb_crop=raw[y:y+96,x:x+96].copy()
    rgb_crop[native_edge[y:y+96,x:x+96]]=[255,255,0]
    axes[1,0].imshow(rgb_crop,interpolation="nearest");axes[1,0].set_title("RGB + GT edges")
    for ax in axes.flat: ax.axis("off")
    fig.suptitle(f"{name} | native crop x={x}, y={y}, 96x96")
    save(fig,folder/(Path(name).stem+".png"))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=Path("outputs/error_analysis/logs/20261124T123522Z_b7_m7_1dabcd"))
    parser.add_argument("--mask-dir",type=Path,help="Verified native M7 validation label PNGs named after each input")
    args=parser.parse_args()
    n=notebook_definitions()
    config=json.loads((args.source/"config.json").read_text())
    fusion=yaml.safe_load(Path("configs/fusion_final.yaml").read_text())
    assert config["weights"]==fusion["methods"]["M7"] and config["views"]==fusion["views"]
    assert Path(config["checkpoint"]).resolve()==Path(fusion["checkpoints"]["B7"]).resolve()
    names=config["split"]
    assert len(names)==len(set(names))==800
    refs={}
    with (args.source/"image_classes.csv").open() as f:
        for row in csv.DictReader(f):
            if row["model"]=="M7" and row["label"]=="1":refs[row["image"]]=row
    assert set(refs)==set(names)
    top=sorted(names,key=lambda name:-float(refs[name]["oracle_gain_pp"]))[:20]
    random20=random.Random(42).sample(sorted(set(names)-set(top)),20)
    out=n.new_run(Path("outputs/error_analysis"),"spatial_M7")
    figs=out/"inspection";figs.mkdir()
    maskdir=args.mask_dir or args.source/"masks"
    model=None
    counts=Counter(); distances=Counter(); sizes=Counter();review=[]; quality=Counter()
    for name in names:
        with Image.open(Path("UAV_dataset/train/imgs")/name) as im:rgb=im.convert("RGB")
        with Image.open(Path("UAV_dataset/train/masks")/name) as im:
            gt=np.asarray(im)
        assert rgb.size==(800,600) and gt.shape==(600,800) and gt.dtype.kind in "ui" and gt.max()<16
        path=maskdir/(Path(name).stem+".png")
        if path.is_file():
            with Image.open(path) as im:pred=np.asarray(im)
            quality["cached_masks"]+=1
        else:
            if model is None:
                model,state=n.load_trained(Path(config["checkpoint"]),"cuda","official")
                assert state["split"]["validation"]==names
                settings=state["config"];del state
            _,pred=predictions(n,model,settings,fusion,rgb)
            quality["regenerated_masks"]+=1
        assert pred.shape==gt.shape and pred.dtype.kind in "ui" and pred.max()<16
        cm=n.confusion_matrix(gt,pred)
        if abs(n.mean_iou(n.iou_values(cm))-float(refs[name]["image_miou"]))>1e-8:
            raise ValueError(f"M7 prediction differs from saved image metric: {name}")
        # Check that nearest-neighbor resizing introduces no new label IDs.
        resized=np.asarray(Image.fromarray(gt).resize((512,512),n.NEAREST))
        assert np.isin(resized,np.unique(gt)).all()
        edge=n.boundary_pixels(gt)
        bands={w:ndi.maximum_filter(edge,size=2*w+1) for w in WIDTHS}
        if name in top or name in random20:
            visualize(figs,name,rgb,gt,pred,n.PALETTE,bands[2])
            review.append(dict(image=name,group="top20" if name in top else "random20",oracle_gain_pp=refs[name]["oracle_gain_pp"],alignment="",displacement="",semantic_confusion="",missed_small="",fragmentation="",notes=""))
        for a,b in PAIRS[::2]:
            e=pair_edge(gt,a,b)
            dist=ndi.distance_transform_edt(~e) if e.any() else None
            for src,dst in [(a,b),(b,a)]:
                true=gt==src;err=true&(pred==dst);total=int(err.sum())
                if dist is None:distances[src,dst,"no_pair_boundary"]+=total
                else:
                    for i,num in enumerate(np.histogram(dist[err],EDGES)[0]):
                        distances[src,dst,f"[{EDGES[i]},{EDGES[i+1]})"]+=int(num)
                labs,num=ndi.label(err)
                for size,num_regions in zip(("1-256","257-4096","4097+"),np.histogram(np.bincount(labs.ravel())[1:],[1,257,4097,np.inf])[0]):
                    sizes[src,dst,size]+=int(num_regions)
                for w,band in bands.items():
                    k=(src,dst,w)
                    for tag,num in [('errors',total),('boundary_errors',(err&band).sum()),('gt',true.sum()),('gt_boundary',(true&band).sum()),('images',total>0)]:
                        counts[k,tag]+=int(num)
    rows=[]
    for a,b in PAIRS:
        for w in WIDTHS:
            total,be,g,gb,images=[counts[(a,b,w),t] for t in ('errors','boundary_errors','gt','gt_boundary','images')]
            rows.append(dict(gt=a,pred=b,width=w,errors=total,boundary_errors=be,interior_errors=total-be,affected_images=images,gt_boundary=gb,gt_interior=g-gb,boundary_share=be/total if total else None,boundary_rate=be/gb if gb else None,interior_rate=(total-be)/(g-gb) if g>gb else None))
    table(out/"directions.csv",rows)
    table(out/"distances.csv",[dict(gt=a,pred=b,distance=k,pixels=v) for (a,b,k),v in distances.items()])
    table(out/"sizes.csv",[dict(gt=a,pred=b,area=k,components=v) for (a,b,k),v in sizes.items()])
    table(out/"review.csv",review)
    (out/"results.json").write_text(json.dumps(dict(directions=rows,distances=[[*k,v] for k,v in distances.items()],sizes=[[*k,v] for k,v in sizes.items()],quality=quality,config=config),indent=2))
    lines=["# M7 定向混淆分析","固定 800 張。邊界帶為所有 GT 四鄰域交界再以方窗向外擴張 w 像素，w=2 重現原定義。距離另以指定類別對直接相鄰的兩側 GT 像素計算歐氏距離；未相鄰的圖獨立統計。錯分區域使用四連通。","|GT → 預測|錯分像素|邊界占比 w=2|邊界錯分率|內部錯分率|","|---|---:|---:|---:|---:|"]
    for r in sorted((r for r in rows if r["width"]==2),key=lambda r:-r["errors"]):
        lines.append(f"|{r['gt']} → {r['pred']}|{r['errors']:,}|{r['boundary_share']:.2%}|{r['boundary_rate']:.2%}|{r['interior_rate']:.2%}|")
    primary=max((r for r in rows if r["width"]==2),key=lambda r:r["errors"])
    near=sum(distances[primary["gt"],primary["pred"],k] for k in ("[0,1)","[1,2)","[2,4)"))
    choice="優先檢視交界裁切" if near>primary["errors"]/2 else "優先檢視完整場景難例取樣"
    lines.append(f"\n最大錯分方向為 {primary['gt']} → {primary['pred']}，其中 {near:,} 像素距離該類別對交界小於 4 px。依此計數的初步建議為{choice}；最終決策需以人工疊圖檢查確認。")
    lines+=["","## 判讀與下一個實驗","先比較 1、2、4、8 像素結果，再對照指定類別對的距離。若錯分集中在近距離，且 inspection 顯示兩側輪廓錯位，優先測交界裁切；若裁切後仍有一致偏移且標註可靠，再測 boundary-weighted CE。若遠離交界也有大片連續錯分，優先對完整場景做 hard-example sampling。只有方向明顯不對稱且類別整體 precision/recall 支持時，再試類別權重。","請用 review.csv 記錄 RGB/GT 疊對、輪廓偏移、整區錯認、小目標完全漏判及疑似碎片。若發現系統性對齊或標籤編碼問題，先修正前處理。尺寸與最近鄰值域檢查不能證明標註正確。小連通區本身也不能證明噪聲。"]
    (out/"report.md").write_text("\n\n".join(lines[:2])+"\n\n"+"\n".join(lines[2:]),encoding="utf8")
    print(out.resolve())

if __name__=="__main__": main()
