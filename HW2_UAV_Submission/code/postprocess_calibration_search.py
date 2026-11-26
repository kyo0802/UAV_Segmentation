"""Exact batched calibration rescoring using cached FP32 probabilities."""
from pathlib import Path
import json
import numpy as np
import torch
import class_postprocess as cp


class FastSearch(cp.Search):
    def __init__(self, n, samples, folder):
        super().__init__(n, samples, folder)
        self.blocks = []
        items = []
        for p, target, _ in samples():
            gt = torch.as_tensor(target.copy(), device=p.device, dtype=torch.long)
            top, labels = p.topk(2, dim=0)
            valid = gt != 255
            fixed = valid & (top[0] >= .70) & (top[0] * .5 > top[1] * 2)
            active = valid & ~fixed
            hist = lambda s: torch.bincount(s, minlength=16)
            items.append((p[:, active], gt[active], hist(gt[valid]),
                          hist(labels[0][fixed]), hist(gt[fixed & (gt==labels[0])])))
            if len(items)==16:
                self.pack(items)
                items=[]
        if items:
            self.pack(items)

    def pack(self, items):
        p, gt, targets, pred, tp = zip(*items)
        sample = torch.cat([torch.full_like(g,i) for i,g in enumerate(gt)])
        self.blocks.append((torch.cat(p,1),torch.cat(gt),sample,torch.stack(targets).cpu().numpy(),
                            torch.stack(pred),torch.stack(tp)))

    @staticmethod
    def scores(block, base, axis, cls, values):
        p, gt, sample, targets, fixed_pred, fixed_tp = block
        batch = len(targets)
        grid = p.new_tensor(values)
        # Treat the packed pixels as one long row to reuse the original decision code.
        low, high = cp.sweep_masks(p[:,:,None], base, axis, cls, (values[0],values[-1]))
        low, high = low.flatten(), high.flatten()
        flat = sample*16 + low
        bins = batch*16
        pred = fixed_pred + torch.bincount(flat,minlength=bins).reshape(batch,16)
        tp = fixed_tp + torch.bincount(flat[low==gt],minlength=bins).reshape(batch,16)
        changed = low!=high
        s, initial, final, truth = sample[changed], low[changed], high[changed], gt[changed]
        if axis=="weights":
            q = p[:,changed]
            rival = q * p.new_tensor(base.weights)[:,None]
            rival[cls].fill_(-1)
            score, label = rival.max(0)
            left = torch.ones_like(label)
            right = torch.full_like(label,len(values)-1)
            for _ in range((len(values)-1).bit_length()):
                mid = (left+right)//2
                weighted = q[cls] * grid[mid]
                wins = (weighted>score) | ((weighted==score) & (cls<label))
                right = torch.where(wins,mid,right)
                left = torch.where(wins,left,mid+1)
            event = left
        else:
            conf = p[:,changed].amax(0) if cls is None else p[cls,changed]
            event = torch.searchsorted(grid,conf.contiguous(),right=True)
        before = (event*batch+s)*16 + initial
        after = (event*batch+s)*16 + final
        size = bins*len(values)
        def count(keys):
            return torch.bincount(keys,minlength=size).reshape(-1,batch,16)
        pred = pred + (count(after)-count(before)).cumsum(0)
        tp = tp + (count(after[final==truth])-count(before[initial==truth])).cumsum(0)
        tp, pred = tp.cpu().numpy(), pred.cpu().numpy()
        union = targets+pred-tp
        iou = np.divide(tp,union,out=np.full(tp.shape,np.nan),where=union>0)
        return np.nanmean(iou,axis=2)

    def scan(self, stage, base, axes, grids, iteration=0):
        winners={}
        for axis,cls in axes:
            values=grids[axis]
            scores=np.nanmean(np.concatenate([self.scores(b,base,axis,cls,values)
                                              for b in self.blocks],axis=1),axis=1)
            current=base.global_threshold if cls is None else getattr(base,axis)[cls]
            idx=max(range(len(values)),key=lambda i:(scores[i],-abs(values[i]-current)))
            winners[axis,cls]=values[idx]
            for i,v in enumerate(values):
                self.rows.append(dict(stage=stage,iteration=iteration,parameter=axis,label=cls,
                                      value=v,image_miou=scores[i],selected=i==idx,
                                      base=json.dumps(cp.asdict(base))))
        cp.write_rows(self.folder/"calibration_search.csv",self.rows)
        print(stage,iteration,axes[-1],flush=True)
        return winners
