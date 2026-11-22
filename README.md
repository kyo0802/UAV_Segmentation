# UAV 語意分割 Notebook

開啟 [UAV_Segmentation.ipynb](UAV_Segmentation.ipynb)，選專案的 `.venv` Python kernel，從第 1 節往下執行。預設只檢查環境與正式資料。訓練、額外驗證和 CSV 都由你手動開啟。

## 操作流程

| 章節 | 要做什麼 |
| --- | --- |
| 1 | 確認 CUDA 實測結果。在「超參數」格修改 `MODEL_NAME` 和 `OVERRIDES`。模型預設值也可在 [unet.yaml](configs/unet.yaml) 或 [segformer.yaml](configs/segformer.yaml) 修改。同名設定以 Notebook 的 `OVERRIDES` 為準。 |
| 2–4 | 讀正式資料、查看影像與遮罩，確認三組 DataLoader 的 batch。訓練和驗證有遮罩。官方測試回傳檔名與原始尺寸。 |
| 5–8 | 閱讀模型、loss、mIoU 和訓練函式。選 `unet` 使用 ResNet34 U-Net，選 `segformer` 後可用 `model.variant` 選 `b0`、`b1` 或 `b2`。 |
| 9 | 準備好才把 `START_TRAINING` 設為 `True` 並執行該格。續訓時填 `RESUME_CHECKPOINT`，沿用原 run 的完整設定。 |
| 10–12 | 查看結果。需要時指定 `CHECKPOINT` 並手動開啟原尺寸驗證或 CSV。 |

訓練預設為 512 × 512、batch size 2、梯度累積 2、CUDA AMP。學習率、warmup、epoch 數及 early stopping patience 在 YAML 的 `training` 區塊。想自行跑短測，可在 Notebook 把 `epochs` 改成 2，並把 `warmup_epochs` 改成 0。

## 啟動 Jupyter

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

新環境可先依 [requirements.txt](requirements.txt) 安裝套件，正式訓練需要可用的 CUDA 版 PyTorch 與 torchvision。請從專案根目錄啟動。

[先前的本機驗證報告](outputs/logs/local_validation/report.md)
