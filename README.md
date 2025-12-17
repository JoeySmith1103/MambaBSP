# Mamba-based Battery State Prediction
## Environment Settings 
建議使用python3.9以上, 使用conda 環境
```python=
conda create -n mambaL python=3.11
conda activate mambaL
```
安裝相關套件 (需要確認torch的版本與CUDA version 相符)
```shell=
cd MambaBSP
pip3 install -r requirements.txt
pip3 install torch torchvision # 在CUDA版本為12.8的情況下
```
備註：若CUDA version並非12.8, 請至https://pytorch.org/get-started/previous-versions/ 確認對應的安裝版本
e.g. CUDA 12.6
```shell=
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```
## 使用Mamba模型訓練
```bash
python3 soc_train_mamba.py \
    --data-dir datasets/training_data/soc/Oxford_full_train_test/train \
    --outdir soc_model_mamba \
    --use-cuda \
```
## 使用Mamba模型進行SoC預測
```bash
python3 soc_pred_mamba.py \
    --model-dir soc_model_mamba \
    --test-dir datasets/testing_data/soc/Oxford_full_cycle/test/Cell1/FullCycle \
    --outdir soc_pred_results_mamba \
    --use-cuda \
    --true-label \
    --plot
```
SoH以及RUL預測任務的參數與SoC任務相同

## SoC任務data input
**必需欄：**
- `Current`: 電流值 (A)
- `Voltage`: 電壓值 (V)
- `SOC`: 充電狀態（0-1 或 0-100，會自動標準化）
- `Temperature`: 溫度值 (°C)

**其他欄：**
- 所有非數值欄會被自動丟棄
- 所有數值欄會進行異常值處理（inf/nan）

### SOC 標準化

模型會自動檢測並標準化 SOC 值：
- `[0, 1]` 範圍：直接使用
- `[0, 100]` 範圍：除以 100 轉換為 `[0, 1]`

## 常見問題

### 1. 記憶體不足

**解決方案：**
- 減少 `--num-augment`（減少資料增強）
- 增加 `--accum-steps`（增加梯度累積步數）
- 減少 `--hidden-dim` 或 `--layer-num`
- 使用更小的 `--min-seq-len`

### 2. 訓練損失不下降

**可能原因：**
- 學習率過大或過小：調整 `--lr`
- 資料品質問題：檢查輸入 CSV 檔案
- 模型容量不足：增加 `--hidden-dim` 或 `--layer-num`

### 3. 預測結果異常

**檢查項：**
- 確保使用與訓練時相同的特徵集（Current, Voltage, Temperature）
- 確保使用訓練時保存的 scaler（`best_model_scaler.pkl`）
- 檢查測試資料的 SOC 範圍是否合理

### 4. 模型加載失敗

**錯誤：** `Missing key(s) in state_dict` 或 `Unexpected key(s)`

**解決方案：**
- 確保 `soc_pred_mamba.py` 中的 `Net` 類與 `soc_train_mamba.py` 中的完全一致
- 重新訓練模型
