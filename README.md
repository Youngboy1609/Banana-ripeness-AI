# Banana Ripeness & Defect AI

A multi-task deep learning system for classifying banana ripeness and detecting physical defects.

## 🚀 News & Improvements
This repository has been significantly upgraded to address class imbalance and performance issues:
- **Architecture**: Switched to **EfficientNet-B1** backbone for better multi-task representation.
- **Data Balancing**: Implemented **WeightedRandomSampler** and a corrected stratified splitting logic (80/10/10).
- **Performance**: Achieved **>99% Macro-F1** on the test set, with 100% recall for rare classes like 'semi-ripe' and 'defective'.

## 🛠️ Features
- **Multi-task Learning**: Single model for both ripeness classification (4 stages) and defect detection.
- **Robust Pipeline**: Includes deduplication, near-duplicate removal, and stratified group splitting.
- **Real-time Demo**: High-speed webcam inference with moving average smoothing.
- **Audit Tools**: Scripts to verify data integrity and visualize model mistakes.

## 📁 Project Structure
- `configs/`: YAML configurations for training and inference.
- `src/`: Core library code (modeling, data processing, metrics).
- `scripts/`:
  - `02_prepare_dataset.py`: Main data processing pipeline.
  - `03_train.py`: Model training.
  - `06_demo_webcam.py`: Real-time webcam demo.
  - `research/`: Diagnostic and audit tools.
- `reports/`: Training logs and classification reports.

## 🚦 Quick Start
1. **Setup environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Prepare Data**:
   ```powershell
   python scripts/02_prepare_dataset.py --config configs/data.yaml
   ```
3. **Train Model**:
   ```powershell
   python scripts/03_train.py --config configs/train.yaml
   ```
4. **Run Live Demo**:
   ```powershell
   python scripts/06_demo_webcam.py --config configs/infer.yaml
   ```

## 📊 Evaluation
Final metrics (Macro-F1):
- **Ripeness Classification**: 99.07%
- **Defect Detection**: 99.76%

Detailed reports can be found in `reports/metrics/`.
