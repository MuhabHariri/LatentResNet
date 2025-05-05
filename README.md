# 🧠 LatentResNet — ImageNet Classifier using LiteAE (Autoencoder) + DeepResNet Blocks

This repository accompanies the paper **"LatentResNet: An Optimized Underwater Fish Classification Model with Low Computational Cost"**

The project is structured for both **researchers** and **practitioners**, offering a clean, modular, and reproducible codebase to experiment with deep learning classification using an autoencoder backbone (LiteAE) and custom DeepResNet blocks.

> 🏗️ This implementation reflects the **base (large)** version of EncodDeepResNet. To explore other variants, simply adjust the configuration and hyperparameters in `src/config.py`.

---

## 📦 Key Features

- ✅ **LiteAE** — Lightweight Autoencoder for Features Compression and Reconstruction  
- ✅ **DeepResNet Blocks** — Efficient residual units with attention  
- ✅ **LatentResNet** — The classification model  
- ✅ Flexible config system for architecture variants  
- ✅ Custom data augmentation pipeline  
- ✅ Multi-GPU distributed training (via `MirroredStrategy`)

---

## 🚀 Getting Started
### 1. Clone the Repository

```bash
git clone https://github.com/MuhabHariri/EncodDeepResNet.git
```
```bash
cd EncodDeepResNet
```


---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```



---

### 3. Prepare Your Dataset
```bash
Dataset/
├── Train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── Val/
    ├── class_1/
    ├── class_2/
    └── ...
```
Update paths in src/config.py: 
```bash
TRAIN_DIR = r"E:\Dataset\Train"
VAL_DIR   = r"E:\Dataset\Val"
```

---


### 4. Train the Model 
```bash
python train.py
```
---
