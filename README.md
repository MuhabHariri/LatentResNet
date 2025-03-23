# EncodDeepResNet — ImageNet Classifier using LiteAE Autoencoder + DeepResNet Blocks

This repository accompanies the paper **"EncodDeepResNet: An Optimized Feature Extraction and Classification Model with Low Computational Cost."**

The project is structured for both **researchers** and **practitioners**, offering a clean, modular, and reproducible codebase to experiment with deep learning classification using an autoencoder backbone and custom DeepResNet blocks.

>  This implementation reflects the **base (large)** version of EncodDeepResNet. To explore other variants, simply adjust the configuration and hyperparameters in `src/config.py`.

---

##  Key Features

- ✅ **LiteAE** — Lightweight Autoencoder for feature extraction  
- ✅ **DeepResNet Blocks** — Efficient residual units with attention  
- ✅ **EncodDeepResNet** — Final classification head  
- ✅ Flexible config system for architecture variants  
- ✅ Custom data augmentation pipeline  
- ✅ Multi-GPU distributed training (via `MirroredStrategy`)

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier
