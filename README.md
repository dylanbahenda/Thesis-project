# Thesis Project – Multimodal Fusion for Medical Imaging

This repository contains the code and experiments from my Master's thesis on multimodal deep learning for medical imaging.  
The project explores **fusion of vision (X-ray images) and language (radiology reports)** using Transformer-based models such as **DeiT**, **TinyBERT**, and **CLIP**, applied to two clinical cases:

- **Chest X-rays** (multi-label classification of 13 abnormalities + no finding)  
- **Appendicular Skeletal X-rays** (binary classification)  

The goal is to compare unimodal and multimodal approaches, and to assess the effectiveness of different fusion strategies.

---

## Repository structure

```
Thesis-project/
│
├── Chest_case/                # Experiments for chest X-rays
│   ├── data_prep.ipynb
│   ├── train_deit.ipynb
│   ├── train_tinybert.ipynb
│   ├── train_fusion.ipynb
│   ├── train_clip.ipynb
│   └── utils.py
│
├── Skeletal_case/             # Experiments for skeletal X-rays
│   ├── data_prep.ipynb
│   ├── train_deit.ipynb
│   ├── train_tinybert.ipynb
│   ├── train_fusion.ipynb
│   ├── train_clip.ipynb
│   ├── fracture_dataset.py
│   └── utils.py
│
├── Thesis/                    # Thesis document
│   ├── Thesis Bahenda.pdf
│
├── Presentation/              # Thesis defense slides
│   └── Thesis Presentation.pptx
│
└── README.md                  # Project description

```
