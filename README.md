<p align="center">
  <img src="./images/Logo.png" alt="Happy-rPPG Toolkit Logo" width="100%">
</p>

# Happy-rPPG Toolkit

> A comprehensive and extensible toolkit for evaluating remote photoplethysmography (rPPG) under dynamic lighting conditions at night.

## 📌 Introduction

**Happy-rPPG Toolkit** is designed to facilitate the research and evaluation of rPPG signal extraction methods in challenging **night-time dynamic lighting** environments. It integrates:

- A novel dataset: **DCLN**
- Multiple state-of-the-art rPPG algorithms
- A modular, reproducible pipeline for training, validation, and cross-dataset testing


---

## 🧠 Dataset: DCLN

**DCLN (Dynamic-lighting Conditions at Night)** is a dataset collected specifically for rPPG signal evaluation under complex lighting environments.

### ✅ Key Features

- **60 volunteers**, each recorded under 4 lighting setups:
  1. Fixed intensity and fixed position
  2. Varying intensity and fixed position
  3. Fixed intensity and moving position
  4. Varying intensity and moving position
- **480 video samples**, covering both **rest** and **post-exercise** states
- Captured in a **darkroom** with synchronized physiological signal acquisition

> 💾 File Format: `.h5`  
> 🗂 Naming Convention: `P1_1` ~ `P60_8`

### 📷 Sample Frame Snapshots

*(Insert grid of frames here from different lighting conditions)*

---

## 🔬 Included Methods

The following rPPG methods are included or supported:

| Method        | Type          | Description                                   |
|---------------|---------------|-----------------------------------------------|
| CHROM         | Traditional   | Color space-based baseline                    |
| POS           | Traditional   | Popular skin-tone enhancement technique       |
| DeepPhys      | Deep Learning | Spatio-temporal CNN for pulse estimation      |
| PhysNet       | Deep Learning | Lightweight CNN with attention mechanisms     |
| EfficientPhys | Deep Learning | Real-time rPPG model                          |
| [YourModel]   | Deep Learning | [Describe your custom model here]             |

> 🧩 New models can be added by creating a new file in `models/` and registering it in `model_factory.py`.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Happy-rPPG.git
cd Happy-rPPG
pip install -r requirements.txt
```
---

## 🧪 Running Experiments
### 1修改`Run.py`
```

```
### 2运行Run.py
```

```
---
## 📊 Evaluation Results
Performance of different models on DCLN dataset:


---
## 🧩 Extending Happy-rPPG Toolkit
* Add a new model: Create your model file in models/, then register it.

* Add a new dataset: Implement a data loader in data/, update data_factory.py.

* Add new evaluation metric: Add in utils/metrics.py.

## 📬 Contact

For issues, suggestions, or collaborations:
📧 Email: your_email@cqut.edu.cn

---

## 🤝 Contributors

---
##  📚 Citation
If you use this toolkit or the DCLN dataset, please cite:
````
@article{your_paper,
  title={Happy-rPPG: A Toolkit and Dataset for Remote Photoplethysmography under Dynamic Lighting at Night},
  author={Hanguang Xiao and others},
  journal={Measurement},
  year={2025}
}
````