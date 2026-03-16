# Artificial Intelligence & Machine Learning Research Project - Comparison of Vision Models for Traversability Estimation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Terrain%20Understanding-green)
![Robotics](https://img.shields.io/badge/Robotics-Mobile%20Robotics-orange)
[![Dataset](https://img.shields.io/badge/Dataset-RELLIS--3D-purple)](https://github.com/unmannedlab/RELLIS-3D)
[![University](https://img.shields.io/badge/University-University%20of%20Adelaide-yellow)](https://www.adelaide.edu.au/)
![Project](https://img.shields.io/badge/Type-AIML%20Research%20Project-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/BrownAssassin/AIML-Research-Project)

This repository contains the code and experiments for the research project:

> **"Comparison of Specialized and Prompt-Driven Vision Models for Off-Road Traversability Estimation on RELLIS-3D"**

Conducted as part of the **COMP SCI 7205: Artificial Intelligence & Machine Learning Research Project** course in the Master of Artificial Intelligence and Machine Learning program at the University of Adelaide.

## 🚀 Project Overview

Autonomous robots operating in off-road environments must determine which terrain regions are safe to traverse. This task is known as traversability estimation.

This project investigates vision-based methods for terrain understanding, evaluating approaches that predict whether terrain is traversable using RGB imagery.

The project focuses on:
- Evaluating computer vision techniques for terrain segmentation
- Analyzing qualitative and quantitative traversability predictions
- Studying failure cases and robustness in real-world environments
- Investigating datasets used for off-road robot navigation research

The project contributes both experimental outputs and a comprehensive research report summarizing methodology, evaluation, and findings.

---

## 🗂 Repository Structure

```
.
├── ckpts/                       # Model checkpoints used during experiments
├── outputs/
│   └── prediction/              # Generated qualitative predictions
├── rellis-output/               # Output results from RELLIS-3D dataset experiments
├── results/
│   └── plots/                   # Evaluation plots and visualizations
├── scripts/                     # Utility scripts for running experiments
│
├── Analysis Presentation.pdf    # Project presentation slides
├── Final Report.pdf             # Final research report
├── Literature Review.pdf        # Literature review document
├── README.md                    # Project README
└── Research Proposal.pdf        # Initial research proposal
```

---

## 📦 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/BrownAssassin/AIML-Research-Project.git
cd AIML-Research-Project
```
2. **Create a virtual environment (recommended):**
```
conda create -n traversability python=3.10
conda activate traversability
```
3. **Install dependencies:**
```bash
pip install numpy pandas matplotlib opencv-python torch torchvision tqdm
```
**Additional packages may be required depending on the scripts used.*

---

## 📁 Datasets

Experiments in this project require the use of off-road robotics datasets designed for terrain perception research.

### RELLIS-3D Dataset

A multi-modal dataset for off-road autonomous navigation containing RGB imagery, LiDAR, and semantic annotations.

Source:
[GitHub](https://github.com/unmannedlab/RELLIS-3D)

Dataset outputs generated during experimentation are stored in:
```
rellis-output/
```

---

## ▶️ Running the Code

Scripts for generating predictions and evaluation plots are located in the `scripts/` directory.

Typical workflow:

1. Load dataset with `download_rellis3d_from_readme.sh`
2. Verify dataset integrity with `rellis3d_sanity_check.py`
3. Run GSAM inference pipeline script (`rellis3d_gsam_singleprocess.py`)
4. Generate predictions and plots (`rellis3d_list_ids.py`, `rellis3d_eval_binary.py`, `plot_metrics.py`)

---

## 📄 Research Deliverables

This repository includes all major deliverables produced during the research project:

- **Literature Review**
- **Research Proposal**
- **Analysis Presentation**
- **Final Research Report**

These documents provide a detailed explanation of:
- Prior work in traversability estimation
- Experimental methodology
- Model evaluation
- Discussion of results and future research directions

---

## 📜 License

This repository is intended for academic and educational use.

If datasets or third-party models are used, please refer to their respective licenses.

---

## 🙋‍♂️ Author

**Mrinank Sivakumar**

*Master of Artificial Intelligence and Machine Learning, University of Adelaide*\
*Bachelor of Information Technology (Honours), University of Ontario Institute of Technology*

[mrinank.sivakumar@gmail.com](mailto:mrinank.sivakumar@gmail.com)
