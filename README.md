# StartNerve Bio-Engine (v6.1) 🧬

### The Operating System for In-Silico Toxicology
**Live Deployment:** [https://startnerve.in](https://startnerve.in)

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red) ![Status](https://img.shields.io/badge/Status-Production-green)

## 📖 Overview
StartNerve is a high-throughput screening (HTS) platform designed to democratize **Computational Toxicology**. It leverages a Random Forest classifier trained on the **NIH/EPA Tox21 dataset** (8k+ compounds) to predict 12 key toxicity endpoints instantly.

## 🚀 Key Features
* **Real-Time Screening:** Sub-second inference for Liver Injury, Cancer Risk (p53), and Stress Response.
* **3D Visualization:** Interactive molecular rendering using `py3Dmol`.
* **Batch Pipeline:** Process 100+ candidates simultaneously via CSV upload.
* **Compliance Ready:** Auto-generates PDF "Certificates of Analysis" for lab documentation.

## 🛠️ Tech Stack
* **Core:** Python 3.13
* **ML Engine:** Scikit-Learn (Random Forest), RDKit (Morgan Fingerprints)
* **Frontend:** Streamlit, Stmol
* **Deployment:** Render Cloud

## 🧪 Scientific Methodology
The model utilizes **2048-bit Morgan Fingerprints** to map chemical structures to biological activity.
* **Training Data:** Tox21 Challenge Dataset (Train/Test Split: 80/20)
* **Validation:** Achieved ~94% accuracy on internal validation sets for binary toxicity classification.

---
*Built by Bhavesh Satbhai. © 2026 StartNerve Technologies.*