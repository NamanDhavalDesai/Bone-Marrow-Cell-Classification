# Bone Marrow Cell Classification & Reliability Analysis

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Medical AI](https://img.shields.io/badge/Domain-Hematology-red)
![Accuracy](https://img.shields.io/badge/Accuracy-84.92%25-green)

## 📌 Project Overview
This project involves the development of a Convolutional Neural Network (CNN) to automate the classification of bone marrow cells from microscopic images. Beyond simple classification, the project focuses on **model optimization** and **reliability** through:
1.  **Ablation Studies:** Systematically isolating the impact of specific architectural improvements.
2.  **Out-of-Distribution (OOD) Detection:** Implementing a mechanism to identify unfamiliar cell types using Softmax Entropy, ensuring the model remains reliable in clinical settings.

## 📊 Dataset Detail
The model is trained on a curated subset of the **Kaggle Bone Marrow Cell Classification** dataset.
*   **Input Size:** 56x56x3 (normalized via z-score).
*   **Total Images:** 56,000 (Balanced).
*   **Classes (8):** 
    *   Artefact (ART), Blast (BLA), Erythroblast (EBO), Lymphocyte (LYT), Band Neutrophil (NGB), Segmented Neutrophil (NGS), Plasma Cell (PLM), and Promyelocyte (PMO).

## 🛠️ Model Evolution & Ablation Study
We started with a **Base CNN** (4 convolutional layers + 1 Dense layer) and introduced three strategic enhancements to improve generalization.

### The Improvements:
*   **Data Augmentation:** Random horizontal flips and rotations ($\pm 10^\circ$) to simulate spatial variations.
*   **Batch Normalization:** Applied before ReLU activations to stabilize learning and accelerate convergence.
*   **Dropout (0.5):** Introduced in the fully connected head to reduce overfitting by preventing co-adaptation of neurons.

### Ablation Performance Comparison:
| Model Configuration | Test Accuracy | Impact |
| :--- | :---: | :--- |
| **Base Model (Baseline)** | 74.70% | Initial performance. |
| **Full Improved Model** | **84.92%** | **+10.22% improvement.** |
| **Ablation 1 (No Augmentation)** | 80.26% | Shows importance of data diversity. |
| **Ablation 2 (No BatchNorm)** | 82.48% | Shows benefit of training stability. |
| **Ablation 3 (No Dropout)** | 80.75% | Shows impact of regularization. |

## 🔍 Out-of-Distribution (OOD) Detection
To ensure the model can detect "unfamiliar" cells not present in the 8 training classes, we implemented an uncertainty estimation method using **Softmax Entropy**:

$$S = -\sum_{i=1}^{K} p_i \log(p_i + \epsilon)$$

*   **Low Entropy:** High confidence (In-Distribution).
*   **High Entropy:** Low confidence (Potential OOD/New Cell Type).
*   **Results:** The distribution showed a clear bimodal separation, validating that entropy is a robust proxy for model uncertainty in medical image streams.

## 📈 Key Insights
*   **Augmentation & Dropout** were the most critical factors in performance, proving that for small-resolution medical images ($56\times56$), regularization is more vital than raw depth.
*   **Misclassification Analysis:** The model occasionally struggled with distinguishing between *Band* and *Segmented* Neutrophils due to high morphological similarity, a common challenge even for human hematologists.

## 📂 File Structure
*   `Bone_Marrow_Cell_Classification.ipynb`: Complete pipeline including data preprocessing, training, ablation study, and OOD analysis.
*   `Report.pdf`: The full technical study.

## 🚀 How to Run
1.  Download the subset of the Bone Marrow Cell dataset from Kaggle.
2.  Open `Bone_Marrow_Cell_Classification.ipynb` in Google Colab or Jupyter.
3.  Run all cells to reproduce the ablation study and the OOD entropy histogram.
