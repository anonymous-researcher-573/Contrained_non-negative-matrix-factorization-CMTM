# 🟢 Constrained Non-negative Matrix Factorization for Guided Minority Topic Modeling

*"Discover minority topics in text datasets through guided, constraint-based topic modeling."*
---

### ⚠️ Review-Only Notice

This repository is provided **solely for peer review purposes**.

- The code is **not licensed for commercial or public use**.
- Redistribution, reuse, or publication of the code or data is **not permitted**.
- For questions, contact the author at [seyedeh.ebrahimi@tuni.fi](mailto:seyedeh.ebrahimi@tuni.fi).

If you're a reviewer, thank you for evaluating this work!

---

## 🧭 Overview

The **Constrained Non-negative Matrix Factorization** algorithm is a novel topic modeling framework designed to uncover **minority topics** in imbalanced text datasets.  
Traditional topic models tend to overlook low-frequency or underrepresented topics — our method integrates **seed word guidance** and **matrix constraints** to enhance minority topic discovery.

Key Highlights:
- 🎯 **Guided topic modeling** using user-defined seed words.
- 🔒 **Controlled constraints** over topic-word and document-topic matrices.
- 🟢 Effective discovery of **low-frequency, minority topics**.
- 🧪 Includes **evaluation scripts**

---

## 📂 Project Structure

```plaintext
📦 Minority_Topic-Model
│
├── data/                   → Synthetic datasets
├── Evaluation.py           → Evaluation script (NMI & Purity metrics across different methods)
├── script.py               → main script
├── script-run.py           → Parameter configuration script
├── sythtetic-data.csv      → Synthetic dataset
├── requirements.txt        → Python dependencies
└── README.md               → Project documentation


