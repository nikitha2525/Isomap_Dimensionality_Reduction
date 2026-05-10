# 🌀 Day 42 — Isomap: Nonlinear Dimensionality Reduction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557C?style=flat-square)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2042-blueviolet?style=flat-square)

**Unfolding nonlinear manifolds through geodesic distance preservation.**

</div>

---

## 📌 Overview

Traditional dimensionality reduction methods like PCA only capture **linear variance** — they fail when data lies on a curved or twisted surface. Isomap solves this by measuring **geodesic distances** (distances along the data manifold), not straight-line Euclidean distances.

This project explores how Isomap transforms high-dimensional, nonlinear data into meaningful lower-dimensional representations — without destroying the underlying structure.

> **Hard truth learned today:** If you blindly apply PCA to every nonlinear dataset, you're destroying structure without even realizing it.

---

## 🧠 What Is Isomap?

**Isomap (Isometric Mapping)** is a nonlinear dimensionality reduction algorithm that extends **Multidimensional Scaling (MDS)** by approximating geodesic distances using a neighborhood graph.

### How it works — 3 steps

```
Step 1: Build a neighborhood graph
         → Connect each point to its k nearest neighbors

Step 2: Compute geodesic distances
         → Use shortest-path algorithms (Dijkstra / Floyd-Warshall)
         → Approximate the true manifold distance between all point pairs

Step 3: Apply MDS on the distance matrix
         → Embed into a lower-dimensional space that preserves those distances
```

### Isomap vs PCA

| Property | PCA | Isomap |
|---|---|---|
| Structure preserved | Linear (flat) | Nonlinear (curved) |
| Distance metric | Euclidean | Geodesic |
| Works on manifolds | ❌ | ✅ |
| Computational cost | Low | Higher |
| Best for | Linear datasets | Curved / complex datasets |

---

## 📂 Datasets Used

### 1️⃣ Digits Dataset (`sklearn.datasets.load_digits`)
- 8×8 pixel images of handwritten digits (0–9)
- 1797 samples × 64 features
- **Goal:** Reduce 64 dimensions → 2D for visualization while keeping digit classes separable

### 2️⃣ Circles Dataset (`sklearn.datasets.make_circles`)
- Synthetic nested circular rings
- Designed to be **impossible** for PCA to separate linearly
- **Goal:** Observe how Isomap unfolds the circular manifold into separable regions

---

## 🔬 What I Implemented

```python
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits, make_circles
import matplotlib.pyplot as plt

# --- Digits Dataset ---
digits = load_digits()
X, y = digits.data, digits.target  # shape: (1797, 64)

isomap = Isomap(n_neighbors=10, n_components=2)
X_transformed = isomap.fit_transform(X)  # shape: (1797, 2)

# --- Circles Dataset ---
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
X_circles_2d = Isomap(n_neighbors=10, n_components=2).fit_transform(X_circles)
```

**Pipeline:**
1. Load datasets from `sklearn`
2. Apply Isomap — reduce dimensions while preserving neighborhood relationships
3. Visualize transformed embeddings as 2D scatter plots
4. Compare nonlinear separation behavior between PCA and Isomap

---

## 💡 Key Insights

- **Isomap preserves global manifold structure** — it respects how data actually curves through space
- **Nonlinear datasets become visually separable** after embedding (the circles dataset is a perfect example)
- **Neighbor selection (`n_neighbors`) directly controls the trade-off** between local detail and global shape — too few neighbors = fragmented graph; too many = shortcuts that violate manifold geometry
- **Dimensionality reduction is not just compression — it's structure preservation**

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| Computationally expensive | Shortest-path computation is O(n² log n) |
| Sensitive to `n_neighbors` | Wrong value → poor or broken embedding |
| Disconnected manifolds | Algorithm fails if the graph is not fully connected |
| Out-of-sample extension | Cannot directly embed new unseen points without refit |

---

## 🗂️ Project Structure

```
day-42-isomap/
├── isomap_digits.py          # Isomap on handwritten digits
├── isomap_circles.py         # Isomap on make_circles dataset
├── isomap_vs_pca.py          # Side-by-side comparison
├── outputs/
│   ├── digits_embedding.png
│   ├── circles_embedding.png
│   └── pca_vs_isomap.png
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-42-isomap
cd day-42-isomap
pip install -r requirements.txt
python isomap_digits.py
```

**Requirements:**
```
scikit-learn
matplotlib
numpy
```

---

## 📐 The Math Behind It

**Geodesic distance approximation:**

Given a dataset $X \in \mathbb{R}^{n \times d}$, Isomap approximates the geodesic distance $d_G(x_i, x_j)$ between any two points as the shortest path through the neighborhood graph:

$$d_G(x_i, x_j) = \min_{\text{path}} \sum_{\text{edges}} \| x_k - x_{k+1} \|_2$$

MDS then finds a low-dimensional embedding $Y \in \mathbb{R}^{n \times p}$ that minimizes:

$$\text{Stress} = \sqrt{\frac{\sum_{i<j}(d_G(x_i, x_j) - \|y_i - y_j\|)^2}{\sum_{i<j} d_G(x_i, x_j)^2}}$$

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 42 of 100 — Dimensionality Reduction via Manifold Learning

| ← Previous | Current | Next → |
|---|---|---|
| [Day 41](#) | **Day 42 — Isomap** | [Day 43](#) (#) |

---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #Isomap #ManifoldLearning #DimensionalityReduction</sub>
</div>
