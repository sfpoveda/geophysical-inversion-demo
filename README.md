# Geophysical Inversion Demo

A simple repository to learn and demonstrate geophysical inversion workflows using Python and SimPEG.

The goal of this project is to build a complete inversion workflow, from creating a synthetic model to recovering it through numerical optimization.

---

### Objective

This repository is intendeed to:

- Learn the fundamentals of geophysical inversion.
- Practice using SimPEG for forward and inverse modeling.
- Build a clean, reproducible scientific Python project.
- Document each step of the inversion process.

---

### Project structure

```text
geophysical-inversion-demo/
│
├── src/                  # Source code
├── notebooks/            # Optional Jupyter notebooks
├── data/                 # Synthetic or input data
├── figures/              # Generated figures
├── README.md
├── pyproject.toml
├── uv.lock
└── .python-version
```

---

## Installation

Clone the repository


```bash
git clone https://github.com/sfpoveda/geophysical-inversion-demo.git
cd geophysical-inversion-demo
```

Create the environment

```bash
uv sync
```

Run the project

```bash
uv run python src/main.py
```

---

### Expected results

At the end of this project the repository should include:


- Synthetic model generation
- Forward simulation
- Noise addition
Inversion using SimPEG
- Recovered model visualization
- Comparison between true and recovered models

Example outputs will include figures such as:

- Mesh
- True model
- Synthetic data
- Predicted data
- Recovered model
- Data misfit evolution

---

### Dependencies

- Python 3.12
- Numpy
- Scipy
- Matplotlib
- Pandas
- SimPEG

---

## Status

Work in progress
