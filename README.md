# DDoS Attack Mitigation Using Deep Reinforcement Learning in SDN

A research-grade implementation of a Deep Reinforcement Learning system for detecting and mitigating DDoS attacks in Software-Defined Networking (SDN) environments.

## 🎯 Overview

This project implements a **PPO (Proximal Policy Optimization)** agent that acts as an intelligent SDN controller, making real-time per-flow decisions (ALLOW, RATE_LIMIT, DROP) based on compressed latent representations of network traffic features extracted by an **autoencoder**.

### Key Features

- 🧠 **3 DRL Agents**: DQN, Double DQN, PPO (primary)
- 🔬 **Autoencoder Feature Extraction**: 20 features → 10-dim latent space
- 📊 **3 ML Baselines**: Random Forest, SVM, XGBoost
- 🌐 **Custom Gym Environment**: SDN controller simulation with asymmetric rewards
- 📝 **IEEE Research Paper**: LaTeX template with all sections
- 📈 **Publication-Quality Visualizations**: Reward curves, confusion matrices, ROC, comparison charts

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test with synthetic data (no dataset needed)
python main.py --test-run

# Full training pipeline (requires CICDDoS2019 dataset)
python main.py --mode full

# Run tests
pytest tests/ -v
```

## 📁 Project Structure

```
ddos-rl-mitigation/
├── main.py                 # Entry point
├── src/
│   ├── config.py           # All hyperparameters
│   ├── data/               # Data loading, preprocessing, autoencoder
│   ├── env/                # Custom SDN Gym environment
│   ├── agents/             # DQN, DDQN, PPO implementations
│   ├── baselines/          # RF, SVM, XGBoost classifiers
│   ├── training/           # Training loop with TensorBoard
│   └── evaluation/         # Metrics, plots, LaTeX tables
├── paper/                  # LaTeX research paper
├── tests/                  # Unit tests
├── data/                   # Dataset (raw + processed)
├── models/                 # Saved checkpoints
└── logs/                   # TensorBoard logs
```

## 📊 Dataset

Download the **CICDDoS2019** dataset from [UNB](https://www.unb.ca/cic/datasets/ddos-2019.html) and place CSV files in `data/raw/`.

## 📝 Research Paper

The paper is in `paper/main.tex` (IEEE double-column format). Compile with:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## 📈 TensorBoard

```bash
tensorboard --logdir=logs/
```
