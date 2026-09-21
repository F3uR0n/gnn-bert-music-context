# GNN-BERT Music Context Understanding

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?style=flat-square&logo=huggingface&logoColor=black)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-GNN-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
![Course](https://img.shields.io/badge/Course-CSE425-informational?style=flat-square)

A four-task neural network project for music context understanding, combining Graph Neural Networks (GraphSAGE) with transformer-based text encoders (DistilBERT) via cross-attention fusion and contrastive alignment.

---

## Overview

This repository implements a hybrid deep learning framework that understands musical context by jointly processing two modalities:

- **Audio structure** — represented as segment-level graphs built from MFCC and chroma features extracted from raw MP3 files.
- **Textual metadata** — captions and tag descriptions encoded by a fine-tuned DistilBERT model.

The project is structured as four progressive tasks, culminating in a cross-modal contrastive dual-encoder that aligns audio and text representations in a shared embedding space.

---

## Architecture

### Task 1 — BERT Baseline for Music Tag Prediction
Fine-tunes `distilbert-base-uncased` on the **MusicCaps** dataset for multi-label tag classification. Each track caption is encoded via the `[CLS]` token, followed by a sigmoid linear head predicting the top-50 music aspect tags. Evaluated with Macro-F1 and Micro-F1.

### Task 2 — GraphSAGE on Audio Structure Graphs
Builds per-track segment graphs from FMA-small audio. Each 5-second segment becomes a graph node (32-dim feature: 12 chroma + 20 MFCC). Sequential and cosine-similarity edges (threshold τ = 0.85) connect segments. A 3-layer GraphSAGE with global mean pooling classifies 8 top-level genres.

### Task 3 — GNN-BERT Cross-Attention Fusion
Fuses the frozen Task 2 GNN encoder with the warm-started Task 1 BERT encoder via a cross-attention mechanism. The graph embedding acts as the query attending over BERT token embeddings. Three ablation variants are compared:

| Model | Accuracy | Macro-F1 | AUC-PR |
|---|---|---|---|
| BERT-only | 0.210 | 0.213 | 0.271 |
| GNN-only (Task 2) | 0.373 | 0.358 | 0.386 |
| Early-Concat | 0.373 | 0.370 | 0.378 |
| **Cross-Attention Fusion** | **0.447** | **0.433** | **0.472** |

### Task 4 — Contrastive Cross-Modal Alignment
Trains a dual-encoder (`ContrastiveDualEncoder`) with InfoNCE loss on paired (audio graph, caption) samples from MusicCaps. Audio graphs are encoded by a `GraphSAGEEncoder`, captions by DistilBERT, and both projected to a shared 128-dim space. Evaluated with Recall@1, Recall@5, and Recall@10 on a retrieval task.

---

## Technologies

| Component | Library / Model |
|---|---|
| Graph Neural Networks | `torch_geometric` — `SAGEConv`, `global_mean_pool` |
| Language Model | `transformers` — `distilbert-base-uncased` |
| Audio Feature Extraction | `librosa` — MFCC, Chroma STFT |
| Data | `datasets` — `google/MusicCaps`; FMA-small |
| Training Framework | `torch`, `torch.nn`, `AdamW` |
| Evaluation | `scikit-learn` — Macro-F1, AUC-PR, Accuracy |
| Visualization | `matplotlib` — t-SNE plots, F1 curves |

---

## Project Structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml                         # Hyperparameters and dataset config
├── data/
│   ├── raw/                            # Downloaded dynamically via wget / HuggingFace datasets
│   ├── processed/
│   │   └── Graphs_examples/            # 25 pre-generated sample audio segment graphs
│   └── splits/                         # task2_split_track_ids.json
├── notebooks/
│   ├── Task_1.ipynb                    # BERT multi-label tag classifier on MusicCaps
│   ├── Task_2.ipynb                    # GraphSAGE genre classifier on FMA-small
│   ├── Task-3.ipynb                    # GNN-BERT cross-attention fusion + ablations
│   ├── Task_4.ipynb                    # Contrastive dual-encoder (InfoNCE) on MusicCaps
│   └── demo_context.ipynb              # End-to-end inference demo
├── src/
│   ├── task_1.py                       # Exported source: BERT tag classifier
│   ├── task_2.py                       # Exported source: graph builder + GraphSAGE
│   ├── fusion_model.py                 # Exported source: Task 3 fusion + ablations
│   ├── contrastive.py                  # Exported source: Task 4 dual encoder
│   ├── train.py                        # Training execution wrapper
│   └── evaluate.py                     # Evaluation execution wrapper
└── result/
    ├── metrics.json                    # Quantitative test-set results
    ├── task3_ablation_comparison.csv   # Ablation table (all 4 models)
    ├── plots/
    │   └── task3_tsne.png              # t-SNE of fused embeddings, colored by genre
    └── models/
        ├── task2_gnn_graphsage.pt      # Saved GraphSAGE weights
        ├── task2_gnn_config.json       # GNN architecture config
        ├── task3_fusion_model.pt       # Saved fusion model + BERT encoder
        └── task4_contrastive_dual_encoder.pt
```

---

## Configuration

Key hyperparameters are defined in [`config.yaml`](config.yaml):

```yaml
dataset:
  audio_primary: fma_small
  audio_tracks: 2000
  text_primary: google/MusicCaps
  segment_sec: 5
  sr: 22050

model:
  gnn_hidden_dim: 64
  gnn_layers: 3
  bert_model: distilbert-base-uncased
  fusion_dim: 128

training:
  batch_size: 32
  epochs_gnn: 30
  epochs_fusion: 8
  epochs_contrastive: 10
  lr: 0.001
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `torch`, `torch_geometric`, `transformers`, `librosa`, `pandas`, `scikit-learn`, `matplotlib`, `networkx`, `datasets`.

**2. Data**

Raw FMA audio and MusicCaps metadata are downloaded automatically inside the notebooks via `wget` and the HuggingFace `datasets` library. The `data/raw/` directory is intentionally empty by design. Pre-built validation graphs are located in `data/processed/`.

---

## Running the Notebooks

All training, evaluation, and visualization were developed interactively on Google Colab. Execute notebooks in the following order:

```
notebooks/Task_1.ipynb   →  Task_2.ipynb   →  Task-3.ipynb   →  Task_4.ipynb
```

> Task 3 requires the saved GNN checkpoint (`task2_gnn_graphsage.pt`) and the train/val/test split (`task2_split_track_ids.json`) produced by Task 2. These must be present before running Task 3.

For a live end-to-end inference demonstration:

```
notebooks/demo_context.ipynb
```

**Viewing pre-computed results without re-running:**

```bash
python src/evaluate.py
```

This reads and prints `result/metrics.json` directly.

---

## Results

Quantitative results are stored in [`result/metrics.json`](result/metrics.json) and [`result/task3_ablation_comparison.csv`](result/task3_ablation_comparison.csv).

The t-SNE visualization of fused test-set embeddings colored by genre is saved at [`result/plots/task3_tsne.png`](result/plots/task3_tsne.png).

---

## Notes on Source Code

The `src/` scripts are notebook exports provided for structural compliance. Because multiple modular components were developed within unified notebooks to manage memory and runtime flow, the export mapping is as follows:

- `task_2.py` covers the logic of: `audio_features`, `graph_builder`, and `gnn_model`
- `fusion_model.py` covers the logic of: `bert_encoder` and `fusion_model`

Full, executable implementations are contained in the `notebooks/` directory.