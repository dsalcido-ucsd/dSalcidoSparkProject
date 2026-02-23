# EB-NeRD Click Prediction & Cold-Start Analysis

**DSC 232R — Spark Project**

## Project Question

> How well can we rank and predict clicks for new articles when historical interactions are sparse, and what's the best hybrid approach using content features + limited early signals?

## Dataset

**EB-NeRD (Ekstra Bladet News Recommendation Dataset)** — a large-scale public dataset released for the RecSys Challenge 2024.

- **Link:** [https://recsys.eb.dk/dataset/](https://recsys.eb.dk/dataset/)
- **Large bundle:** 6 weeks of activity (Apr 27 – Jun 8, 2023)
- ~37.9M impression logs, ~1.1M users, ~125K articles, ~213M historical interactions
- Well above the 10GB minimum — the exploded impression–candidate training table reaches tens to hundreds of GB

## Repository Structure

```
├── README.md
├── MS2_Data_Exploration.ipynb     # PySpark data exploration
└── expanse_setup.sh               # Expanse environment setup
```

## Milestones

| MS | Deliverable | Status |
|----|-------------|--------|
| MS2 | Data Exploration (this milestone) | ✅ |
| MS3 | Preprocessing & Feature Engineering | Planned |
| MS4 | Modeling & Evaluation | Planned |

## SDSC Expanse Environment Setup

### Cluster Configuration

| Resource | Value |
|---|---|
| Cluster | SDSC Expanse |
| Partition | `shared` (single-node) |
| Account | `uci157` |
| Nodes | 1 |
| CPUs | 8 |
| Memory | 64 GB |
| Time Limit | 1 hr 15 min |
| Job Launcher | Galyleo (Jupyter) |

### SparkSession Configuration

I derived executor settings from the allocated resources using the formula:

```
Executor instances = Total Cores − 1 = 8 − 1 = 7
Driver memory     = 8 GB  (1 core reserved for the driver)
Executor memory   = (Total Memory − Driver Memory) / Executor Instances
                  = (64 GB − 8 GB) / 7
                  = 8 GB per executor
```

```python
spark = (
    SparkSession.builder
    .appName("EB-NeRD Data Exploration")
    .master("local[7]")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.executor.instances", "7")
    .config("spark.sql.shuffle.partitions", 200)
    .config("spark.sql.parquet.enableVectorizedReader", "true")
    .getOrCreate()
)
```

**Why these settings:**

- **1 core for the driver** so it can coordinate tasks and collect results for plotting without competing with executors.
- **8 GB driver memory** is enough for the small aggregated DataFrames I collect for matplotlib; the heavy work stays distributed.
- **7 executors × 8 GB = 56 GB** uses the remaining memory evenly, which matters for the shuffle-heavy joins and explode operations on ~38M impressions.
- **200 shuffle partitions** is a reasonable middle ground for this data size. I'll tune it up in MS3 when the exploded training table hits ~440M rows.
- **Vectorized parquet reader** uses Arrow for faster columnar I/O since the whole dataset is parquet.

## Abstract

This project builds an end-to-end large-scale pipeline for native recommendation click prediction and ranking using the EB-NeRD dataset, with a focus on the cold-start problem for newly published content. Using Spark on SDSC Expanse, we transform impression logs into an impression–candidate training table, engineer time-aware features (to prevent label leakage), and generate cohorts that represent cold items (early life impressions) versus warm items (sufficient interaction history). We then train and tune CTR/ranking models using Ray, comparing interaction-only baselines, content-only models that generalize to unseen items, and hybrid approaches that blend content with early engagement signals. Evaluation emphasizes ranking quality (e.g., NDCG@k and MRR@k) and calibrated CTR prediction, reported separately for cold and warm items to quantify cold-start degradation and the lift provided by hybrid strategies.

