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
├── MS2_Data_Exploration.ipynb          # Milestone 2: PySpark data exploration
├── MS3_Preprocessing_and_Modeling.ipynb # Milestone 3: Full preprocessing pipeline + 3 distributed models
└── expanse_setup.sh                    # Expanse environment setup
```

## Notebooks

| Notebook | Branch | Description |
|---|---|---|
| [MS2_Data_Exploration.ipynb](MS2_Data_Exploration.ipynb) | `Milestone2` | PySpark schema inspection, null analysis, impression/click distributions, article EDA |
| [MS3_Preprocessing_and_Modeling.ipynb](MS3_Preprocessing_and_Modeling.ipynb) | `Milestone3` | End-to-end Spark MLlib preprocessing pipeline, Random Forest + 2 GBT models, fitting analysis |

## Milestones

| MS | Deliverable | Branch | Status |
|----|-------------|--------|--------|
| MS2 | Data Exploration | `Milestone2` | ✅ Complete |
| MS3 | Preprocessing & First Distributed Model | `Milestone3` | ✅ Complete |
| MS4 | Advanced Modeling (XGBoost + Neural) | TBD | Planned |

## SDSC Expanse Environment Setup

### Cluster Configuration

| Resource | Value |
|---|---|
| Cluster | SDSC Expanse |
| Partition | `shared` (single-node) |
| Account | `uci157` |
| Nodes | 1 |
| CPUs | 16 |
| Memory | 128 GB |
| Time Limit | 12 hr (720 min) |
| Job Launcher | Galyleo (Jupyter) |

### SparkSession Configuration

In `local[N]` mode the driver IS the executor, so only `spark.driver.memory` matters. For MS3, 96 GB is allocated to the driver, leaving headroom for the OS and JVM overhead on the 128 GB node.

**MS2 (Exploration):**
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

**MS3 (Preprocessing & Modeling):**
```python
spark = (
    SparkSession.builder
    .appName("EB-NeRD MS3 Preprocessing & Modeling")
    .master("local[15]")
    .config("spark.driver.memory", "96g")
    .config("spark.driver.maxResultSize", "16g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.local.dir", "/expanse/lustre/scratch/<user>/temp_project/spark_local")
    .config("spark.sql.parquet.enableVectorizedReader", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .config("spark.checkpoint.compress", "true")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.3")
    .getOrCreate()
)
```

**Why these settings (MS3):**

- **96 GB driver memory** — in `local[*]` mode the driver IS the executor, so `spark.executor.memory` has no effect. 96 GB fits the 440M-row exploded candidate table, pipeline transforms, and three tree ensemble models in memory on the 128 GB node.
- **800 shuffle partitions** — 15 threads over a ~100M-row downsampled table; finer partitions keep per-task heap footprint manageable.
- **Lustre scratch for `spark.local.dir`** — shuffle spill and checkpoint data go to `/expanse/lustre/scratch/` to avoid filling the home directory quota.
- **Disabled broadcast joins** — joining large tables with broadcast would OOM; forcing sort-merge join is safer.
- **Checkpointing** — breaks long lineage chains before the pipeline fit and after transforms to prevent re-computation OOM.
- **Vectorized parquet reader** — uses Arrow for faster columnar I/O since the whole dataset is parquet.

## Abstract

This project builds an end-to-end large-scale pipeline for native recommendation click prediction and ranking using the EB-NeRD dataset, with a focus on the cold-start problem for newly published content. Using Spark on SDSC Expanse, we transform impression logs into an impression–candidate training table, engineer time-aware features (to prevent label leakage), and generate cohorts that represent cold items (early life impressions) versus warm items (sufficient interaction history). We then train and tune CTR/ranking models using Ray, comparing interaction-only baselines, content-only models that generalize to unseen items, and hybrid approaches that blend content with early engagement signals. Evaluation emphasizes ranking quality (e.g., NDCG@k and MRR@k) and calibrated CTR prediction, reported separately for cold and warm items to quantify cold-start degradation and the lift provided by hybrid strategies.

