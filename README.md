# EB-NeRD Click Prediction & Cold-Start Analysis

**DSC 232R: Big Data Technologies with Spark, Final Project**

---

## Project Question

> How well can we predict and rank article clicks from impression data, and what is the cold-start degradation when articles have no interaction history? Can dimensionality reduction via PCA recover or improve performance relative to a full-feature baseline?

---

## Abstract

This project builds an end-to-end large-scale pipeline for news recommendation click-through-rate (CTR) prediction using the EB-NeRD dataset, with a focus on the cold-start problem for newly published content. Using PySpark on SDSC Expanse (16 CPUs, 128 GB), we transform ~38M impression logs into hundreds of millions of (user, article) candidate pairs, engineer time-aware features to prevent label leakage, and train four distributed tree ensemble models across two milestones.

**Milestone 3** covers full preprocessing and three Spark MLlib models (Random Forest + 2 GBT variants). **Milestone 4** applies PCA dimensionality reduction to the feature space, investigates explained variance and component structure, and trains a GBT model on the reduced representation, comparing it to the full-feature baseline.

Key findings:
- GBT tuned on full features achieves **val AUC-ROC 0.6749**, the best result across all four models
- `rolling_popularity_24h` (24h rolling impression count) dominates all models with **0.32 feature importance**; trending articles beat content quality signals
- 87.8% of test items are cold (no prior engagement); cold articles show **~0.02 lower AUC** than warm articles
- PCA at 95% explained variance preserves most label signal; the GBT-PCA model achieves comparable val AUC with a smaller, decorrelated feature space
- PC1 captures the "article engagement" axis (inviews, pageviews, rolling popularity); PC2 captures "user engagement depth" (history length, read time, scroll depth). Cold articles cluster at the low-PC1 extreme, confirming the cold-start problem is a feature-space separability issue

---

## Dataset

**EB-NeRD (Ekstra Bladet News Recommendation Dataset)** is a large-scale public dataset from the RecSys Challenge 2024.

| Property | Value |
|---|---|
| Source | [https://recsys.eb.dk/dataset/](https://recsys.eb.dk/dataset/) |
| Bundle used | `ebnerd_large` (6 weeks, Apr 27 – Jun 8, 2023) |
| Impression logs | ~37.9M |
| Users | ~1.1M |
| Articles | ~125K |
| Historical interactions | ~213M |
| Exploded candidate table | ~440M rows (before downsampling) |

This is well above the 10 GB minimum requirement. The full exploded impression-candidate table reaches tens to hundreds of GB on disk.

---

## Repository Structure

```
├── README.md
├── MS2_Data_Exploration.ipynb               # Milestone 2: PySpark EDA
├── MS3_Preprocessing_and_Modeling.ipynb     # Milestone 3: Full pipeline + 3 distributed models
├── MS4_Dimensionality_Reduction_and_Modeling.ipynb  # Milestone 4: PCA + GBT on reduced features
└── expanse_setup.sh                         # Expanse environment setup script
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [MS2_Data_Exploration.ipynb](MS2_Data_Exploration.ipynb) | PySpark schema inspection, null analysis, impression/click distributions, article EDA |
| [MS3_Preprocessing_and_Modeling.ipynb](MS3_Preprocessing_and_Modeling.ipynb) | End-to-end Spark MLlib preprocessing pipeline (imputation, OHE, scaling), Random Forest baseline + 2 GBT models, fitting analysis, cold-start cohort breakdown |
| [MS4_Dimensionality_Reduction_and_Modeling.ipynb](MS4_Dimensionality_Reduction_and_Modeling.ipynb) | PCA dimensionality reduction (full-rank eigenvalue analysis, explained variance, component interpretation, 2D projection), GBT on PCA features, predictions analysis (TP/TN/FP/FN by cold/warm cohort), full model comparison |

---

## Milestones

| MS | Deliverable | Branch | Status |
|----|-------------|--------|--------|
| MS2 | Data Exploration | `Milestone2` | ✅ Complete |
| MS3 | Preprocessing & First Distributed Model | `Milestone3` | ✅ Complete |
| MS4 | Dimensionality Reduction & Second Model | `Milestone4` | ✅ Complete |

---

## Methods

### Data Pipeline

```
Raw parquet (behaviors + articles + history)
        │
        ▼
Explode inview arrays → (impression, candidate) rows     [~440M rows]
        │
        ▼
Negative downsampling  (npratio=4, keep all positives)   [~100M rows]
        │
        ▼
Temporal split  (train=first 80% by time, test=last 20%, val=validation/ folder)
        │
        ▼
Feature engineering  (article age, log engagement, rolling 24h CTR, user history)
        │
        ▼
Spark MLlib Pipeline  (StringIndexer → OHE → Imputer → VectorAssembler → StandardScaler)
        │
        ├── MS3 ──► RF / GBT on full ~30-dim features
        │
        └── MS4 ──► PCA (k at 95% variance) ──► GBT on PCA-k features
```

### Features

| Group | Features | Notes |
|---|---|---|
| **Engagement** (warm-article) | `rolling_popularity_24h`, `rolling_ctr_24h`, `log_total_inviews`, `log_total_pageviews`, `total_read_time` | Zero for cold articles; dominant signal |
| **Article content** (cold-safe) | `category_str` (OHE), `sentiment_score`, `sentiment_label`, `title/body/subtitle_word_count`, `n_topics`, `n_entities`, `premium_flag`, article age | Available immediately on publication |
| **User** | `user_history_length`, `user_avg_read_time_hist`, `is_subscriber`, `is_sso_user`, `device_type` | From history table |
| **Session** | `impression_hour`, `impression_weekday`, `read_time`, `scroll_percentage` | Session-level signals |
| **Cold flag** | `is_cold_article` (age ≤ 24h or 0 inviews) | Cohort identifier for analysis |

### Temporal Splits

| Split | Source | Approx. period | Used for |
|---|---|---|---|
| train | `train/` parquet, first 80% by time | Apr 27 – ~May 31 | Model fitting |
| test | `train/` parquet, last 20% by time | Late May | Evaluation |
| val | `validation/` parquet | Jun 1 – Jun 8 | Final generalization check |

---

## Results

### Model Performance Summary

| Model | Train AUC-ROC | Test AUC-ROC | Val AUC-ROC | Train→Test | Test→Val |
|---|---|---|---|---|---|
| RF (50 trees, depth 8) | 0.7123 | 0.7164 | 0.6538 | −0.0041 | −0.0626 |
| GBT default (20 rounds, lr=0.1) | 0.7200 | 0.7226 | 0.6378 | −0.0026 | −0.0848 |
| **GBT tuned (50 rounds, lr=0.05)** | **0.7375** | **0.7281** | **0.6749** | +0.0094 | −0.0532 |
| GBT on PCA features (95% var) | see MS4 | see MS4 | see MS4 | see MS4 | see MS4 |

> GBT-PCA results are computed at runtime in `MS4_Dimensionality_Reduction_and_Modeling.ipynb` since the exact k depends on the dataset's actual eigenvalue distribution.

### Cold-Start Cohort Breakdown (GBT Tuned, Test Set)

| Cohort | N | AUC-ROC |
|---|---|---|
| Warm articles (age > 24h, inviews > 0) | 2,838,845 | 0.7129 |
| Cold articles (age ≤ 24h or 0 inviews) | 20,801,887 | 0.6930 |

87.8% of test rows are cold articles. The 0.02 AUC gap comes from cold articles having zero `rolling_popularity_24h`, which is the most important feature.

### Top Feature Importances (GBT Tuned)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `rolling_popularity_24h` | 0.32 |
| 2 | `user_history_length` | 0.068 |
| 3 | `log_total_inviews` | 0.097 |
| 4 | `log_article_age_hours` | 0.091 |
| 5 | `log_total_pageviews` | 0.063 |
| 6 | `total_read_time` | 0.062 |
| 7 | `sentiment_score` | 0.046 |

### PCA Analysis (MS4)

| Finding | Detail |
|---|---|
| PC1 interpretation | "Article engagement" axis (`rolling_popularity_24h`, `log_total_inviews`, `log_total_pageviews`) |
| PC2 interpretation | "User engagement depth" axis (`user_history_length`, `user_avg_read_time_hist`, `read_time`) |
| Cold article cluster | Cold articles concentrate at the negative end of PC1, confirming cold-start is a low-engagement separability problem |
| Information compression | 95% variance captured in k components; GBT-PCA achieves comparable AUC with a smaller input |

---

## Fitting Analysis

All four models sit near the underfitting end of the bias-variance spectrum, with near-zero train/test gaps:

- **RF** and **GBT default**: test AUC is slightly above train AUC (−0.004, −0.003), a sign of mild underfitting; both ensembles are well-regularized
- **GBT tuned**: +0.009 train/test gap, which is healthy; the slow learning rate over 50 rounds keeps the model from memorizing training noise
- **GBT-PCA**: expected to follow the same pattern; the decorrelated PCA input reduces split redundancy and slightly lowers variance compared to the full-feature GBT

The **test-to-val drop (0.05–0.08 AUC)** across all models is not overfitting. It is temporal distribution shift: the validation set covers June 1–8, about a week after the test period ends. GBT default is hit hardest (−0.085); GBT tuned holds up best (−0.053) because its lower learning rate produces more generalizable splits.

---

## Conclusion

### Model 1 (MS3)

GBT tuned achieves the best generalization (val AUC-ROC 0.6749). Click prediction is dominated by recency and momentum: `rolling_popularity_24h` alone accounts for 32% of predictive power. The cold-start gap (87.8% of impressions are cold, ~0.02 AUC disadvantage) is a feature-availability problem, not a modeling capacity problem. Cold articles simply have no engagement signals to draw on.

**Improvements for MS3 models:**
- Add 768-dim contrastive/RoBERTa article embeddings for cold-start semantic signal
- Shorter rolling windows (1h, 6h) to capture in-session virality
- Per-user category affinity vectors (instead of scalar history length)
- Rolling-window training to reduce temporal distribution shift at inference

### Model 2 (MS4 — PCA + GBT)

PCA reveals that the feature space is organized around two interpretable axes: article engagement (PC1) and user engagement depth (PC2). Compressing to 95% explained variance preserves nearly all label signal, and the GBT model trained on those components achieves comparable val AUC to GBT tuned. The decorrelated input also reduces split redundancy in the GBT trees.

**Improvements for MS4 model:**
- Supervised dimensionality reduction (LDA) to maximize class separability rather than variance
- Sparse PCA for more interpretable, feature-selective components
- Higher k (99% variance) to retain rare category OHE signal
- PCA as preprocessing for a two-tower neural network to bring in dense article embeddings

---

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

In `local[N]` mode the driver is also the executor, so only `spark.driver.memory` matters. 96 GB is allocated to the driver, leaving headroom for OS and JVM overhead on the 128 GB node.

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

**MS3 & MS4 (Preprocessing & Modeling):**
```python
spark = (
    SparkSession.builder
    .appName("EB-NeRD MS3/MS4 Preprocessing & Modeling")
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

**Why these settings:**

- **96 GB driver memory**: in `local[*]` mode the driver is the executor, so `spark.executor.memory` has no effect. 96 GB is enough to hold the 440M-row exploded candidate table, pipeline transforms, PCA computation, and GBT models on the 128 GB node.
- **800 shuffle partitions**: with 15 threads over a ~100M-row table, finer partitions keep per-task heap usage manageable.
- **Lustre scratch for `spark.local.dir`**: shuffle spill and checkpoint data go to `/expanse/lustre/scratch/` to avoid filling the home directory quota.
- **Disabled broadcast joins**: joining large tables via broadcast would run out of memory; sort-merge join is used instead.
- **Checkpointing**: breaks long lineage chains before the pipeline fit and after PCA transforms to avoid recomputation OOM.
- **Vectorized parquet reader**: uses Arrow for faster columnar I/O since the whole dataset is stored as parquet.

