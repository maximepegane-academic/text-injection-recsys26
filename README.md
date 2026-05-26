# RecBole for Next-Basket Recommendation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![RecBole](https://img.shields.io/badge/Based_on-RecBole_1.2-green.svg)](https://recbole.io/)

This is an extension of the [RecBole](https://recbole.io/) benchmark library, specifically developed for the paper: 
> *"Can Text Improve Next-Basket Recommendation? The Offline and Online Evidence from a Live Click & Collect Platform"* (RecSys 2026).

This repository provides a unified framework to reproduce our research on **multimodal e-grocery recommendation**. It introduces architectures that fuse ID embeddings with Pre-trained Language Model (PLM) text embeddings to break popularity bias.

---

##  General Purpose

In e-grocery and Next-Basket Recommendation (NBR), systems frequently collapse into exploitation engines that maximize offline metrics by predicting habitual repeat purchases (e.g., buying the same milk every week). While this reflects past behavior accurately, it generates limited incremental revenue.

**RecBole-CAT** is designed to evaluate how textual semantics (via PLMs) can improve long-tail item discovery without destroying the user's core routine basket. It includes standard ID-based baselines, early-fusion NLP models, computationally efficient late-fusion models (Gated Mean-Pooling), and our proposed **Cross-Attention Transformer (CAT)**.

##  Key Differences with RecBole

While built on the robust RecBole framework, this repository introduces several major modifications tailored for multimodal and industry-aligned evaluation:

1. **Polars and batch data processing:** we use Polars for data processing, and applies basket unrolling by user chunks to save memory usage and treat large datasets.
2. **Explore vs. Repeat Evaluation:** We split standard `Recall@K` and `NDCG@K` into **Repeat Recall** (items the user has bought before) and **Explore Recall** (novel discovery).
3. **Popularity Segmentation:** Metrics are automatically grouped into global popularity buckets (`Top 30`, `30-300`, `300-3K`, `3K-30K`, `30K+`) to accurately measure the "Tail Transition Rank" and evaluate true long-tail performance.
4. **New Architectures Implemented:**
   * `CAT` (Cross-Attention Transformer for Deep Fusion)
   * `GatedPoolingRecommender` (Dynamic Neural Gate for Shallow Fusion)
   * `SASRecNLP` (Standard Early Fusion Baseline)
   * `SeqPop` / `SeqPersonalPop` (Rigorous behavioral baselines)

---


```bash
# Clone the repository
git clone [https://github.com/YourUsername/RecBole-CAT.git](https://github.com/YourUsername/RecBole-CAT.git)
cd RecBole-CAT

# Install dependencies
pip install -r requirements.txt
pip install -e . --verbose
```
---

##  Recipes for reproducibility

Below are the command used to reproduce the results. The datasets splits are available at https://www.kaggle.com/datasets/lcmaxime/maximepegane-academic-text-injection-recsys26
download and unzip the content in `./data/instacart/`

### Baseline

**Global Popularity**
```bash
python -m run_experiment --dataset=instacart --model=SeqPop --epochs=1 --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=False --optimize=True
```

**Personal Popularity**
```bash
python -m run_experiment --dataset=instacart --model=SeqPersonalPop --epochs=1 --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=False --optimize=True
```

**SASRec**
```bash
python -m run_experiment --dataset=instacart --model=SASRec --batch_data_processing=True --epochs=1 --full_nlp=False --hugging_face_model=efederici/sentence-bert-base --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=True --optimize=True --clip_grad_norm=1.0 --learning_rate=5.063466e-04 --hidden_size=32 --n_layers=1 --n_heads=1
```
### NLP injected architectures

**SASRecNLP**
```bash
-m run_experiment --dataset=instacart --model=SASRecNLP --batch_data_processing=True --epochs=1 --full_nlp=False --hugging_face_model=efederici/sentence-bert-base --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=True --optimize=True --clip_grad_norm=1.0 --learning_rate=1.534467e-04 --hidden_size=16 --n_layers=1 --n_heads=2
```

**GLLF**
```bash
python -m run_experiment --dataset=instacart --model=GatedPoolingRecommender --epochs=1 --hugging_face_model=efederici/sentence-bert-base --repeatable=True --log_wandb=True --wandb_project=htuning_instacart --is_hyperparam_search=True --optimize=True --clip_grad_norm=1.0 --learning_rate=1.606167e-04 --hidden_size=32 --batch_data_processing=True
```

**CAT**
```bash
-m run_experiment --dataset=instacart --model=CAT --batch_data_processing=True --epochs=1 --full_nlp=False --hugging_face_model=efederici/sentence-bert-base --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=True --optimize=True --clip_grad_norm=1.0 --learning_rate=2.796893e-07 --hidden_size=32 --n_layers=4 --n_heads=4
```

full token representation version:
```bash
-m run_experiment --dataset=instacart --model=CAT --batch_data_processing=True --epochs=1 --full_nlp=True --hugging_face_model=efederici/sentence-bert-base --repeatable=True --log_wandb=True --wandb_project=measure_instacart --is_hyperparam_search=True --optimize=True --clip_grad_norm=1.0 --learning_rate=2.796893e-07 --hidden_size=32 --n_layers=4 --n_heads=4
```


