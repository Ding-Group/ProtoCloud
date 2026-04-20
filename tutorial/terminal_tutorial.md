# Terminal call

## 1. Command-line interface (quick start) 

Run the model straight from the terminal after cloning the repository. 
Datasets must annotated with a `celltype` column in `adata.obs` and a `gene_name` column in `adata.var`.
If raw count data are used as input, the counts can be stored in `adata.layers['counts']` or directly in `adata.X`.

```bash
python main.py --dataset_name <dataset> --model_mode train
```

ProtoCloud supports four running modes by `--model_mode`, default mode is `apply`:

| Option  | Description                |
| ------- | -------------------------- |
| `train` | Train the model |
| `test`  | Test the model on train-test-split data |
| `apply` | Apply the model to whole dataset |
|  `plot` | Load and plot result files using test data |


## Parameters

This document provides an overview of the parameters used in the ProtoCloud. Please review and modify these parameters according to your needs.

**Path Parameters:**

| Parameter       | Description                | Default           |
| --------------- | -------------------------- | ----------------- |
| `--data_dir`    | Directory for input data   | `./data/`         |
| `--model_dir`   | Directory for saved models | `./saved_models/` |
| `--results_dir` | Directory for results      | `./results/`      |

**Data Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset_name` | Name of the dataset | `PBMC_10K` |
| `--raw` | Use raw data (1) or normalized data (0) | `1` |
| `--batch_size` | Batch size for training | `128` |
| `--topngene` | Number of genes to select | - |
| `--test_ratio` | Ratio of test data | `0.2` |
| `--data_balance` | Balance training data (1) or not (0) | `1` |
| `--new_label` | Use previous predicted label as target (1) or not (0) | `0` |
| `--index_file` | Path of split indices `.csv` file with columns `train_idx` and `test_idx` | `None` |

**Model Parameters:** 

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cont_train` | Load existing model and continue training | `0` |
| `--model_validation` | Show validation accuracy in training stage | `1` |
| `--pretrain_model_pth` | Full path of pre-trained model to load | - |
| `--encoder_layer_sizes` | List of encoder layer sizes | - |
| `--decoder_layer_sizes` | List of decoder layer sizes | - |
| `--prototypes_per_class` | Number of prototypes per class | `6` |
| `--latent_dim` | Dimension of latent space | `20` |

**Loss Function Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--two_step` | Use two-step training (1) or not (0) | `1` |
| `--recon_coef` | Reconstruction loss coefficient | `10` |
| `--kl_coef` | KL divergence loss coefficient | `2` |
| `--ortho_coef` | Orthogonality loss coefficient | `0.3` |
| `--atomic_coef` | Atomic loss coefficient | `1` |

**Network Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Learning rate | `1e-3` |
| `--epochs` | Number of training epochs | `100` |
| `--seed` | Random seed | `7` |

**Results and Visualization Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--save_file` | Save all files | `1` |
| `--visual_result` | Generate all visualizations below | `1` |
| `--plot_trend` | Plot training trend vs epoch | `0` |
| `--cm` | Plot confusion matrix | `0` |
| `--umap` | Plot UMAP visualization | `0` |
| `--two_latent` | Plot misclassified points UMAP | `0` |
| `--protocorr` | Plot prototype correlation | `0` |
| `--distance_dist` | Plot latent distance distribution to prototypes | `0` |

**Explainability Parameters:**

| Parameter    | Description                     | Default |
| ------------ | ------------------------------- | ------- |
| `--lrp`      | Generate LRP-based explanations | `1`     |
| `--plot_lrp` | Plot LRP explanation plots      | `0`     |

## Examples

**Apply new dataset to a pre-trained model:**

```bash
python src/ProtoCloud/main.py \
  --dataset <data_name>
  --model_mode apply \
  --pretrain_model_pth ./saved_models/my_model.pth \
```


**Continue training on existing model:**

```bash
python src/ProtoCloud/main.py \
  --dataset <data_name>
  --model_mode train \
  --cont_train 1 \
  --epochs 30 \
  --pretrain_model_pth ./saved_models/<my_model>.pth
  # <Optional> To use predicted label as new label, you must apply the new data first
  --new_label 1 \
  --results_dir <path_to_applied_result_path>
```
