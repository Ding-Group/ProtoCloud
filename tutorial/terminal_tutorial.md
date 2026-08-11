# Terminal call

## 1. Command-line interface (quick start) 

Run the model straight from the terminal after cloning the repository.

```bash
python main.py --dataset_name my_dataset --model_mode train
```

### Input data

The dataset is located by name, not by full path: `--dataset_name <dataset>` loads
`<data_dir>/<dataset>.h5ad`. `--data_dir` must end with a path separator (its default
`./data/` already does), so `--dataset_name PBMC_10K` reads `./data/PBMC_10K.h5ad`.

The `.h5ad` file must provide:

| Location | Requirement |
| -------- | ----------- |
| `adata.var['gene_name']` | Gene names. Required; the run stops if this column is missing. `adata.var_names` is not used as a fallback. |
| `adata.obs['celltype']` | Cell type labels. Required for `train` and for any evaluation; without it the data can only be passed through a pretrained model. |
| `adata.layers['counts']` | Raw counts. Used when `--raw 1` (the default). |
| `adata.X` | Used when `--raw 0`, and also when `--raw 1` but no `counts` layer exists. |

The default decoder is a negative binomial (`--obs_dist nb`), which expects raw integer
counts. Supply counts in `adata.layers['counts']` (or in `adata.X` with no `counts`
layer present) and leave `--raw 1`. Use `--raw 0` only when `adata.X` holds the values
you want the model to see.

Files are read as-is by default. Pass `--preprocess_data 1` to run the built-in
filtering and normalization controlled by `--filter_gene_by_counts`,
`--filter_cell_by_counts`, `--normalize_total` and `--log1p`.

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
| `--dataset_name` | Name of the dataset | - |
| `--raw` | Use raw data (1) or normalized data (0) | `1` |
| `--batch_size` | Batch size for training | `128` |
| `--topngene` | Number of genes to select | - |
| `--test_ratio` | Ratio of test data | `0.1` |
| `--data_balance` | Balance training data (1) or not (0) | `1` |
| `--new_label` | Use previous predicted label as target (1) or not (0) | `0` |
| `--index_file` | Path of split indices `.csv` file with columns `train_idx` and `test_idx` | `None` |
| `--split` | Cluster name in `adata.obs` to use as the test set; `--test_ratio` is used when unset | `None` |
| `--preprocess_data` | Run the built-in preprocessing (1) or use the file as-is (0) | `0` |
| `--filter_gene_by_counts` | Minimum counts for a gene to be kept (preprocessing only) | `500` |
| `--filter_cell_by_counts` | Minimum counts for a cell to be kept (preprocessing only) | `1000` |
| `--normalize_total` | Target total counts per cell (preprocessing only) | `1e4` |
| `--log1p` | Apply log1p after normalization (preprocessing only) | `1` |

**Model Parameters:** 

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_mode` | Run mode: `train`, `test`, `apply` or `plot` | `apply` |
| `--model_name` | Model name used to build the experiment code | `protoCloud` |
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
| `--stage1_ortho_coef` | Orthogonality coefficient during stage 1 of two-step training | `0.0` |
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
| `--two_latent` | Plot pairs of raw latent dimensions | `0` |
| `--protocorr` | Plot prototype correlation | `0` |
| `--distance_dist` | Plot latent distance distribution to prototypes | `0` |

**Explainability Parameters:**

| Parameter    | Description                     | Default |
| ------------ | ------------------------------- | ------- |
| `--prp`      | Generate PRP-based explanations | `1`     |
| `--plot_prp` | Plot PRP explanation plots      | `0`     |


## Output files

Both output roots are scoped by dataset name: `--model_dir ./saved_models/` with
`--dataset_name PBMC_10K` writes to `./saved_models/PBMC_10K/`. `<exp_code>` is
generated from the run arguments, e.g. `protoCloud_lr1e-03_e100_b128_PBMC_10K`.

```text
<model_dir>/<dataset_name>/
    <exp_code>.pth             model weights
    model_dict.pkl             architecture needed to rebuild the model
    cell_encoder.joblib        cell type to integer mapping
    gene_names.txt             gene order the model expects
    cls_threshold.csv          per-class certainty thresholds
    calibrator_model.pkl       similarity calibrator
    prototype_checkpoint.npy   written when --prp 1

<results_dir>/<dataset_name>/
    <exp_code>_pred.csv        predictions
    <exp_code>_idx.csv         train/test split indices
    <exp_code>_latent.npy      latent embeddings (--save_file 1)
    <exp_code>_prototypes.npy  prototype coordinates (--save_file 1)
    <exp_code>_trend.npy       per-epoch training trend
    args.pkl                   full argument set of the run
    training.log               console output of the run
    plots/                     figures
    mis_anno/
    prp/                       PRP outputs (--prp 1)
```

Consecutive runs on the same dataset share these directories and overwrite each other.
Pass a per-run `--results_dir` (and `--model_dir`) when several runs must be kept side
by side.

## Examples

**Apply new dataset to a pre-trained model:**

```bash
python main.py \
  --dataset_name my_dataset \
  --model_mode apply \
  --pretrain_model_pth ./saved_models/my_dataset/protoCloud_lr1e-03_e100_b128_my_dataset.pth
```



`--pretrain_model_pth` points at the `.pth` file itself, and the rest of the model is
read from the directory containing it. That directory must also hold `model_dict.pkl`,
`cell_encoder.joblib`, `cls_threshold.csv` and `calibrator_model.pkl` - the layout a
previous `--model_mode train` run produced. Genes are matched by name against
`gene_names.txt`, so the new dataset does not need the same gene set; missing genes are
zero-filled and the overlap is reported.

**Continue training on existing model:**

```bash
python main.py \
  --dataset_name my_dataset \
  --model_mode train \
  --cont_train 1 \
  --epochs 30 \
  --pretrain_model_pth ./saved_models/my_dataset/protoCloud_lr1e-03_e100_b128_my_dataset.pth \
  `# <Optional> To use predicted label as new label, you must apply the new data first` \
  --new_label 1 \
  --results_dir ./results/my_dataset
```
