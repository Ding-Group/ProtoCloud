# protoCloud

ProtoCloud is a prototype-based interpretable deep learning model for single-cell RNA sequence analysis. It combines variational autoencoder (VAE) architecture with prototype learning to provide both cell type classification and interpretable features.

## Usage

### Basic Usage

```bash
python main.py --dataset_name PBMC_10K --model_mode train
```

### Running Modes

ProtoCloud supports four running modes:

```bash
# Train a new model
python main.py --model_mode train

# Test the model on test data
python main.py --model_mode test

# Apply the model to all data with reparametrization
python main.py --model_mode apply

# Load and plot result files using test data
python main.py --model_mode plot
```

## Parameters

This document provides an overview of the parameters used in the ProtoCloud. Please review and modify these parameters according to your needs.

### Path Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Directory for input data | `./data/` |
| `--model_dir` | Directory for saved models | `./saved_models/` |
| `--results_dir` | Directory for results | `./results/` |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset_name` | Name of the dataset | `PBMC_10K` |
| `--raw` | Use raw data (1) or normalized data (0) | `1` |
| `--batch_size` | Batch size for training | `128` |
| `--topngene` | Number of genes to select | - |
| `--test_ratio` | Ratio of test data | `0.1` |
| `--data_balance` | Use weighted sampling for training data (1) or not (0) | `1` |
| `--new_label` | Use previous predicted label as target (1) or not (0) | `0` |
| `--index_file` | Full path of split indices with columns train_idx and test_idx | `None` |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Name of the model | `protoCloud` |
| `--cont_train` | Load existing model and continue training | `0` |
| `--model_validation` | Show validation accuracy in training stage | `1` |
| `--pretrain_model_pth` | Full path of pre-trained model to load | - |
| `--encoder_layer_sizes` | List of encoder layer sizes | - |
| `--decoder_layer_sizes` | List of decoder layer sizes | - |
| `--prototypes_per_class` | Number of prototypes per class | `6` |
| `--latent_dim` | Dimension of latent space | `20` |

### Loss Function Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--obs_dist` | Observation distribution ('nb' or 'normal') | `nb` |
| `--two_step` | Use two-step training (1) or not (0) | `1` |
| `--recon_coef` | Reconstruction loss coefficient | `10` |
| `--kl_coef` | KL divergence loss coefficient | `2` |
| `--ortho_coef` | Orthogonality loss coefficient | `0.3` |
| `--atomic_coef` | Atomic loss coefficient | `1` |

### Network Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Learning rate | `1e-3` |
| `--epochs` | Number of training epochs | `100` |
| `--seed` | Random seed | `7` |

### Results and Visualization Parameters

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

### Explainability Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prp` | Generate all PRP-based explanations | `1` |
| `--lrp` | Generate LRP-based explanations | `1` |
| `--plot_prp` | Plot all PRP explanation plots | `0` |
| `--plot_lrp` | Plot LRP explanation plots | `0` |

## Examples

### Training a model with custom parameters

```bash
python main.py \
  --model_mode train \
  --dataset_name PBMC_10K \
  --raw 1 \
  --batch_size 256 \
  --test_ratio 0.2 \
  --latent_dim 30 \
  --prototypes_per_class 8 \
  --encoder_layer_sizes 128 64 \
  --decoder_layer_sizes 64 128 \
  --epochs 200
```

### Testing a pre-trained model

```bash
python main.py \
  --model_mode test \
  --pretrain_model_pth ./saved_models/my_model.pth \
  --save_file 1 \
  --visual_result 1
```



#### Example Usage
To train the Prototype VAE model using the `PBMC_10K` dataset, use the following command:
```
python main.py --dataset PBMC_10K --model_mode train
```


## Reference
The code is built upon protoVAE (https://github.com/SrishtiGautam/ProtoVAE) and LRP implementation from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

For detailed information about each parameter and its impact on the ProtoCloud model, please refer to the code documentation and relevant literature.

