# protoCloud


## Update
The code is built upon protoVAE (https://github.com/SrishtiGautam/ProtoVAE) and LRP implementation from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet.

## Prototype VAE User Guide
# Prototype VAE - Model Manual

This document provides an overview of the parameters used in the Prototype VAE model. Please review and modify these parameters according to your needs.

### Path Parameters
- `data_dir`: Path to the directory containing the dataset. Default: `./data/`
- `model_dir`: Path to the directory to save trained models. Default: `./saved_models/`
- `results_dir`: Path to the directory to save experimental results. Default: `./results/`

### Data Parameters
- `dataset`: Name of the dataset to use. Default: `PBMC_10K`
- `raw`: Whether to use raw data or normalized data (0 or 1). Default: `1`
- `batch_size`: Batch size for training. Default: `128`
- `topngene`: Number of genes to select. 
- `test_ratio`: Ratio of data to use for testing. Default: `0.1`
- `split`: Cluster name of test data. If `None`, use `test_ratio`.

### Model Parameters
- `model`: Model type to use. Default: `protoVAE`
- `model_mode`: Mode of the model ("train", "test", "apply", "plot"). Default: `test`
- `cont_train`: Load existing model and continue training (0 or 1). Default: `0`
- `pretrain_model_pth`: Full path of pre-trained model to load.
- `two_step`: Use two-step training or not (0 or 1). Default: `1`
- `ortho_coef`: Orthogonality loss coefficient. Default: `0.3`
- `activation`: Activation function for the model (currently supports 'relu'). Default: `relu`
- `use_bn`: Use batch normalization or not (0 or 1). Default: `1`
- `optimizer`: Optimizer for training ('Adam' or 'AdamW'). Default: `AdamW`
- `lr`: Learning rate for training. Default: `1e-3`
- `epochs`: Number of training epochs. Default: `100`
- `target_accu`: Minimum target accuracy to save the model. Default: `0.7`
- `seed`: Random seed for reproducibility. Default: `7`
- `obs_dist`: Observation distribution ('nb' or 'normal'). Default: `nb`
- `nb_dispersion`: Dispersion for negative binomial distribution ('celltype' or 'gene'). Default: `celltype`
- `prototypes_per_class`: Number of prototypes per class. Default: `6`
- `hidden_dim`: Dimension of hidden layers. Default: `32`
- `latent_dim`: Dimension of latent space. Default: `20`

### Result Plotting Parameters
- `visual_result`: Generate visual results or not (0 or 1). Default: `1`
- `plot_trend`: Plot training trend vs epoch (0 or 1). Default: `0`
- `cm`: Plot confusion matrix (0 or 1). Default: `0`
- `umap`: Plot UMAP visualization (0 or 1). Default: `0`
- `protocorr`: Plot prototype correlation (0 or 1). Default: `0`
- `prp`: Generate LRP based explanations (0 or 1). Default: `1`
- `plot_prp`: Plot LRP explanation plots (0 or 1). Default: `0`

For detailed information about each parameter and its impact on the ProtoCloud model, please refer to the code documentation and relevant literature.


#### Example Usage
To train the Prototype VAE model using the `PBMC_10K` dataset, use the following command:
```
python main.py --dataset PBMC_10K --model_mode train
```

Please adjust the parameters according to your specific requirements.
Note: The default values mentioned here are based on the provided arguments in the code. You can modify them as per your needs.
For additional assistance, please refer to the code documentation or contact the developer.
