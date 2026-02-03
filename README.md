# ProtoVAE

The official PyTorch implementation of "ProtoVAE: A Trustworthy Self-Explainable Prototypical Variational Model" (NeurIPS 2022, https://nips.cc/Conferences/2022/ScheduleMultitrack?event=53023) by Srishti Gautam,  Ahcene Boubekki, Stine Hansen, Suaiba Amina Salahuddin, Robert Jenssen, Marina MC Höhne, Michael Kampffmeyer.

The code is built upon ProtoPNet's official implementation (https://github.com/cfchen-duke/ProtoPNet) and LRP implementation from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet.

## Update
result1: from updated model with 2-step training

<!-- ### Setup

Install a new conda environment 
```sh
conda env create -f requirements.yml
conda activate protovae
``` -->


## Prototype VAE User Guide

### Path Parameters
- `--dataset`: Specify the dataset to use. Available options are: `PBMC_10K`, `TSCA_lung`, `TSCA_spleen`, and `TSCA_oesophagus`.
- `--data_dir`: Path to the directory where the data is located.
- `--model_dir`: Path to the directory where the trained models will be saved.
- `--results_dir`: Path to the directory where the results will be saved.

#### Model Parameters
- `--model`: Specify the model to use. Available options are: `celltypist` and `protoVAE`.
- `--model_mode`: Specify the mode of the model. Available options are: `train` and `test`.
- `--test_ratio`: Ratio of data to use for testing.

#### Optimizer Parameters + Survival Loss Function
- `--activation`: Specify the activation function. Available options are: `relu` and `leakyrelu`.
- `--optimizer`: Specify the optimizer. Available options are: `Adam`.
- `--lr`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.
- `--epochs`: Number of epochs for training.
- `--target_accu`: Minimum target accuracy to save the model.
- `--seed`: Seed value for reproducibility.

#### protoVAE Parameters
- `--prototypes_per_class`: Number of prototypes per class.
- `--hidden_dim`: Dimension of the hidden layer.
- `--latent_dim`: Dimension of the latent space.

#### Result plotting Parameters
- `--visual_result`: Generate all visual results.
- `--plot_trend`: Plot training trend vs epoch.
- `--cm`: Plot confusion matrix.
- `--umap`: Plot UMAP.
- `--prp`: Apply LRP

#### Example Usage
To train the Prototype VAE model using the `PBMC_10K` dataset, use the following command:
```
python main.py --dataset PBMC_10K --model protoVAE --model_mode train
```

To test the trained model, use the following command:
```
python main.py --dataset PBMC_10K --model protoVAE --model_mode test
```

Please adjust the parameters according to your specific requirements.

Note: The default values mentioned here are based on the provided arguments in the code. You can modify them as per your needs.

For additional assistance, please refer to the code documentation or contact the developer.
