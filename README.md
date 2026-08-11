# ProtoCloud
[![DOI](https://zenodo.org/badge/678277836.svg)](https://doi.org/10.5281/zenodo.18882740)
[![Cell Genomics](https://img.shields.io/badge/Cell%20Genomics-Published-green)](https://doi.org/10.1016/j.xgen.2026.101217)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.06.704364-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.02.06.704364v1)

ProtoCloud is a prototype-based interpretable deep learning model for single-cell RNA sequence analysis. It combines variational autoencoder (VAE) architecture with prototype learning to provide both cell type classification and interpretable features.

![ProtoCloud overview: scRNA-seq UMI counts and reference cell types are encoded into a prototype-structured latent space, yielding a predicted cell type with a similarity score and a certain/ambiguous call, plus gene-level explanations obtained by propagating prototype relevance back to the input genes.](assets/graphical_abstract.png)

*Graphical abstract from Guo & Ding, Cell Genomics (2026).*

## Installation

ProtoCloud is installed from source.

```bash
git clone https://github.com/Ding-Group/ProtoCloud.git
cd ProtoCloud
pip install -e .
```

To install into a dedicated conda environment instead:

```bash
git clone https://github.com/Ding-Group/ProtoCloud.git
cd ProtoCloud
conda env create -f requirements.yml   # creates an environment named protoCloud
conda activate protoCloud
pip install -e .
```

Check that the installation worked:

```bash
python -c "import ProtoCloud; print(ProtoCloud.__version__)"
```

Core dependencies (see `pyproject.toml` for the exact pip requirements):
numpy, pandas, scipy, scanpy, anndata, scikit-learn, torch, torchvision,
matplotlib, matplotlib-venn, seaborn, umap-learn.


**Command line.** `main.py` sits at the repository root and is not installed as a
console command, so the terminal workflow is run as `python main.py` from inside a
clone of this repository.

## Usage

ProtoCloud can be used in **two different ways: via terminal call or package installation** — whichever fits best with your workflow.

The shortest terminal run: put your dataset at `./data/ref_dataset.h5ad`, with cell type
labels in `adata.obs['celltype']`, gene names in `adata.var['gene_name']` and raw counts
in `adata.layers['counts']`. Then, from the repository root:

```bash
python main.py --dataset_name ref_dataset --model_mode train
```

This writes the trained model to `./saved_models/ref_dataset/`, and the predictions,
plots and gene-level PRP explanations to `./results/ref_dataset/`.

To annotate another dataset with that model, put it at `./data/query_dataset.h5ad` and
point `--pretrain_model_pth` at the checkpoint the training run wrote:

```bash
python main.py \
  --dataset_name query_dataset \
  --model_mode apply \
  --pretrain_model_pth ./saved_models/ref_dataset/protoCloud_lr1e-03_e100_b128_ref_dataset.pth
```

The query dataset needs `adata.var['gene_name']` but not `adata.obs['celltype']`. Genes are
matched to the reference model by name, so the two datasets need not share a gene set;
missing genes are zero-filled and the overlap is reported. Per-cell results are written to
`./results/query_dataset/<exp_code>_pred.csv`, with the predicted type in `pred1`, the
prototype similarity in `sim_score`, and the certainty calls in `certainty` and
`calibrated_certainty`.

For more terminal usage examples, see [tutorial/terminal_tutorial.md](tutorial/terminal_tutorial.md). For API usage, see [tutorial/api_tutorial.ipynb](tutorial/api_tutorial.ipynb).


## Citation

If you use ProtoCloud in your research, please cite:

```bibtex
@article{guo2026protocloud,
  title   = {ProtoCloud: A prototypical self-explaining model for single-cell analysis},
  author  = {Guo, Kaiyun and Ding, Jiarui},
  journal = {Cell Genomics},
  year    = {2026},
  note    = {Advance online publication},
  doi     = {10.1016/j.xgen.2026.101217},
  url     = {https://doi.org/10.1016/j.xgen.2026.101217}
}
```

## Reference
The code is built upon protoVAE (https://github.com/SrishtiGautam/ProtoVAE) and LRP implementation from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet. We thank the respective authors for making their code available to the community.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

For detailed information about each parameter and its impact on the ProtoCloud model, please refer to the code documentation and relevant literature.



