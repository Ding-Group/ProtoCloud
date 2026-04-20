# ProtoCloud
[![DOI](https://zenodo.org/badge/678277836.svg)](https://doi.org/10.5281/zenodo.18882740)
[![Cell Genomics](https://img.shields.io/badge/Cell%20Genomics-Published-green)](https://doi.org/10.1016/j.xgen.2026.101217)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.06.704364-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.02.06.704364v1)

ProtoCloud is a prototype-based interpretable deep learning model for single-cell RNA sequence analysis. It combines variational autoencoder (VAE) architecture with prototype learning to provide both cell type classification and interpretable features.

## Installation

```bash
conda env create -f requirements.yml
conda activate protocloud
!pip install -e .
```

Dependencies:
  - numpy>=1.26.4
  - pandas>=2.2.2
  - scipy>=1.13.1
  - scanpy>=1.10.2
  - anndata>=0.10.8
  - scikit-learn
  - pytorch>=1.12.1
  - torchvision>=0.13.1
  - matplotlib
  - matplotlib_venn
  - seaborn
  - umap-learn

## Usage

ProtoCloud can be used in **two different ways: via terminal call or package installation** —whichever fits best with your workflow.
For terminal usage examples, see [tutorial/terminal_tutorial.md](tutorial/terminal_tutorial.md). For API usage, see [tutorial/api_tutorial.ipynb](tutorial/api_tutorial.ipynb).


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



