import json
import os

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from torch.distributions import Multinomial

from src.utils import *
import src.glo as glo



def load_data(dataset_name, 
                 data_dir, 
                 raw:bool, 
                 celltype_col = "celltype",
                 gene_col = "gene_name",
                 topngene=None, **kwargs):
    """
    PBMC_10K: scvi-pbmcs_10k
    Tissue Stability Cell Atlas (TSCA): https://www.tissuestabilitycellatlas.org/
        Lung: https://cellgeni.cog.sanger.ac.uk/tissue-stability/lung.cellxgene.h5ad
        Spleen: https://cellgeni.cog.sanger.ac.uk/tissue-stability/spleen.cellxgene.h5ad
        Oesophagus: https://cellgeni.cog.sanger.ac.uk/tissue-stability/oesophagus.cellxgene.h5ad
    
    """
    if topngene is not None:
        save_path = data_dir + dataset_name + str(topngene) + '.h5ad'
    else:
        save_path = data_dir + dataset_name + '.h5ad'

    if os.path.exists(save_path):
        print('Loading dataset...')
        adata = sc.read(save_path)
    else:
        print(save_path)
        raise ValueError("Dataset not found!")
    
    adata.obs["celltype"] = adata.obs[celltype_col].astype("category")
    adata.var["gene_name"] = adata.var[gene_col]
    adata.uns['celltypes'] = np.unique(adata.obs["celltype"])
    adata.var_names_make_unique()
    # adata = adata[~adata.obs['celltype'].isna(), :]
    del adata
    print(adata)
    # adata.uns['log1p']["base"] = None

    return adata


        elif dataset_name.startswith("TSCA"):    # == "TSCA_lung" or dataset_name == "TSCA_spleen" or dataset_name == "TSCA_oesophagus":
            species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                # if data file does not exist, download, preprocess & save it
                print('Downloading dataset...')
                if dataset_name == "TSCA_lung":
                    adata = sc.read(data_dir +'lung.cellxgene.h5ad',
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/lung.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7509735', 'gene_ids-HCATisStab7509736', 'gene_ids-HCATisStab7587202', 'gene_ids-HCATisStab7587205', 'gene_ids-HCATisStab7587208', 'gene_ids-HCATisStab7587211', 'gene_ids-HCATisStab7646032', 'gene_ids-HCATisStab7646033', 'gene_ids-HCATisStab7646034', 'gene_ids-HCATisStab7646035', 'gene_ids-HCATisStab7659968', 'gene_ids-HCATisStab7659969', 'gene_ids-HCATisStab7659970', 'gene_ids-HCATisStab7659971', 'gene_ids-HCATisStab7747197', 'gene_ids-HCATisStab7747198', 'gene_ids-HCATisStab7747199', 'gene_ids-HCATisStab7747200'], axis = 1)
                
                elif dataset_name == "TSCA_spleen":
                    adata = sc.read(data_dir + 'spleen.cellxgene.h5ad', 
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/spleen.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7463847', 'gene_ids-HCATisStab7463848', 'gene_ids-HCATisStab7463849', 'gene_ids-HCATisStab7587200', 'gene_ids-HCATisStab7587203', 'gene_ids-HCATisStab7587206', 'gene_ids-HCATisStab7587209', 'gene_ids-HCATisStabAug177078016', 'gene_ids-HCATisStabAug177078017', 'gene_ids-HCATisStabAug177078018', 'gene_ids-HCATisStabAug177078019', 'gene_ids-HCATisStabAug177276391', 'gene_ids-HCATisStabAug177276392', 'gene_ids-HCATisStabAug177276394', 'gene_ids-HCATisStabAug177376561', 'gene_ids-HCATisStabAug177376563', 'gene_ids-HCATisStabAug177376565', 'gene_ids-HCATisStabAug177376567'], axis = 1)
                else:
                    # "TSCA_oesophagus"
                    adata = sc.read(data_dir + 'oesophagus.cellxgene.h5ad', 
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/oesophagus.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7413620', 'gene_ids-HCATisStab7413621', 'gene_ids-HCATisStab7413622', 'gene_ids-HCATisStab7587201', 'gene_ids-HCATisStab7587204', 'gene_ids-HCATisStab7587207', 'gene_ids-HCATisStab7587210', 'gene_ids-HCATisStab7619064', 'gene_ids-HCATisStab7619065', 'gene_ids-HCATisStab7619066', 'gene_ids-HCATisStab7619067', 'gene_ids-HCATisStab7646028', 'gene_ids-HCATisStab7646029', 'gene_ids-HCATisStab7646030', 'gene_ids-HCATisStab7646031', 'gene_ids-HCATisStabAug177184858', 'gene_ids-HCATisStabAug177184862', 'gene_ids-HCATisStabAug177184863', 'gene_ids-HCATisStabAug177376562', 'gene_ids-HCATisStabAug177376564', 'gene_ids-HCATisStabAug177376566', 'gene_ids-HCATisStabAug177376568'], axis = 1)
                

                adata.obs["celltype"] = adata.obs['Celltypes_updated_July_2020'].astype("category")
                adata.var["gene_name"] = adata.var['gene_col'].tolist()
                adata.obs = adata.obs.drop(columns = ['Donor', 'Time', 'donor_time', 'Celltypes_GenomeBiol_2019', 'Celltypes_updated_July_2020'])

                # recover raw counts according to obs['n_counts]
                raw = np.expm1(adata.raw.X.toarray())
                raw = np.round(raw)
                new_adata = anndata.AnnData(raw, obs = adata.obs, var = adata.var)
                new_adata.uns = adata.uns.copy()
                print("Recovered counts:", sum(np.sum(raw, axis = 1) == new_adata.obs['n_counts']))

            # get color for umap
            # cell_color = adata.uns["Celltypes_updated_July_2020_colors"]
            cell_color = None
        
        elif dataset_name == "RGC":
            species = "mouse"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(data_dir + 'rgc_atlas.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

            cell_color = None
        
        elif dataset_name.endswith("ONC"):
            species = "mouse"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(data_dir + dataset_name.lower() + '.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

                preprocess_arg['filter_gene_by_counts'] = False
                preprocess_arg['filter_cell_by_counts'] = False

            cell_color = None
        
        elif dataset_name == "ICA":
            species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(data_dir + dataset_name.lower() + '.h5ad')
                adata.var["gene_name"] = adata.var.index
            
            cell_color = None
        
        # elif dataset_name == "SM3":
        #     species = "human"
        #     if os.path.exists(save_path):
        #         print('Loading dataset...')
        #         adata = sc.read(save_path)
        #     else:
        #         adata = sc.read(data_dir + 'SM3_full.h5ad').T

        #         annotation = pd.read_csv(data_dir + 'PBMCs.allruns.barcode_annotation.txt', sep="\t")
        #         adata.obs = annotation
        #         adata = adata[annotation['QC_status'] == 'QCpass']

        #         adata.obs['celltype'] = adata.obs['celltype_lvl2_inex_10khvg_reads_res08_new'].tolist()
        #         adata.var["gene_name"] = adata.var.index.tolist()

        #         adata = _preprocess(adata, **preprocess_arg)
        #         adata.write(save_path, compression='gzip')
        #     cell_color = None

        # elif dataset_name == "GCA":
        #     species = "human"
        #     if os.path.exists(save_path):
        #         print('Loading dataset...')
        #         adata = sc.read(save_path)
        #     else:
        #         # if data file does not exist, download, preprocess & save it
        #         adata = sc.read(data_dir +'gutcellatlas.h5ad',
        #                         backup_url="https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Full_obj_raw_counts_nosoupx_v2.h5ad",
        #                         )
        #         adata.obs["celltype"] = adata.obs['category'].astype("category")
        #         adata.var["gene_name"] = adata.var.index.tolist()

        #         adata = _preprocess(adata,
        #             filter_gene_by_counts = 500,
        #             filter_cell_by_counts = 1000,
        #             normalize_total = 1e4,
        #             log1p = True,
        #             top_n_genes = topngene)
                
        #         adata.write(save_path, compression = 'gzip')
        #     cell_color = None
        
        elif dataset_name == "TS_spleen":
            species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                adata = sc.read(data_dir + 'new_spleen.h5ad')

            cell_color = None

        elif dataset_name.startswith("ATAC"):
            species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(data_dir + dataset_name + '.h5ad')

                # only for top_n_genes, assume X is raw
                adata = _preprocess(adata,
                        filter_gene_by_counts = False,
                        filter_cell_by_counts = False,
                        normalize_total = 1e4,
                        log1p = True)
                adata.X = adata.layers['counts']
                del adata.layers
                # adata.write(save_path, compression='gzip')
        
        elif os.path.exists(data_dir + dataset_name + '.h5ad'):
            # load un-predefined datasets
            print('Loading outside dataset, this will not process the data')

            if topngene is not None:
                top_gene_path = data_dir + dataset_name + str(topngene) + '.h5ad'
                if os.path.exists(top_gene_path):
                    adata = sc.read(top_gene_path)
                else:
                    adata = sc.read(data_dir + dataset_name + '.h5ad')
                    adata = _top_n_genes(adata)
                    adata.write(top_gene_path, compression='gzip')
            else:            
                adata = sc.read(save_path)

            # assert 'celltype' in adata.obs.columns, "The 'celltype' column is NOT present in adata.obs."
            assert 'gene_name' in adata.var.columns, "The 'gene_name' column is NOT present in adata.var."

            cell_color = None

        else:
            print(save_path)
            raise ValueError("Dataset not found!")
        

        #######################################################
        adata.var_names_make_unique()
        adata = adata#[~adata.obs['celltype'].isna(), :]
        del adata
        print(adata)
        # adata.uns['log1p']["base"] = None
        gene_names = adata.var["gene_name"].values

        # if raw and 'counts' in adata.layers:
        #     print("\tUsing layer 'counts' as input")
        #     X = np.float32(np.array(adata.layers['counts'].todense()) if sparse.isspmatrix(adata.layers['counts']) else adata.layers['counts'])
        # else:
        #     print("\tUsing adata.X as input")
        #     X = np.float32(np.array(adata.X.todense()) if sparse.isspmatrix(adata.X) else adata.X)

        if "celltype" in adata.obs.columns:
            celltypes = adata.obs["celltype"]
        else:
            print(f"\tDidn't find celltype annotation in dataset!")
            celltypes = None
        cell_encoder = None

        # if 'batch' in adata.obs_names:
        #     batch = adata.obs['batch'].tolist()
        #     n_batch = np.unique(batch)
        # else:
        #     n_batch = 1
        #     batch = np.ones(len(Y))


def load_PBMCs(dataset_name, data_dir):
    # if data file does not exist, download, preprocess & save it
    print('Preprocessing dataset...')
    if dataset_name == "PBMC_10K":
        adata = sc.read(data_dir + 'pbmc10k.h5ad')
    elif dataset_name == "PBMC_20K":
        adata = sc.read(data_dir +  'pbmc20k.h5ad')
    elif dataset_name == "PBMC_30K":
        adata = sc.read(data_dir +  'pbmc30k.h5ad')
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['batch'] = [int(name.split("pbmc")[-1]) for name in adata.obs['Experiment'].tolist()]                
    
    elif dataset_name == "PBMC1":
        adata = sc.read(data_dir +  'pbmc1.h5ad')
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        adata.var["gene_name"] = adata.var.index.tolist()
    elif dataset_name == "PBMC2":
        adata = sc.read(data_dir +  'pbmc2.h5ad')
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        adata.var["gene_name"] = adata.var.index.tolist()
    else:
        raise ValueError("Dataset not found!")
                    
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    adata.var["gene_name"] = adata.var.index.tolist()

    return adata