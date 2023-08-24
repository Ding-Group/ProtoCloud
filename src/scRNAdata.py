import json
import os

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import *
import src.glo as glo
EPS = glo.get_value('EPS')


class scRNAData():
    """
    PBMC_10K: scvi-pbmcs_10k
    Tissue Stability Cell Atlas (TSCA): https://www.tissuestabilitycellatlas.org/
        Lung: https://cellgeni.cog.sanger.ac.uk/tissue-stability/lung.cellxgene.h5ad
        Spleen: https://cellgeni.cog.sanger.ac.uk/tissue-stability/spleen.cellxgene.h5ad
        Oesophagus: https://cellgeni.cog.sanger.ac.uk/tissue-stability/oesophagus.cellxgene.h5ad
    
    """
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.data_dir = args.data_dir
        self.raw = args.raw
        self.topngene = args.topngene

        self.split = args.split
        self.time_order = None
        self.time_col = None
        self.test_ratio = args.test_ratio

        if self.topngene is not None:
            self.save_path = self.data_dir + self.dataset_name + str(self.topngene) + '.h5ad'
        else:
            self.save_path = self.data_dir + self.dataset_name + '.h5ad'


        # Loading and preparing data
        if self.dataset_name.startswith("PBMC"):
            self.species = "human"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                # if data file does not exist, download, preprocess & save it
                print('Preprocessing dataset...')
                if self.dataset_name == "PBMC_10K":
                    adata = sc.read(self.data_dir + 'pbmc10k.h5ad')
                elif self.dataset_name == "PBMC_20K":
                    adata = sc.read(self.data_dir +  'pbmc20k.h5ad')
                else:
                    adata = sc.read(self.data_dir +  'pbmc30k.h5ad')
                    adata.obs["celltype"] = adata.obs["CellType"].astype("category")
                    adata.var["gene_name"] = adata.var.index.tolist()
                    adata.obs['batch'] = [int(name.split("pbmc")[-1]) for name in adata.obs['Experiment'].tolist()]
                # adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
                # adata.var = adata.var.set_index("gene_symbols")
                # adata.var["gene_name"] = adata.var.index.tolist()
                
                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True)
                self.adata.write(self.save_path, compression='gzip')

            self.cell_color = None

        elif self.dataset_name.startswith("TSCA"):    # == "TSCA_lung" or self.dataset_name == "TSCA_spleen" or self.dataset_name == "TSCA_oesophagus":
            self.species = "human"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                # if data file does not exist, download, preprocess & save it
                print('Downloading dataset...')
                if self.dataset_name == "TSCA_lung":
                    adata = sc.read(self.data_dir +'lung.cellxgene.h5ad',
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/lung.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7509735', 'gene_ids-HCATisStab7509736', 'gene_ids-HCATisStab7587202', 'gene_ids-HCATisStab7587205', 'gene_ids-HCATisStab7587208', 'gene_ids-HCATisStab7587211', 'gene_ids-HCATisStab7646032', 'gene_ids-HCATisStab7646033', 'gene_ids-HCATisStab7646034', 'gene_ids-HCATisStab7646035', 'gene_ids-HCATisStab7659968', 'gene_ids-HCATisStab7659969', 'gene_ids-HCATisStab7659970', 'gene_ids-HCATisStab7659971', 'gene_ids-HCATisStab7747197', 'gene_ids-HCATisStab7747198', 'gene_ids-HCATisStab7747199', 'gene_ids-HCATisStab7747200'], axis = 1)
                
                elif self.dataset_name == "TSCA_spleen":
                    adata = sc.read(self.data_dir + 'spleen.cellxgene.h5ad', 
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/spleen.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7463847', 'gene_ids-HCATisStab7463848', 'gene_ids-HCATisStab7463849', 'gene_ids-HCATisStab7587200', 'gene_ids-HCATisStab7587203', 'gene_ids-HCATisStab7587206', 'gene_ids-HCATisStab7587209', 'gene_ids-HCATisStabAug177078016', 'gene_ids-HCATisStabAug177078017', 'gene_ids-HCATisStabAug177078018', 'gene_ids-HCATisStabAug177078019', 'gene_ids-HCATisStabAug177276391', 'gene_ids-HCATisStabAug177276392', 'gene_ids-HCATisStabAug177276394', 'gene_ids-HCATisStabAug177376561', 'gene_ids-HCATisStabAug177376563', 'gene_ids-HCATisStabAug177376565', 'gene_ids-HCATisStabAug177376567'], axis = 1)
                else:
                    # "TSCA_oesophagus"
                    adata = sc.read(self.data_dir + 'oesophagus.cellxgene.h5ad', 
                                    backup_url="https://cellgeni.cog.sanger.ac.uk/tissue-stability/oesophagus.cellxgene.h5ad", )
                    adata.var = adata.var.drop(['gene_ids-HCATisStab7413620', 'gene_ids-HCATisStab7413621', 'gene_ids-HCATisStab7413622', 'gene_ids-HCATisStab7587201', 'gene_ids-HCATisStab7587204', 'gene_ids-HCATisStab7587207', 'gene_ids-HCATisStab7587210', 'gene_ids-HCATisStab7619064', 'gene_ids-HCATisStab7619065', 'gene_ids-HCATisStab7619066', 'gene_ids-HCATisStab7619067', 'gene_ids-HCATisStab7646028', 'gene_ids-HCATisStab7646029', 'gene_ids-HCATisStab7646030', 'gene_ids-HCATisStab7646031', 'gene_ids-HCATisStabAug177184858', 'gene_ids-HCATisStabAug177184862', 'gene_ids-HCATisStabAug177184863', 'gene_ids-HCATisStabAug177376562', 'gene_ids-HCATisStabAug177376564', 'gene_ids-HCATisStabAug177376566', 'gene_ids-HCATisStabAug177376568'], axis = 1)
                

                adata.obs["celltype"] = adata.obs['Celltypes_updated_July_2020'].astype("category")
                adata.var["gene_name"] = adata.var.index.tolist()
                adata.obs = adata.obs.drop(columns = ['Donor', 'Time', 'donor_time', 'Celltypes_GenomeBiol_2019', 'Celltypes_updated_July_2020'])

                # recover raw counts according to obs['n_counts]
                raw = np.expm1(adata.raw.X.toarray())
                raw = np.round(raw)
                new_adata = anndata.AnnData(raw, obs = adata.obs, var = adata.var)
                new_adata.uns = adata.uns.copy()
                print("Recovered counts:", sum(np.sum(raw, axis = 1) == new_adata.obs['n_counts']))

                self.adata = self._preprocess(new_adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True)
                self.adata.write(self.save_path, compression='gzip')
            # get color for umap
            # self.cell_color = self.adata.uns["Celltypes_updated_July_2020_colors"]
            self.cell_color = None
        
        elif self.dataset_name == "RGC":
            self.species = "mouse"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                adata = sc.read(self.data_dir + 'rgc.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True)
                self.adata.write(self.save_path, compression='gzip')
            self.cell_color = None
        
        elif self.dataset_name.endswith("ONC"):
            self.species = "mouse"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                adata = sc.read(self.data_dir + self.dataset_name.lower() + '.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True)
                self.adata.write(self.save_path, compression='gzip')
            self.cell_color = None

        
        elif self.dataset_name == "SM3":
            self.species = "human"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                adata = sc.read(self.data_dir + 'SM3_full.h5ad').T

                annotation = pd.read_csv(self.data_dir + 'PBMCs.allruns.barcode_annotation.txt', sep="\t")
                adata.obs = annotation
                adata = adata[annotation['QC_status'] == 'QCpass']

                adata.obs['celltype'] = adata.obs['celltype_lvl2_inex_10khvg_reads_res08_new'].tolist()
                adata.var["gene_name"] = adata.var.index.tolist()

                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True)
                self.adata.write(self.save_path, compression='gzip')
            self.cell_color = None

        elif self.dataset_name == "GCA":
            self.species = "human"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                # if data file does not exist, download, preprocess & save it
                adata = sc.read(self.data_dir +'gutcellatlas.h5ad',
                                backup_url="https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Full_obj_raw_counts_nosoupx_v2.h5ad",
                                )
                adata.obs["celltype"] = adata.obs['category'].astype("category")
                adata.var["gene_name"] = adata.var.index.tolist()

                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True,
                    top_n_genes = self.topngene)
                
                self.adata.write(self.save_path, compression = 'gzip')
            self.cell_color = None
        
        elif self.dataset_name == "TS_spleen":
            self.species = "human"
            if os.path.exists(self.save_path):
                print('Loading dataset...')
                self.adata = sc.read(self.save_path)
            else:
                adata = sc.read(self.data_dir + 'new_spleen.h5ad')
                self.adata = self._preprocess(adata,
                    filter_gene_by_counts = 500,
                    filter_cell_by_counts = 1000,
                    normalize_total = 1e4,
                    log1p = True,
                    )
                self.adata.write(self.save_path, compression='gzip')
            self.cell_color = None

        else:
            raise ValueError("Dataset not found!")
        

        print(self.adata)
        # self.adata.uns['log1p']["base"] = None
        if self.raw:
            print("\tUsing raw counts as input")
            self.X = self.adata.layers['counts']
        else:
            self.X = self.adata.X
        self.X = np.float32(self.X.toarray() if sparse.isspmatrix(self.X) else self.X)

        self.gene_names = self.adata.var["gene_name"].tolist()
        self.celltypes = self.adata.obs["celltype"].tolist()
        # self.celltypes = self.adata.obs["Curated_annotation"].tolist()

        if 'batch' in self.adata.obs_names:
            self.batch = self.adata.obs['batch'].tolist()
            self.n_batch = np.unique(self.batch)
        else:
            self.n_batch = 1
            self.batch = np.ones(len(self.celltypes))
        
        self.Y, self.cell_encoder = self._label_encoder(self.celltypes)
    

    def all_data(self):
        return (self.X, self.Y, self.gene_names, self.cell_encoder)


    def split_data(self, args):
        """split data into train and test"""
        if self.split == None:
            train_idx, test_idx = train_test_split(range(len(self.Y)),
                                                   test_size = self.test_ratio, 
                                                   shuffle = True)
        elif self.split in self.time_order:
            # split data by ctrl and treatment
            train_idx, test_idx = self._special_train_test(self.time_col, self.split)
        else:
            raise ValueError("split method not found!")
        
        s1 = pd.Series(train_idx, name = 'train_idx')
        s2 = pd.Series(test_idx, name = 'test_idx')
        df = pd.concat([s1, s2], axis = 1)
        save_file(df, args, '_idx.csv')

        train_X = self.X[train_idx]
        test_X = self.X[test_idx]
        train_Y = self.Y[train_idx]
        test_Y = self.Y[test_idx]
        train_b = self.batch[train_idx]
        test_b = self.batch[test_idx]
        return (train_X, test_X, train_Y, test_Y, train_b, test_b)
    

    def _special_train_test(self, obs_name, time_point):
        # Convert time_point to its corresponding categorical code
        train_code = self.adata.obs[obs_name].cat.categories.tolist().index(time_point)
        
        # get indexes of train and test data
        train_indexes = np.where(self.adata.obs[obs_name].cat.codes < train_code)[0]
        test_indexes = np.where(self.adata.obs[obs_name].cat.codes == train_code)[0]
        
        # split test data into 2 parts
        np.random.shuffle(test_indexes)
        split_idx = int(self.test_ratio * len(test_indexes))
        sub_test, sub_train = test_indexes[:split_idx], test_indexes[split_idx:]

        train_indexes = np.concatenate((train_indexes, sub_train))
        return train_indexes, sub_test


    def assign_dataloader(self, X, Y, batch, batch_size):
        dataset = CustomDataset(X, Y, batch)
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
        )
        return data_loader



    def _label_encoder(self, labels):
        """ Generate the numerical labels."""
        encoder = LabelEncoder()
        encoder.fit(labels)
        Y = encoder.transform(labels)
        Y = np.asarray(Y)
        return Y, encoder


    def _preprocess(self, adata,
                    filter_gene_by_counts = 3,  # step 1
                    filter_cell_by_counts = False,  # step 2
                    normalize_total = 1e4,  # 3. whether to normalize the raw data and to what sum
                    log1p = True,  # 4. whether to log1p the normalized data
                    ):
        
        adata = self._remove_by_species(adata)
        
        # step 1: filter genes
        if not filter_gene_by_counts is False:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts = filter_gene_by_counts if isinstance(filter_gene_by_counts, int) else None
            )

        # step 2: filter cells
        if isinstance(type(filter_cell_by_counts), int):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts = filter_cell_by_counts
            )
        
        # select top n genes
        if isinstance(self.topngene, int):
            print("Selecting top %d genes ..." %self.topngene)
            sc.pp.highly_variable_genes(adata, 
                                        n_top_genes = self.topngene,
                                        flavor = 'cell_ranger',
            )
            adata = adata[:, adata.var['highly_variable']]
        
        # Save raw counts
        adata.layers["counts"] = adata.X.copy()

        # step 3: normalize total
        if normalize_total:
            print("Normalizing total counts ...")
            sc.pp.normalize_total(
                adata,
                target_sum = normalize_total
                if isinstance(normalize_total, float) else None)

        # step 4: log1p
        if log1p:
            print("Log1p transforming ...")
            sc.pp.log1p(adata)
        
        return adata


    def _remove_by_species(self, adata):
        """
        Remove some highly expressed gene, such as MT genes and ribosome coding genes
        """
        import re
        print("Removing MT & ribosome coding genes ...")

        names = adata.var.index
        upper =  list(x.upper() for x in names)

        if self.species == 'human':
            rb_gene = names[[bool(re.search('RPL|RPS|MRPS|MRPL', gene)) for gene in upper]].tolist()
            mt_gene = names[[bool(re.search('MT-', gene)) for gene in upper]].tolist()
            prob_gene = names[[bool(re.search('MALAT1|MTRNR', gene)) for gene in upper]].tolist()
        elif self.species == 'mouse':
            rb_gene = names[[bool(re.search('RPL|RPS|MRPS|MRPL', gene)) for gene in upper]].tolist()
            mt_gene = names[[bool(re.search('MT-', gene)) for gene in upper]].tolist()
            prob_gene = names[[bool(re.search('MALAT1', gene)) for gene in upper]].tolist()
        else:
            raise ValueError('Species must be human or mouse!')
            
        mask = ~names.isin(rb_gene + mt_gene + prob_gene)
        return adata[:, mask]
    

    def sort_obs(self, obs_name, time_order):
        # sort adata.obs by time points
        self.adata.obs[obs_name] = pd.Categorical(self.adata.obs[obs_name], 
                                                  categories = time_order, 
                                                  ordered = True)
        self.adata.obs.sort_values(obs_name, inplace = True)


    def type_specific_mean(self):
        celltype_mean = {}
        for c in self.cell_encoder.classes_:
            _sub_expre = np.mean(self.adata[self.adata.obs['celltype'] == c].X, axis = 0)
            celltype_mean[c] = _sub_expre
        celltype_mean = torch.tensor(np.array(list(celltype_mean.values())))
        return celltype_mean
    

    def gene_subset(self, model_dataname):
        """
        load the pretrained model genes and resize the data
        """
        # load model used gene_names
        fpath = self.data_dir + model_dataname + '.h5ad'
        print("load gene names from:", fpath)
        model_genes = load_var_names(fpath)
        print("number of genes used in loaded model: ", len(model_genes))

        # get shared genes
        import collections
        share_genes = list((collections.Counter(model_genes) & collections.Counter(self.adata.var.index)).elements())
        print("\tShared genes in loaded model: %.2f%%"%(len(share_genes)/len(model_genes)*100))
        print("\tShared genes in new dataset: %.2f%%"%(len(share_genes)/len(self.gene_names)*100))
        gene_mask = [True if i in share_genes else False for i in model_genes]
        var_indices = [self.adata.var_names.get_loc(gene) for gene in share_genes]

        # build new data object
        test_X = torch.zeros(self.adata.X.shape[0], len(model_genes))
        test_X[:, np.where(gene_mask)[0]] = torch.Tensor(self.X[:, var_indices])

        adata = anndata.AnnData(X = test_X.numpy())
        adata.obs = self.adata.obs.copy()
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata.var['gene_name']  = model_genes
        adata.var.index = adata.var['gene_name']

        self.adata = adata
        self.X = test_X
        self.gene_names = self.adata.var["gene_name"]

        return self.X



class CustomDataset(Dataset):
    def __init__(self, x, y, batch):
        # self.x = torch.Tensor(x)
        # self.y = torch.LongTensor(y)
        # self.batch = torch.LongTensor(batch)
        self.x = x
        self.y = y
        self.batch = batch

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.batch[index]
