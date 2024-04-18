import json
import os

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import *
import src.glo as glo
EPS = glo.get_value('EPS')

species = {
    "PBMC": "human",
    "TSCA": "human",
    "RGC": "mouse",
    "ONC": "mouse",
    "ICA": "human",
    "SM3": "human", 
    "ATAC": "human",
    "TS_spleen": "human",
}


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
        self.species = None

        if self.topngene is not None:
            save_path = self.data_dir + self.dataset_name + str(self.topngene) + '.h5ad'
        else:
            save_path = self.data_dir + self.dataset_name + '.h5ad'

        preprocess_arg = {
            "filter_gene_by_counts": 500,
            "filter_cell_by_counts": 1000,
            "normalize_total": 1e4,
            "log1p": True
        }
        # Loading and preparing data
        if self.dataset_name.startswith("PBMC"):
            self.species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                # if data file does not exist, download, preprocess & save it
                print('Preprocessing dataset...')
                if self.dataset_name == "PBMC_10K":
                    adata = sc.read(self.data_dir + 'pbmc10k.h5ad')
                elif self.dataset_name == "PBMC_20K":
                    adata = sc.read(self.data_dir +  'pbmc20k.h5ad')
                elif self.dataset_name == "PBMC_30K":
                    adata = sc.read(self.data_dir +  'pbmc30k.h5ad')
                    adata.obs["celltype"] = adata.obs["CellType"].astype("category")
                    adata.var["gene_name"] = adata.var.index.tolist()
                    adata.obs['batch'] = [int(name.split("pbmc")[-1]) for name in adata.obs['Experiment'].tolist()]                
                
                elif self.dataset_name == "PBMC1":
                    adata = sc.read(self.data_dir +  'pbmc1.h5ad')
                    adata.obs["celltype"] = adata.obs["CellType"].astype("category")
                    adata.var["gene_name"] = adata.var.index.tolist()
                    preprocess_arg['filter_gene_by_counts'] = False
                    preprocess_arg['filter_cell_by_counts'] = False
                elif self.dataset_name == "PBMC2":
                    adata = sc.read(self.data_dir +  'pbmc2.h5ad')
                    adata.obs["celltype"] = adata.obs["CellType"].astype("category")
                    adata.var["gene_name"] = adata.var.index.tolist()
                    preprocess_arg['filter_gene_by_counts'] = False
                    preprocess_arg['filter_cell_by_counts'] = False
                else:
                    raise ValueError("Dataset not found!")
                # adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
                # adata.var = adata.var.set_index("gene_symbols")
                # adata.var["gene_name"] = adata.var.index.tolist()
                
                adata = self._preprocess(adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')

            self.cell_color = None

        elif self.dataset_name.startswith("TSCA"):    # == "TSCA_lung" or self.dataset_name == "TSCA_spleen" or self.dataset_name == "TSCA_oesophagus":
            self.species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
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

                adata = self._preprocess(new_adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')
            # get color for umap
            # self.cell_color = adata.uns["Celltypes_updated_July_2020_colors"]
            self.cell_color = None
        
        elif self.dataset_name == "RGC":
            self.species = "mouse"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(self.data_dir + 'rgc_atlas.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

                adata = self._preprocess(adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')
            self.cell_color = None
        
        elif self.dataset_name.endswith("ONC"):
            self.species = "mouse"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(self.data_dir + self.dataset_name.lower() + '.h5ad')
                adata.obs["celltype"] = adata.obs['Type']
                adata.var["gene_name"] = adata.var.index
                adata.obs['batch'] = [int(name.split("Batch")[-1]) for name in adata.obs['Batch'].tolist()]

                preprocess_arg['filter_gene_by_counts'] = False
                preprocess_arg['filter_cell_by_counts'] = False
                adata = self._preprocess(adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')
            self.cell_color = None
        
        elif self.dataset_name == "ICA":
            self.species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(self.data_dir + self.dataset_name.lower() + '.h5ad')
                adata.var["gene_name"] = adata.var.index
                
                adata = self._preprocess(adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')
            self.cell_color = None
        
        # elif self.dataset_name == "SM3":
        #     self.species = "human"
        #     if os.path.exists(save_path):
        #         print('Loading dataset...')
        #         adata = sc.read(save_path)
        #     else:
        #         adata = sc.read(self.data_dir + 'SM3_full.h5ad').T

        #         annotation = pd.read_csv(self.data_dir + 'PBMCs.allruns.barcode_annotation.txt', sep="\t")
        #         adata.obs = annotation
        #         adata = adata[annotation['QC_status'] == 'QCpass']

        #         adata.obs['celltype'] = adata.obs['celltype_lvl2_inex_10khvg_reads_res08_new'].tolist()
        #         adata.var["gene_name"] = adata.var.index.tolist()

        #         adata = self._preprocess(adata, **preprocess_arg)
        #         adata.write(save_path, compression='gzip')
        #     self.cell_color = None

        # elif self.dataset_name == "GCA":
        #     self.species = "human"
        #     if os.path.exists(save_path):
        #         print('Loading dataset...')
        #         adata = sc.read(save_path)
        #     else:
        #         # if data file does not exist, download, preprocess & save it
        #         adata = sc.read(self.data_dir +'gutcellatlas.h5ad',
        #                         backup_url="https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Full_obj_raw_counts_nosoupx_v2.h5ad",
        #                         )
        #         adata.obs["celltype"] = adata.obs['category'].astype("category")
        #         adata.var["gene_name"] = adata.var.index.tolist()

        #         adata = self._preprocess(adata,
        #             filter_gene_by_counts = 500,
        #             filter_cell_by_counts = 1000,
        #             normalize_total = 1e4,
        #             log1p = True,
        #             top_n_genes = self.topngene)
                
        #         adata.write(save_path, compression = 'gzip')
        #     self.cell_color = None
        
        elif self.dataset_name == "TS_spleen":
            self.species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                adata = sc.read(self.data_dir + 'new_spleen.h5ad')
                adata = self._preprocess(adata, **preprocess_arg)
                adata.write(save_path, compression='gzip')
            self.cell_color = None

        elif self.dataset_name.startswith("ATAC"):
            self.species = "human"
            if os.path.exists(save_path):
                print('Loading dataset...')
                adata = sc.read(save_path)
            else:
                print('Preprocessing dataset...')
                adata = sc.read(self.data_dir + self.dataset_name + '.h5ad')

                # only for top_n_genes, assume X is raw
                adata = self._preprocess(adata,
                        filter_gene_by_counts = False,
                        filter_cell_by_counts = False,
                        normalize_total = 1e4,
                        log1p = True)
                adata.X = adata.layers['counts']
                del adata.layers
                # adata.write(save_path, compression='gzip')
        
        elif os.path.exists(self.data_dir + self.dataset_name + '.h5ad'):
            # load un-predefined datasets
            print('Loading outside dataset, this will not process the data')

            if self.topngene is not None:
                top_gene_path = self.data_dir + self.dataset_name + str(self.topngene) + '.h5ad'
                if os.path.exists(top_gene_path):
                    adata = sc.read(top_gene_path)
                else:
                    adata = sc.read(self.data_dir + self.dataset_name + '.h5ad')
                    adata = self._top_n_genes(adata)
                    adata.write(top_gene_path, compression='gzip')
            else:            
                adata = sc.read(save_path)

            # assert 'celltype' in adata.obs.columns, "The 'celltype' column is NOT present in adata.obs."
            assert 'gene_name' in adata.var.columns, "The 'gene_name' column is NOT present in adata.var."

            self.cell_color = None

        else:
            print(save_path)
            raise ValueError("Dataset not found!")
        

        #######################################################
        adata.var_names_make_unique()
        self.adata = adata#[~adata.obs['celltype'].isna(), :]
        del adata
        print(self.adata)
        # self.adata.uns['log1p']["base"] = None
        self.gene_names = self.adata.var["gene_name"].tolist()

        if self.raw and 'counts' in self.adata.layers:
            print("\tUsing layer 'counts' as input")
            self.X = np.float32(np.array(self.adata.layers['counts'].todense()) if sparse.isspmatrix(self.adata.layers['counts']) else self.adata.layers['counts'])
        else:
            print("\tUsing adata.X as input")
            self.X = np.float32(np.array(self.adata.X.todense()) if sparse.isspmatrix(self.adata.X) else self.adata.X)

        if "celltype" in self.adata.obs.columns:
            celltypes = self.adata.obs["celltype"]
            _, self.cell_encoder = self._label_encoder(np.unique(celltypes))
            self.Y = self.cell_encoder.transform(celltypes)
        else:
            print(f"\tDidn't find celltype annotation in dataset!")
            self.Y = None

        # if 'batch' in self.adata.obs_names:
        #     self.batch = self.adata.obs['batch'].tolist()
        #     self.n_batch = np.unique(self.batch)
        # else:
        #     self.n_batch = 1
        #     self.batch = np.ones(len(self.Y))


    def split_data(self, args):
        """split data into train and test"""
        if args.new_label:
            train_idx, test_idx = self.use_pred_label(args)

        else:
            if self.split == None:
                train_idx, test_idx = train_test_split(range(len(self.Y)),
                                                    test_size = self.test_ratio, 
                                                    shuffle = True)
            # elif self.split in self.time_order:
            #     # split data by ctrl and treatment
            #     train_idx, test_idx = self._special_train_test(self.time_col, self.split)
            else:
                raise ValueError("split method not found!")\
        
        s1 = pd.Series(train_idx, name = 'train_idx')
        s2 = pd.Series(test_idx, name = 'test_idx')
        df = pd.concat([s1, s2], axis = 1)
        save_file(df, args, '_idx.csv')

        train_X = self.X[train_idx]
        test_X = self.X[test_idx]
        train_Y = self.Y[train_idx]
        test_Y = self.Y[test_idx]
        # train_b = self.batch[train_idx]
        # test_b = self.batch[test_idx]

        if args.data_balance and args.model_mode == "train":
            train_X, train_Y = self.augment_rares(train_X, train_Y)
        for c in np.unique(train_Y):
            portion = np.sum(train_Y == c) / train_Y.shape[0]
            print(self.cell_encoder.inverse_transform([c]), "%.3f"%portion)

        return (train_X, test_X, train_Y, test_Y)
                #, train_b, test_b)
    

    # def weighted_sampling(self, train_Y):
    #     print("\tweighted sampling...")
    #     train_Y = torch.from_numpy(train_Y)
    #     class_sample_count = torch.tensor(
    #         [(train_Y == t).sum() for t in torch.unique(train_Y, sorted=True)])

    #     weight = 1. / class_sample_count.float()
    #     weight = 1. / class_sample_count.float()
    #     samples_weight = torch.tensor([weight[y] for y in train_Y])

    #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    #     return sampler


    def augment_rares(self, X, Y):
        print("\tAugmenting rare cell types")
        avg_p = 1 / len(np.unique(Y)) / 2
        # avg_p = 0.06 if avg_p >= 0.1 else avg_p
        min_type_num = int(avg_p * X.shape[0])

        rares = []
        num_sample = 0
        for c in np.unique(Y):
            num_cells = np.sum(Y == c)
            if num_cells < min_type_num:
                num_sample += min_type_num - num_cells
                rares.append(c)
        print("Rare cell types:", len(rares))

        new_X = torch.zeros((X.shape[0] + num_sample, X.shape[1]))
        new_Y = torch.zeros(Y.shape[0] + num_sample, dtype=int)
        new_X[:X.shape[0]] = torch.from_numpy(X)
        new_Y[:Y.shape[0]] = torch.from_numpy(Y)

        start_idx = X.shape[0]
        for c in rares:
            rare = X[Y == c]
            num_sample = min_type_num - np.sum(Y == c)

            rates = torch.from_numpy(rare)
            idx = np.tile(np.arange(rare.shape[0]), num_sample // rare.shape[0] + 1)
            rates = rates[np.random.choice(idx, num_sample, replace=False)]

            from torch.distributions import Multinomial
            for i in range(num_sample):
                rate_i = rates[i, :] + 0.01
                n_i = torch.sum(rate_i)
                p = rate_i.to(torch.float64) / n_i

                u_i = np.random.randint(low=50, high=100)
                new_umi = np.ceil(n_i * u_i / 100).type(torch.int32)
                new_X[start_idx + i] = Multinomial(total_count=int(new_umi), probs=p).sample()

            new_Y[start_idx:start_idx + num_sample] = torch.full((1, num_sample), c)
            start_idx += num_sample

            del rare, rates

        new_X = new_X.numpy()
        new_Y = new_Y.numpy()

        return new_X, new_Y


    # def _special_train_test(self, obs_name, time_point):
    #     # Convert time_point to its corresponding categorical code
    #     train_code = self.adata.obs[obs_name].cat.categories.tolist().index(time_point)
        
    #     # get indexes of train and test data
    #     train_indexes = np.where(self.adata.obs[obs_name].cat.codes < train_code)[0]
    #     test_indexes = np.where(self.adata.obs[obs_name].cat.codes == train_code)[0]
        
    #     # split test data into 2 parts
    #     np.random.shuffle(test_indexes)
    #     split_idx = int(self.test_ratio * len(test_indexes))
    #     sub_test, sub_train = test_indexes[:split_idx], test_indexes[split_idx:]

    #     train_indexes = np.concatenate((train_indexes, sub_train))
    #     return train_indexes, sub_test


    def assign_dataloader(self, X, Y, batch_size, batch=None):
        if batch is not None:
            dataset = CustomDataset(X, Y, batch)
        else:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(X),
                                            torch.LongTensor(Y))

        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size = batch_size,
            shuffle = True,   # sampler mutually exclusive with shuffle
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
                    filter_gene_by_counts = False,  # step 1
                    filter_cell_by_counts = False,  # step 2
                    normalize_total = 1e4,  # 3. whether to normalize the raw data and to what sum
                    log1p = True,  # 4. whether to log1p the normalized data
                    ):
        
        adata = self._remove_by_species(adata)
        
        # step 1: filter genes
        if isinstance(type(filter_gene_by_counts), int):
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts = filter_gene_by_counts
            )

        # step 2: filter cells
        if isinstance(type(filter_cell_by_counts), int):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts = filter_cell_by_counts
            )
        
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

        # select top n genes
        if isinstance(self.topngene, int):
            adata = self._top_n_genes(adata)
        
        return adata
    
    def _top_n_genes(self, adata):
        print("Selecting top %d genes ..." %self.topngene)
        sc.pp.highly_variable_genes(adata, 
                                    n_top_genes = self.topngene,
                                    flavor = 'cell_ranger',
        )
        adata = adata[:, adata.var['highly_variable']]
        return adata


    def _remove_by_species(self, adata):
        """
        Remove some highly expressed gene, such as MT genes and ribosome coding genes
        """
        print("Removing MT & ribosome coding genes ...")
        names = adata.var.index
        upper =  list(x.upper() for x in names)
        
        import re
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
    

    def gene_subset(self, pretrain_model_pth, args):
        """
        load the pretrained model genes and resize the data
        """
        model_dir = os.path.dirname(pretrain_model_pth)
        model_dataname = os.path.basename(pretrain_model_pth)
        model_dataname = "_".join(model_dataname.split("_")[4:])[:-4]

        args.model_data = model_dataname
        print("Use genes as: ", model_dataname)

        # load model used gene_names
        try:
            print("load saved gene names from:", model_dir)
            model_genes = data_info_loader('gene_names', model_dir)
        except FileNotFoundError:
            fpath = self.data_dir + model_dataname + '.h5ad'
            print("load gene names from:", fpath)
            model_genes = load_var_names(fpath)
        print("number of genes in loaded model: ", len(model_genes))

        # get shared genes
        import collections
        share_genes = list((collections.Counter(model_genes) & collections.Counter(self.adata.var.index)).elements())
        print("\tShared genes in loaded model: %.2f%%"%(len(share_genes) / len(model_genes)*100))
        print("\tShared genes in new dataset: %.2f%%"%(len(share_genes) / len(self.gene_names)*100))
        gene_mask = [True if i in share_genes else False for i in model_genes]
        var_indices = [self.adata.var_names.get_loc(gene) for gene in share_genes]
        
        # build new data object     
        test_X = torch.zeros(self.X.shape[0], len(model_genes))
        test_X[:, np.where(gene_mask)[0]] = torch.Tensor(self.X[:, var_indices])
        adata = anndata.AnnData(X = test_X.numpy())
        adata.obs = self.adata.obs.copy()
        adata.var['gene_name']  = model_genes
        adata.var.index = adata.var['gene_name']

        self.adata = adata
        self.X = self.adata.X
        self.gene_names = self.adata.var["gene_name"].tolist()


    def use_pred_label(self, args):
        """
        Use pred label with high certainty as train, rest as test
        """
        assert args.pretrain_model_pth is not None
        try:
            model_exp_code = os.path.basename(args.pretrain_model_pth)[:-4]
            model_exp_code = "_".join(model_exp_code.split("_")[:4] + [self.dataset_name])
            path = os.path.join(args.results_dir, model_exp_code + '_pred.csv')
            predicted = load_file(args, path=path)
        except FileNotFoundError:
            raise FileNotFoundError("Apply the model for prediction first")
        
        # train: only use label with assigned and prob >= threshold
        prob = predicted['prob1'].values
        orig = predicted['celltype'].values
        prob_mask = np.array([(i >= 0.6 and o not in ["Unassigned", "Other"]) 
                            for i, o in zip(prob, orig)])
        # predicted = predicted[prob_mask]

        # self.adata = self.adata[prob_mask, :]
        # self.X = self.adata.X
        self.cell_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
        self.Y = self.cell_encoder.transform(predicted['idx1'].values)
        print(self.cell_encoder.classes_)
        print("Using predicted label from pretrained model")
        
        # balance out the training & testing
        train_r = sum(prob_mask)/len(prob_mask)
        if train_r > (1 - args.test_ratio):
            n = (train_r - (1-args.test_ratio)) * len(prob_mask)
            shuffle = np.random.choice(sum(prob_mask), size = int(n), replace=False)
            train_idx = np.array([i for i in np.where(prob_mask)[0] if i not in shuffle])
            test_idx = np.concatenate((np.where(~prob_mask)[0], np.where(prob_mask)[0][shuffle]))
        else:
            train_idx = np.where(prob_mask)[0]
            test_idx = np.where(~prob_mask)[0]

        return train_idx, test_idx



class CustomDataset(Dataset):
    def __init__(self, x, y, batch):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
        self.batch = torch.LongTensor(batch)
        # self.x = x
        # self.y = y
        # self.batch = batch

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.batch[index]
    # def __getitem__(self, index):
    #     try:
    #         return self.x[index], self.y[index], self.batch[index]
    #     except KeyError as e:
    #         print(f"KeyError: {e}")
    #         print(f"Index {index} does not exist.")
    #         print(f"Length of x: {len(self.x)}, Length of y: {len(self.y)}, Length of batch: {len(self.batch)}")
    #         raise


