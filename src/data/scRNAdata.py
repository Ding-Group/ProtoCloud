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
    def __init__(self, adata, raw:bool, topngene:int=None, split=0.9, **kwargs):
        self.raw = raw
        self.topngene = topngene

        self.split = split
        self.time_order = None
        self.time_col = None
        self.species = None

        preprocess_arg = {
            "filter_gene_by_counts": 500,
            "filter_cell_by_counts": 1000,
            "normalize_total": 1e4,
            "log1p": True
        }
        #######################################################
        adata.var_names_make_unique()
        self.adata = adata#[~adata.obs['celltype'].isna(), :]
        del adata
        print(self.adata)
        # self.adata.uns['log1p']["base"] = None
        self.gene_names = self.adata.var["gene_name"].values
        if "celltype" in self.adata.obs.columns:
            self.celltypes = self.adata.obs["celltype"].values
        else:
            print(f"\tDidn't find celltype annotation in dataset!")
            self.celltypes = None
        self.cell_encoder = None

        # if 'batch' in self.adata.obs_names:
        #     self.batch = self.adata.obs['batch'].tolist()
        #     self.n_batch = np.unique(self.batch)
        # else:
        #     self.n_batch = 1
        #     self.batch = np.ones(len(self.Y))


    @staticmethod
    def to_dense(adata, raw=True):
        if raw and 'counts' in adata.layers:
            print("\tUsing layer 'counts' as input")
            X = np.array(adata.layers['counts'].todense(), dtype=np.int32) \
                            if sparse.isspmatrix(adata.layers['counts']) \
                                else adata.layers['counts'].astype(np.int32)
        else:
            print("\tUsing adata.X as input")
            X = np.array(adata.X.todense(), dtype=np.int32) \
                            if sparse.isspmatrix(adata.X) \
                                else adata.X.astype(np.int32)
        return X

    @staticmethod
    def to_sparse_tensor(adata, raw=True):
        if raw and 'counts' in adata.layers:
            print("\tUsing layer 'counts' as input")
            X = adata.layers['counts']
        else:
            print("\tUsing adata.X as input")
            X = adata.X
        X = all_to_coo(X)
        return X
    

    def get_split_idx(self, new_label, test_ratio, 
                      results_dir, exp_code, index_file = None, pretrain_model_pth = None, **kwargs):
        """get split index for training set and test set"""
        if test_ratio == 1:
            # all data for test
            return np.array(), np.array(range(len(self.adata.shape[0])))

        if new_label:
            train_idx, test_idx = self.use_pred_label(pretrain_model_pth, results_dir, 
                                                        exp_code, test_ratio, **kwargs)
        elif index_file is not None:
            print("\tUsing existing index from:", index_file)
            indices = load_file(results_dir, path=index_file, **kwargs)
            train_idx = indices['train_idx'].dropna().values.astype(int) 
            test_idx = indices['test_idx'].dropna().values.astype(int) 
        else:
            if self.split is None:
                train_idx, test_idx = train_test_split(range(self.adata.shape[0]),
                                                    test_size = test_ratio, 
                                                    shuffle = True)
            else:
                raise ValueError("split method not found!")

            s1 = pd.Series(train_idx, name = 'train_idx')
            s2 = pd.Series(test_idx, name = 'test_idx')
            df = pd.concat([s1, s2], axis = 1)
            save_file(df, results_dir, exp_code, '_idx.csv')
        
        return train_idx, test_idx


    def split_data(self, train_idx, test_idx, 
                   data_balance = True, 
                   model_mode = "train", **kwargs):
        """split data into train and test"""
        X = self.to_dense(self.adata, raw=self.raw)
        # X = self.to_sparse_tensor(self.adata, raw=self.raw)

        # all data for test
        if len(train_idx) == 0:
            return (None, X, None, self.celltypes)

        train_X = X[train_idx]
        test_X = X[test_idx]

        train_Y = self.celltypes[train_idx]
        test_Y = self.celltypes[test_idx]

        # if training new model
        if self.cell_encoder is None:
            _, self.cell_encoder = self._label_encoder(np.unique(train_Y))
        train_Y = self.cell_encoder.transform(train_Y)

        # train_b = self.batch[train_idx]
        # test_b = self.batch[test_idx]

        if data_balance and model_mode == "train":
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
    #     samples_weight = torch.tensor([weight[y] for y in train_Y])

    #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    #     return sampler

    @staticmethod
    def augment_rares(X, Y):
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

            # multinomial
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

    @staticmethod
    def assign_dataloader(X, Y, batch_size, batch=None):
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


    @staticmethod
    def _label_encoder(labels):
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
                    raw_layer = "counts"
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
            adata = self._top_n_genes(self.topngene, adata)
        
        return adata
    
    @staticmethod
    def _top_n_genes(topngene, adata):
        print("Selecting top %d genes ..." %topngene)
        sc.pp.highly_variable_genes(adata, 
                                    n_top_genes = topngene,
                                    flavor = 'cell_ranger',
                                    subset=True,
        )
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
    

    # def sort_obs(self, obs_name, time_order):
    #     # sort adata.obs by time points
    #     self.adata.obs[obs_name] = pd.Categorical(self.adata.obs[obs_name], 
    #                                               categories = time_order, 
    #                                               ordered = True)
    #     self.adata.obs.sort_values(obs_name, inplace = True)


    # def type_specific_mean(self):
    #     celltype_mean = {}
    #     for c in self.cell_encoder.classes_:
    #         _sub_expre = np.mean(self.adata[self.adata.obs['celltype'] == c].X, axis = 0)
    #         celltype_mean[c] = _sub_expre
    #     celltype_mean = torch.tensor(np.array(list(celltype_mean.values())))
    #     return celltype_mean
    

    def gene_subset(self, pretrain_model_pth, **kwargs):
        """
        load the pretrained model genes and resize the data
        """
        model_dir = os.path.dirname(pretrain_model_pth)
        model_dataname = os.path.basename(pretrain_model_pth)
        model_dataname = "_".join(model_dataname.split("_")[4:])[:-4]
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
        share_genes = [gene for gene in model_genes if gene in self.gene_names]
        print(len(share_genes), len(np.unique(share_genes)))
        print("\tShared genes in loaded model: %.2f%%"%(len(share_genes) / len(model_genes)*100))
        print("\tShared genes in new dataset: %.2f%%"%(len(share_genes) / len(self.gene_names)*100))
        gene_mask = [True if i in share_genes else False for i in model_genes]
        print(np.sum(gene_mask))
        # var_indices = [self.gene_names.get_loc(gene) for gene in share_genes]
        var_indices = [np.where(self.gene_names == gene)[0][0] \
                        for gene in share_genes if (self.gene_names == gene).any()]

        
        # build new data object
        new_shape = (self.adata.X.shape[0], len(model_genes))
        new_adata = sc.AnnData(csr_matrix(new_shape, dtype=self.adata.X.dtype))
        new_adata.X = self._resize_and_fill(self.adata.X, new_shape, gene_mask, var_indices)
        for layer_name, layer in self.adata.layers.items():
            new_adata.layers[layer_name] = self._resize_and_fill(layer, new_shape, gene_mask, var_indices)
        # test_X = torch.zeros(self.adata.X.shape[0], len(model_genes))
        # test_X[:, np.where(gene_mask)[0]] = torch.Tensor(self.X[:, var_indices])
        # new_adata = anndata.AnnData(X = test_X.numpy())
        new_adata.obs = self.adata.obs.copy()
        new_adata.var['gene_name']  = model_genes
        new_adata.var.index = new_adata.var['gene_name']
        print(new_adata)

        self.adata = new_adata
        self.gene_names = self.adata.var["gene_name"].tolist()
        self.cell_encoder = data_info_loader('cell_encoder', os.path.dirname(pretrain_model_pth))


    def use_pred_label(self, pretrain_model_pth, results_dir, exp_code,
                        test_ratio, prob_mask=True, **kwargs):
        """
        Use pred label with high certainty as train, rest as test
        """
        assert pretrain_model_pth is not None
        try:
            model_exp_code = os.path.basename(pretrain_model_pth)[:-4]
            model_exp_code = "_".join(model_exp_code.split("_")[:4] + [self.dataset_name])
            path = os.path.join(results_dir, model_exp_code + '_pred.csv')
            predicted = load_file(results_dir, exp_code, path=path)
        except FileNotFoundError:
            raise FileNotFoundError("Apply the model for prediction first")
        
        # train: only use label with assigned and prob >= threshold
        if prob_mask:
            prob = predicted['prob1'].values
            orig = predicted['celltype'].values
            prob_mask = np.array([(i >= 0.6 and o not in ["Unassigned", "Other"]) 
                                for i, o in zip(prob, orig)])
            predicted = predicted[prob_mask]
            self.adata = self.adata[prob_mask, :]

        # self.X = self.adata.X
        self.cell_encoder = data_info_loader('cell_encoder', os.path.dirname(pretrain_model_pth))
        # self.celltype = self.cell_encoder.transform(predicted['idx1'].values)
        self.celltypes = predicted['pred1'].values
        print(self.cell_encoder.classes_)
        print("Using predicted label from pretrained model")
        
        # balance out the training & testing
        train_r = sum(prob_mask)/len(prob_mask)
        if train_r > (1 - test_ratio):
            n = (train_r - (1-test_ratio)) * len(prob_mask)
            shuffle = np.random.choice(sum(prob_mask), size = int(n), replace=False)
            train_idx = np.array([i for i in np.where(prob_mask)[0] if i not in shuffle])
            test_idx = np.concatenate((np.where(~prob_mask)[0], np.where(prob_mask)[0][shuffle]))
        else:
            train_idx = np.where(prob_mask)[0]
            test_idx = np.where(~prob_mask)[0]

        return train_idx, test_idx 


    @staticmethod
    def _resize_and_fill(orig, new_shape, gene_mask, var_indices):
        print(new_shape, len(gene_mask), len(var_indices))
        new_matrix = csr_matrix(new_shape, dtype=orig.dtype)
        new_matrix[:, np.where(gene_mask)[0]] = orig[:, var_indices]

        return new_matrix

    # @staticmethod
    # def _resize_and_fill(orig, new_shape, gene_mask, var_indices):
    #     print(new_shape, len(gene_mask), len(var_indices))
        
    #     new_data = np.zeros(new_shape, dtype=orig.dtype)
    #     orig_dense = orig.todense() if sparse.issparse(orig) else orig
    #     new_data[:, np.where(gene_mask)[0]] = orig_dense[:, var_indices]
    #     new_matrix = csr_matrix(new_data)

    #     return new_matrix



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


