import json
import os

import torch
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from torch.distributions import Multinomial
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils import *
import ProtoCloud.glo as glo
EPS = glo.get_value('EPS')

# species = {
#     "PBMC": "human",
#     "TSCA": "human",
#     "RGC": "mouse",
#     "ONC": "mouse",
#     "ICA": "human",
#     "SM3": "human", 
#     "ATAC": "human",
#     "TS_spleen": "human",
# }


class scRNAData():
    """Handler for loading, preprocessing, and splitting single-cell RNA-seq data.

    Manages the full data pipeline from raw .h5ad files to train/test splits
    ready for model training. Supports species-specific gene filtering,
    highly variable gene selection, data balancing via rare cell type
    augmentation, and integration with pretrained model gene spaces.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Used to locate .h5ad files in ``data_dir``.
    data_dir : str
        Directory path containing the .h5ad data files.
    raw : bool
        Whether to use raw counts (``'counts'`` layer) as input.
    topngene : int or None
        Number of top highly variable genes to select. If None, all genes
        are kept.
    preprocess_data : bool, optional
        Whether to preprocess external datasets (filtering, normalization,
        log-transform), by default False.
    species : {'human', 'mouse'}, optional
        Species type for gene filtering, by default 'human'.
    filter_gene_by_counts : int, optional
        Minimum counts threshold for gene filtering, by default 500.
    filter_cell_by_counts : int, optional
        Minimum counts threshold for cell filtering, by default 1000.
    normalize_total : float, optional
        Target sum for library size normalization, by default 1e4.
    log1p : bool, optional
        Whether to apply log1p transformation, by default True.

    Attributes
    ----------
    adata : anndata.AnnData
        The loaded and processed single-cell data object.
    gene_names : numpy.ndarray
        Array of gene names from ``adata.var['gene_name']``.
    celltypes : numpy.ndarray or None
        Cell type annotations from ``adata.obs['celltype']``, or None if
        not available.
    cell_encoder : sklearn.preprocessing.LabelEncoder or None
        Fitted label encoder for cell type labels. Initialized during
        ``split_data`` or loaded from a pretrained model.
        
    """
    def __init__(self, dataset_name, data_dir, raw:bool=1, topngene=None, 
                preprocess_data = False, species='human',
                filter_gene_by_counts=500, filter_cell_by_counts=1000, normalize_total=1e4, log1p=True, 
                **kwargs):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.raw = raw
        self.topngene = topngene

        self.time_order = None
        self.time_col = None
        self.species = species

        save_path = self.data_dir + self.dataset_name + str(self.topngene) + '.h5ad'
        if self.topngene is not None and os.path.exists(save_path):
            # top_n_gene dataset already exists and preprocessed
            print('Loading dataset...')
            adata = sc.read(save_path)
        
        else:
            # load dataset (and preprocess)
            save_path = self.data_dir + self.dataset_name + '.h5ad'

            if os.path.exists(save_path):
                # load un-predefined datasets
                print('Loading outside dataset, this will not process the data')
                adata = sc.read(save_path)
                assert 'gene_name' in adata.var.columns, "The 'gene_name' column is NOT present in adata.var."
                
                if preprocess_data:
                    preprocess_arg = {
                        "filter_gene_by_counts": filter_gene_by_counts,
                        "filter_cell_by_counts": filter_cell_by_counts,
                        "normalize_total": normalize_total,
                        "log1p": log1p
                    }
                    adata = self._preprocess(adata, **preprocess_arg)

                self.cell_color = None

            else:
                print(save_path)
                raise ValueError("Dataset not found!")
        
            # select top n genes
            # self.adata.uns['log1p']["base"] = None
            if isinstance(self.topngene, int):
                adata = self._top_n_genes(self.topngene, adata, raw=self.raw)
                top_gene_path = self.data_dir + self.dataset_name + str(self.topngene) + '.h5ad'
                adata.write(top_gene_path, compression='gzip')
        #######################################################
        adata.var_names_make_unique()
        self.adata = adata#[~adata.obs['celltype'].isna(), :]
        del adata
        print(self.adata)
        # self.adata.uns['log1p']["base"] = None
        self.gene_names = self.adata.var["gene_name"].values

        # if self.raw and 'counts' in self.adata.layers:
        #     print("\tUsing layer 'counts' as input")
        #     self.X = np.float32(np.array(self.adata.layers['counts'].todense()) if sparse.isspmatrix(self.adata.layers['counts']) else self.adata.layers['counts'])
        # else:
        #     print("\tUsing adata.X as input")
        #     self.X = np.float32(np.array(self.adata.X.todense()) if sparse.isspmatrix(self.adata.X) else self.adata.X)

        if "celltype" in self.adata.obs.columns:
            self.celltypes = self.adata.obs["celltype"].values
        else:
            print(f"\tDidn't find celltype annotation in dataset!")
            self.celltypes = None
        self.cell_encoder = None


    @staticmethod
    def to_dense(adata, raw=True):
        """Convert AnnData expression matrix to a dense numpy array.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object containing gene expression data.
        raw : bool, optional
            If True and a ``'counts'`` layer exists, use raw counts;
            otherwise use ``adata.X``, by default True.

        Returns
        -------
        numpy.ndarray
            Dense gene expression matrix of shape ``(n_cells, n_genes)``
            with dtype int32.
        """
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
        """Convert AnnData expression matrix to COO sparse format.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object containing gene expression data.
        raw : bool, optional
            If True and a ``'counts'`` layer exists, use raw counts;
            otherwise use ``adata.X``, by default True.

        Returns
        -------
        scipy.sparse.coo_matrix
            Gene expression data in COO sparse format.
        """
        if raw and 'counts' in adata.layers:
            print("\tUsing layer 'counts' as input")
            X = adata.layers['counts']
        else:
            print("\tUsing adata.X as input")
            X = adata.X
        X = all_to_coo(X)
        return X
    

    def get_split_idx(self, test_ratio, new_label = False, 
                      results_dir = None, exp_code = None, index_file = None, pretrain_model_pth = None, **kwargs):
        """Get train/test split indices.

        Supports three modes: loading from an existing index file, using
        predicted labels from a pretrained model, or random splitting.
        The resulting indices are saved to a CSV file in `result_dir`.

        Parameters
        ----------
        new_label : bool
            If True, use predicted labels from a pretrained model to
            determine the split (high-certainty as train, rest as test).
        test_ratio : float
            Fraction of data to use as test set (0.0 to 1.0).
        results_dir : str
            Directory to save the split index file.
        exp_code : str
            Experiment code used for file naming.
        index_file : str, optional
            Path to an existing index file to load splits from,
            by default None.
        pretrain_model_pth : str, optional
            Path to a pretrained model checkpoint, required when
            ``new_label=True``, by default None.


        Returns
        -------
        train_idx : numpy.ndarray
            Integer indices for training samples.
        test_idx : numpy.ndarray
            Integer indices for test samples.
        """
        if exp_code is None:
            exp_code = 'protocloud'
        if test_ratio == 1:
            # all data for test
            return np.array([], dtype=int), np.arange(self.adata.shape[0])
        
        if index_file is not None:
            print("\tUsing existing index from:", index_file)
            indices = load_file(results_dir, path=index_file, **kwargs)
            train_idx = indices['train_idx'].dropna().values.astype(int)
            test_idx = indices['test_idx'].dropna().values.astype(int) 

            return train_idx, test_idx
        
        elif new_label:
            train_idx, test_idx = self.use_pred_label(pretrain_model_pth, results_dir, 
                                                        exp_code, test_ratio, **kwargs)
        else:
            if test_ratio == 0:
                train_idx = np.arange(self.adata.shape[0])
                test_idx = np.array([], dtype=int)
            else:
                train_idx, test_idx = train_test_split(range(self.adata.shape[0]),
                                                    test_size = test_ratio, 
                                                    shuffle = True)

        s1 = pd.Series(train_idx, name = 'train_idx')
        s2 = pd.Series(test_idx, name = 'test_idx')
        df = pd.concat([s1, s2], axis = 1)
        if results_dir is not None and exp_code is not None:
            save_file(df, results_dir, exp_code, '_idx.csv')

        return train_idx, test_idx


    def split_data(self, train_idx, test_idx, 
                   data_balance = True, 
                   model_mode = "train", **kwargs):
        """Split data into training and test sets.

        Converts expression data to dense format, encodes training labels
        with ``LabelEncoder``, and augments rare cell types.

        Parameters
        ----------
        train_idx : array-like
            Indices of training samples.
        test_idx : array-like
            Indices of test samples.
        data_balance : bool, optional
            Whether to augment rare cell types in training data,
            by default True.
        model_mode : {'train', 'test'}, optional
            Model mode; balancing is only applied when ``'train'``,
            by default ``'train'``.

        Returns
        -------
        train_X : numpy.ndarray
            Training expression matrix of shape ``(n_train, n_genes)``.
        test_X : numpy.ndarray
            Test expression matrix of shape ``(n_test, n_genes)``.
        train_Y : numpy.ndarray
            Encoded numeric training labels (0-indexed).
        test_Y : numpy.ndarray
            Original string cell type labels for the test set.
        """
        X = self.to_dense(self.adata, raw=self.raw)
        # X = self.to_sparse_tensor(self.adata, raw=self.raw)

        # all data for test
        if len(train_idx) == 0:
            return (None, X, None, self.celltypes)
        # elif len(test_idx) == 0:
        #     return (X, None, self.celltypes, None)

        test_X = X[test_idx, :]
        test_Y = self.celltypes[test_idx]

        train_X = X[train_idx, :]
        train_Y = self.celltypes[train_idx]

        # if training new model
        if self.cell_encoder is None:
            _, self.cell_encoder = self._label_encoder(np.unique(train_Y))
        train_Y = self.cell_encoder.transform(train_Y)


        if data_balance and model_mode == "train":
            train_X, train_Y = self.augment_rares(train_X, train_Y, self.cell_encoder)
        # for c in np.unique(train_Y):
        #     portion = np.sum(train_Y == c) / train_Y.shape[0]
        #     print(self.cell_encoder.inverse_transform([c]), "%.3f"%portion)

        return (train_X, test_X, train_Y, test_Y)


    @staticmethod
    def augment_rares(X, Y, cell_encoder=None):
        """Oversample rare cell types via multinomial sampling.

        Cell types with fewer cells than ``1 / (2 * num_of_celltypes)`` of the
        total are considered rare.

        Parameters
        ----------
        X : numpy.ndarray
            Gene expression matrix of shape ``(n_cells, n_genes)``.
        Y : numpy.ndarray
            Numeric cell type labels of shape ``(n_cells,)``.
        cell_encoder : sklearn.preprocessing.LabelEncoder or None, optional
            Fitted label encoder, used only to print the names of the rare
            cell types. Names are omitted if None, by default None.

        Returns
        -------
        augmented_X : numpy.ndarray
            Expression matrix with synthetic samples appended.
        augmented_Y : numpy.ndarray
            Labels including synthetic samples.
        """
        print("\tAugmenting rare cell types")
        min_p = 1 / len(np.unique(Y)) / 2
        min_type_num = int(min_p * X.shape[0])

        rares = []
        num_sample = 0
        for c in np.unique(Y):
            num_cells = np.sum(Y == c)
            if num_cells < min_type_num:
                num_sample += min_type_num - num_cells
                rares.append(c)
        print("Rare cell types:", len(rares))
        if cell_encoder is not None:
            print("\tRare cell types:", [cell_encoder.inverse_transform([c])[0] for c in rares])
        print(f"\tTotal samples to add: {num_sample}")
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
                    filter_gene_by_counts = False,
                    filter_cell_by_counts = False,
                    normalize_total = 1e4,
                    log1p = True,
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
        
        return adata
    

    @staticmethod
    def _top_n_genes(topngene, adata, raw=True):
        print("Selecting top %d genes ..." %topngene)
        sc.pp.highly_variable_genes(adata, 
                                    n_top_genes = topngene,
                                    layer = "counts" if raw else None,
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
            print(f'Species not found: Species must be human or mouse, got {self.species}')
            return adata
            
        mask = ~names.isin(rb_gene + mt_gene + prob_gene)
        return adata[:, mask]
    

    def gene_subset(self, pretrain_model_pth, **kwargs):
        """Resize the dataset to match a pretrained model's gene space.

        Finds the intersection of genes between the current dataset and
        the pretrained model, creates a new AnnData object aligned
        to the model's gene ordering.

        Parameters
        ----------
        pretrain_model_pth : str
            Path to the pretrained model checkpoint file.

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
        self.gene_names = self.adata.var["gene_name"].values
        self.cell_encoder = data_info_loader('cell_encoder', os.path.dirname(pretrain_model_pth))


    def use_pred_label(self, pretrain_model_pth, results_dir, exp_code=None,
                        test_ratio=None, prob_mask=True, **kwargs):
        """Use pretrained model predictions to define train/test splits.

        High-certainty predictions become the training set; uncertain
        predictions become the test set. Replaces ``self.celltypes``
        with the predicted labels.

        Parameters
        ----------
        pretrain_model_pth : str
            Path to the pretrained model checkpoint file.
        results_dir : str
            Directory containing prediction result files.
        exp_code : str
            Experiment code for file naming.
        test_ratio : float
            Unused; kept for API compatibility.
        prob_mask : bool, optional
            If True, filter by prediction certainty, by default True.


        Returns
        -------
        train_idx : numpy.ndarray
            Indices of high-certainty (certain) predictions.
        test_idx : numpy.ndarray
            Indices of uncertain predictions.

        Raises
        ------
        FileNotFoundError
            If the prediction CSV file is not found.
        """
        assert pretrain_model_pth is not None
        if exp_code is None:
            exp_code = 'protocloud'
        try:
            model_exp_code = os.path.basename(pretrain_model_pth)[:-4]
            model_exp_code = "_".join(model_exp_code.split("_")[:4] + [self.dataset_name])
            path = os.path.join(results_dir, model_exp_code + '_pred.csv')
            predicted = load_file(results_dir, exp_code, path=path)
        except FileNotFoundError:
            raise FileNotFoundError("Apply the model for prediction first")
        
        # train: only use label with assigned and prob >= threshold
        if prob_mask:
            prob_mask = predicted['certainty'] == "certain"

        self.cell_encoder = data_info_loader('cell_encoder', os.path.dirname(pretrain_model_pth))
        self.celltypes = predicted['pred1'].values
        print(self.cell_encoder.classes_)
        print("Using predicted label from pretrained model")
        
        train_idx = np.where(prob_mask)[0]
        test_idx = np.where(~prob_mask)[0]

        print(len(train_idx), max(train_idx), train_idx)
        print(len(test_idx), max(test_idx), test_idx)

        return train_idx, test_idx 


    @staticmethod
    def _resize_and_fill(orig, new_shape, gene_mask, var_indices):
        print(new_shape, len(gene_mask), len(var_indices))
        new_matrix = csr_matrix(new_shape, dtype=orig.dtype)
        new_matrix[:, np.where(gene_mask)[0]] = orig[:, var_indices]

        return new_matrix


class CustomDataset(Dataset):
    def __init__(self, x, y, batch):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
        self.batch = torch.LongTensor(batch)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.batch[index]


