import os
import torch
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

### data processing
#######################################################
def load_var_names(filename):
    import h5py
    
    with h5py.File(filename, 'r') as f:
        var_names = [name.decode('utf-8') for name in f['var']['gene_name']]
    return var_names



# def ordered_class(data, args):
#     """
#     Return the ordered class index for the dataset.
#     For RGC: sorted by celltype number
#     Others: sorted by similiar celltypes
#     """
#     # if args.dataset == 'RGC':
#     #     # Sort by extracting the number at the beginning of each string
#     #     sorted_label = sorted(data.cell_encoder.classes_, key=lambda x: int(x.split('_')[0]))
#     #     sorted_index = data.cell_encoder.transform(sorted_label)
#     #     return sorted_index
#     # else:
#     #     return data.cell_encoder.transform(data.cell_encoder.classes_)
#     return data.cell_encoder.transform(data.cell_encoder.classes_)



def info_saver(info, model_dir, file_name, **kwargs):
    # save model-relate data info for reuse
    if file_name == 'cell_encoder':   # save label encoder
        joblib.dump(info, os.path.join(model_dir, 'cell_encoder.joblib'))
    elif file_name == 'gene_names':
        with open(os.path.join(model_dir, 'gene_names.txt'), 'w') as f:
            for gene in info:
                f.write(f"{gene}\n")
    elif file_name == 'cls_threshold':
        info.to_csv(os.path.join(model_dir, 'cls_threshold.csv'))
    else:
        raise NotImplementedError()


def info_loader(file_name, model_dir):
    if file_name == 'cell_encoder':     # load LabelEncoder
        return joblib.load(os.path.join(model_dir, 'cell_encoder.joblib'))
    elif file_name == 'gene_names':
        with open(os.path.join(model_dir, 'gene_names.txt'), 'r') as f:
            gene_names = [line.strip() for line in f.readlines()]
        return gene_names
    elif file_name == 'cls_threshold':
        return pd.read_csv(os.path.join(model_dir, 'cls_threshold.csv'))
    else:
        raise NotImplementedError()


def all_to_coo(X):
    """
    Convert dense numpy array and other sparse matrix format to torch.sparse_coo Tensor
    """
    if not sparse.isspmatrix(X):
        X = sparse.coo_matrix(X)
    elif not sparse.isspmatrix_coo(X):
        X = X.tocoo()
    else:
        pass

    indices = torch.LongTensor([X.row, X.col])
    values = torch.FloatTensor(X.data, dtype=torch.float32)
    shape = torch.Size(X.shape)

    pt_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return pt_tensor
