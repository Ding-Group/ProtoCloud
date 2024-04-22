import os
import pickle 
import math
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report
import joblib

import src.glo as glo
EPS = glo.get_value('EPS')


### Settings
#######################################################
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def seed_torch(device, seed = 7):
    """
    Sets Seed for reproducible experiments.
    """
    print("Global seed set to {}".format(seed))
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model_dict(model_dict, model_path):
    with open(model_path + 'model_dict.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    print("model dict saved")


def load_model_dict(model_path, device="cpu"):
    with open(os.path.join(model_path, 'model_dict.pkl'), 'rb') as f:
        state_dict = pickle.load(f)
    # path = os.path.join(model_path, 'model_dict.pkl')
    # state_dict = torch.load(path, map_location=device)
    return state_dict



### data processing
#######################################################
def one_hot_encoder(target, n_cls):
    assert torch.max(target).item() <= n_cls

    target = target.view(-1, 1)
    onehot = torch.zeros(target.size(0), n_cls)
    onehot = onehot.to(target.device)
    onehot.scatter_(1, target.long(), 1)

    return onehot


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


def data_info_saver(info, model_dir, file_name, **kwargs):
    # save model-relate data info for reuse
    if file_name == 'cell_encoder':   # save label encoder
        joblib.dump(info, os.path.join(model_dir, 'cell_encoder.joblib'))
    elif file_name == 'gene_names':
        with open(os.path.join(model_dir, 'gene_names.txt'), 'w') as f:
            for gene in info:
                f.write(f"{gene}\n")
    else:
        raise NotImplementedError()


def data_info_loader(file_name, model_dir):
    if file_name == 'cell_encoder':     # load LabelEncoder
        return joblib.load(os.path.join(model_dir, 'cell_encoder.joblib'))
    elif file_name == 'gene_names':
        with open(os.path.join(model_dir, 'gene_names.txt'), 'r') as f:
            gene_names = [line.strip() for line in f.readlines()]
        return gene_names
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



### Model Settings
#######################################################
def get_custom_exp_code(model_name, lr, epochs, batch_size, 
                        dataset_name, topngene, split, **kwargs):
    '''
    Creates Experiment Code from argparse + Folder Name to Save Results
    model_lr_epochs_batchsize_dataset
    '''
    param_code = model_name
    # ### Model Type
    # param_code += '_p%s'%str(kwargs[prototypes_per_class])
    # param_code += '_l%s'%str(kwargs[latent_dim])
    # Learning Rate
    param_code += '_lr%s' %format(lr, '.0e')
    param_code += '_e%s' %format(epochs, 'd')
    # Batch Size
    param_code += '_b%s' % str(batch_size)

    ### dataset
    if topngene != None:
        param_code += '_top%s' % str(topngene)
    if split != None:
        param_code += '_%s' % str(split)
    param_code += '_%s' % dataset_name

    return param_code



### Model Assistance
#######################################################
def log_likelihood_nb(x, mu, theta, eps = EPS):
    # theta should be 1 / r, here theta = r
    log_mu_theta = torch.log(mu + theta + eps)

    ll = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1) \
        + theta * (torch.log(theta + eps) - log_mu_theta) \
        + x * (torch.log(mu + eps) - log_mu_theta)

    del log_mu_theta
    torch.cuda.empty_cache()

    return ll


def log_likelihood_normal(x, mu, logvar):
    var = torch.exp(logvar)
    ll = -0.5 * len(x) * torch.log(2 * math.pi * var) \
                        - torch.sum((x - mu) ** 2) / (2 * var)
    return ll




### Results
#######################################################
def save_model(model, model_dir, exp_code, accu=None, log=print):
    # save model to model_dir with name exp_code + accu
    save_path = os.path.join(model_dir, (exp_code + '.pth'))
    if accu != None:
        with open(os.path.join(model_dir,'saved_model.txt'), 'w') as f:
            f.write('epoch:' + exp_code + ", acc:{0:.4f}".format(accu))

    print('\tSaving model to {0}...'.format(save_path))
    torch.save(model.state_dict(), save_path)
    print('Model saved')


def save_model_w_condition(model, model_dir, exp_code, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    save_path = os.path.join(model_dir, exp_code + '.pth')
    if accu > target_accu:
        log('\tAccuracy above {0:.2f}%'.format(target_accu * 100))
        print('Saving model to {0}...'.format(save_path))
        with open(os.path.join(model_dir,'saved_model.txt'), 'w') as f:
            f.write('Setting:' + exp_code + ", acc:{0:.4f}".format(accu))
        torch.save(model.state_dict(), save_path)
        print('Model saved')

    else:
        print('\tAccuracy below {0:.2f}%, model not saved'.format(target_accu * 100))


def save_prototype_cells(model, result_dir, exp_code):
    prototype_cells = model.get_prototype_cells().detach().cpu().numpy()
    prototype_cells = (prototype_cells + 1) / 2.0     # scale to [0,1]
    protoCell_dir = os.path.join(result_dir, exp_code + '_protoCells.npy')

    np.save(protoCell_dir, prototype_cells)
    print("\tPrototypes saved")


def print_results(epoch, acc, loss=None, recon=None, kl=None, ce=None, ortho=None, atomic=None, is_train=True):
    if is_train:
        print('Train epoch: {0}'.format(epoch),
            '\taccu: {0}%'.format(acc * 100),
            '\tloss: {0}'.format(loss),
            '\trecons: {0}'.format(recon),
            '\tKL: {0}'.format(kl),
            '\tcross ent: {0}'.format(ce),
            '\tortho: {0}'.format(ortho),
            '\tatomic: {0}'.format(atomic),
            )
    else:
        print('Valid epoch: {0}'.format(epoch),
            '\taccu: {0}%'.format(acc * 100)
            )


def model_metrics(predicted):
    """
    Evaluate the prediction results. 
    Returns the error rate, a testing report, and a confusion matrix of the results.
    Args:
        predicted: (pd.DataFrame) DataFrame containing actual and predicted labels
                col_names: 'celltype', 'prob1', 'prob2', 'idx1', 'idx2'
    """
    orig_y = predicted['celltype']
    pred_y = predicted['idx1']

    rep = classification_report(orig_y, pred_y, output_dict = True)

    unique_elements = np.unique(np.concatenate((orig_y, pred_y)))
    cm = confusion_matrix(orig_y, pred_y, labels=unique_elements)
    # cm = confusion_matrix(orig_y, pred_y)
    if cm is None:
        raise Exception("Some error in generating confusion matrix")
    misclass_rate = 1 - accuracy_score(orig_y, pred_y)
    
    return misclass_rate, rep, cm



def save_file(results, save_dir=None, exp_code=None, file_ending=None, save_path=None, **kwargs):
    # args.results_dir, args.exp_code
    """
    Save the results to save dir or save path
    Args:
        results:
            (AnnData) AnnData object containing actual and predicted labels
                adata.obs: 'celltype', 'predicted_labels'
            OR (pd.DataFrame) DataFrame containing actual and predicted labels
                col_names: 'celltype', 'predicted_labels'
            OR (tuple) (misclass_rate, rep, cm)
    """
    if save_path is None:
        save_path = save_dir + exp_code + file_ending

    if isinstance(results, anndata.AnnData):
        results.write(save_path)
    elif isinstance(results, pd.DataFrame):
        results.to_csv(save_path)
    elif isinstance(results, np.ndarray):
        np.save(save_path, results)
    elif isinstance(results, tuple):
        np.save(save_path, results)
    else:
        raise NotImplementedError("Data type not supported")
    # print("Saved results")


def load_file(results_dir, exp_code=None, file_ending=None, path=None, **kwargs):
    if path is None:
        file_path = os.path.join(results_dir, exp_code + file_ending)
    else:
        file_path = path
    
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle = True)
        elif file_path.endswith('.h5ad'):
            return sc.read(file_path)
        else:
            raise NotImplementedError("File format not supported")
    except:
        raise FileNotFoundError(file_path + " not found")


### Plot
#######################################################
def mutual_genes(list1, list2, mutually_exclusive=True):
    """
    Get non-overlapping list with order retained
    """
    if mutually_exclusive:
        ul1 = [i for i in list1 if i not in list2]
        ul2 = [i for i in list2 if i not in list1]
        return ul1 + ul2
    else:
        ul2 = [i for i in list2 if i not in list1]
        return list1 + ul2


def minmax_scale_matrix(matrix_np):
    row_mins = matrix_np.min(axis=1, keepdims=True)
    row_maxs = matrix_np.max(axis=1, keepdims=True)
    
    scaled_matrix = (matrix_np - row_mins) / (row_maxs - row_mins)
    return scaled_matrix