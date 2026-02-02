import os
import pickle 
import math
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import NearestNeighbors
from collections import Counter

import ProtoCloud
from ProtoCloud import glo
EPS = glo.get_value('EPS')


### Settings
#######################################################
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)



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
                col_names: 'label', 'prob1', 'prob2', 'pred1', 'pred2'
    """
    orig_y = predicted['label']
    pred_y = predicted['pred1']

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



def process_prediction_file(predicted, model_encoder, label=None, model_dir=None):
    predicted['certainty'] = None
    predicted['certainty_threshold'] = None
    predicted['ll_threshold'] = None
    predicted["mis_pred"] = None
    predicted["mis_anno"] = None

    predicted['pred1'] = model_encoder.inverse_transform(predicted['idx1'])
    predicted['pred2'] = model_encoder.inverse_transform(predicted['idx2'])
    predicted['label'] = label

    predicted = pd.DataFrame(predicted)
    # print(predicted.head())
    
    # Prediction
    predicted = get_threshold(predicted, model_dir)
    # print(predicted.head())
    predicted = identify_TypeError(predicted)
    return predicted



def get_threshold(predicted, model_dir=None):
    """
    Load or compute class threshold
    """
    if model_dir is not None:
        cls_threshold = ProtoCloud.data.info_loader("cls_threshold", model_dir)
        celltypes = np.unique(cls_threshold['label'].values)
        for c in celltypes:
            predicted.loc[predicted['pred1'] == c, "ll_threshold"] = cls_threshold.loc[cls_threshold['label'] == c, "ll_threshold"].values[0]
            predicted.loc[predicted['pred1'] == c, "certainty_threshold"] = cls_threshold.loc[cls_threshold['label'] == c, "certainty_threshold"].values[0]
    
    else:
        celltypes = np.unique(predicted['pred1'].values)
        # certainty threshold
        for c in celltypes:
            cls_sim = predicted.loc[predicted['pred1'] == c, 'sim_score'].values
            sim_threshold = compute_threshold(cls_sim)
            
            predicted.loc[predicted['pred1'] == c, 'certainty_threshold'] = sim_threshold
        
        # log-likelihood threshold
        if 'll' in predicted.columns:
            for c in celltypes:
                cls_ll = predicted.loc[predicted['pred1'] == c, 'll'].values
                ll_threshold = compute_threshold(cls_ll)

                predicted.loc[predicted['pred1'] == c, 'll_threshold'] = ll_threshold

    return predicted



def get_cls_threshold(predicted):
    """
    return per class training data threshold
    """
    predicted = identify_TypeError(predicted)
    predicted = predicted.groupby('label').first().reset_index()

    return predicted[['label', 'certainty_threshold','ll_threshold']]



def identify_TypeError(predicted):
    predicted['certainty'] = 'certain'
    predicted.loc[predicted['sim_score'] < predicted['certainty_threshold'], 'certainty'] = 'ambiguous'

    if not predicted['label'].isnull().any():        
        if all(x in np.unique(predicted['label']) for x in np.unique(predicted['pred1'])):
            predicted["mis_pred"] = predicted['pred1'] != predicted['label']
            predicted["mis_anno"] = False
            predicted.loc[(predicted['mis_pred'] == True) & (predicted['certainty'] == "certain"), 'mis_anno'] = True

    return predicted



### Plot
#######################################################
def compute_threshold(score):
    # return np.quantile(score, 0.25)
    return np.quantile(score, 0.1)

    # Q1 = np.percentile(score, 25)
    # Q3 = np.percentile(score, 75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    
    # return lower_bound


def mutual_genes(list1, list2, celltype_specific=True):
    if celltype_specific:
        ul1 = [i for i in list1 if i not in list2]
        ul2 = [i for i in list2 if i not in list1]
        return ul1 + ul2
    else:
        ul2 = [i for i in list2 if i not in list1]
        return list1 + ul2


def get_avg_expression(adata, marker_genes):
    shared_markers = [gene for gene in marker_genes if gene in adata.var['gene_name']]
    print("shared_markers:", len(shared_markers))
    
    return shared_markers


def get_dotplot(adata, marker_genes, groupby='celltype', celltype_order=None, path=None):
    with plt.rc_context():  # Use this to set figure params like size and dpi
        sc.pl.dotplot(adata, 
                      marker_genes,
                      groupby = groupby,
                      gene_symbols = "gene_name",
                      categories_order = celltype_order,
                      standard_scale="var",
                      cmap='Blues',
                      show = False if path is not None else True,
                     )
        if path is not None:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def minmax_scale_matrix(matrix_np):
    row_mins = matrix_np.min(axis=1, keepdims=True)
    row_maxs = matrix_np.max(axis=1, keepdims=True)
    
    scaled_matrix = (matrix_np - row_mins) / (row_maxs - row_mins)
    return scaled_matrix



def rank_HRG(types:list, gene_names, prp_path, filename):
    scaler = MinMaxScaler()
    gene_names = np.array(gene_names)
    
    proto_rel1 = np.load(prp_path + types[0].replace("/", " OR ") + filename, allow_pickle=True)
    proto_rel1 = scaler.fit_transform(proto_rel1.T).T

    gene_rel1 = proto_rel1[0]
    topn = gene_rel1.shape[0]

    df = pd.DataFrame(data = gene_rel1).T
    df = pd.DataFrame(df.sum().nlargest(topn), columns=[f"{types[0]}_rel"])
    df['idx'] = df.index.tolist()
    df[f'{types[0]}_rank'] = np.arange(topn)
    df['gene_name'] = gene_names[df.index.tolist()].tolist()
    df = df[['idx', 'gene_name', f"{types[0]}_rel", f'{types[0]}_rank']]
    
    for type2 in types[1:]:
        proto_rel2 = np.load(prp_path + type2.replace("/", " OR ") + filename, allow_pickle=True)
        proto_rel2 = scaler.fit_transform(proto_rel2.T).T
        gene_rel2 = proto_rel2[0]

        df1 = pd.DataFrame(data = gene_rel2).T
        df1 = pd.DataFrame(df1.sum().nlargest(topn), columns=[f"{type2}_rel"])
        df1['idx'] = df1.index.tolist()
        df1[f'{type2}_rank'] = np.arange(topn)

        df = df.merge(df1, on='idx')
    return df


def calculate_batch_entropy(features, batch_labels, n_neighbors=10):
    # Calculate k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(features)
    _, indices = nbrs.kneighbors(features)

    entropies = []

    for i in range(features.shape[0]):
        # Get the batch labels of the neighbors (excluding the sample itself)
        neighbor_batches = batch_labels[indices[i][1:]]
        batch_counts = Counter(neighbor_batches)
        total_neighbors = sum(batch_counts.values())

        # Calculate entropy for the current sample
        entropy = 0.0
        for batch, count in batch_counts.items():
            p = count / total_neighbors
            entropy -= p * np.log(p)
        
        entropies.append(entropy)
    
    # Calculate and return the average entropy
    average_entropy = np.mean(entropies)
    return average_entropy