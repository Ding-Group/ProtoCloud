import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data import *
from .model import protoCloud
from .. import utils
import ProtoCloud.glo as glo
EPS = glo.get_value('EPS')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0


def run_model(model,
            train_X, train_Y,
            batch_size = None,
            epochs = 100,
            lr = 1e-3,
            optimizer = "AdamW",
            two_step = True,
            recon_coef = 10,
            kl_coef = 2,
            ortho_coef = 0.3,
            stage1_ortho_coef = 0.0,
            atomic_coef= 1,
            validate_model = False, test_X = None, test_Y = None,
            model_dir = None, results_dir = None,
            **kwargs):
    """Train ProtoCloud model with two-stage curriculum.

    Parameters
    ----------
    model : protoCloud model
        The model to train.
    train_X : numpy.ndarray
        Training gene expression matrix.
    train_Y : numpy.ndarray
        Training labels (numeric, 0-indexed).
    batch_size : int, optional
        Mini-batch size, by default 1028.
    epochs : int, optional
        Number of training epochs, by default 100.
    lr : float, optional
        Learning rate, by default 1e-3.
    optimizer : {'Adam', 'AdamW'}, optional
        Optimizer type, by default 'AdamW'.
    two_step : bool, optional
        Whether to use two-stage training, by default True.
    recon_coef : float, optional
        Reconstruction loss coefficient, by default 10.
    kl_coef : float, optional
        KL divergence loss coefficient (stage 2), by default 2.
    ortho_coef : float, optional
        Orthogonality loss coefficient (stage 2), by default 0.3.
    stage1_ortho_coef : float, optional
        Orthogonality loss coefficient for stage 1 in two-step
        training, by default 0.0.
    atomic_coef : float, optional
        Atomic loss coefficient (stage 2), by default 1.
    validate_model : bool, optional
        Whether to compute validation accuracy, by default True.
    test_X : numpy.ndarray, optional
        Test expression matrix for validation.
    test_Y : numpy.ndarray, optional
        Test labels for validation.
    model_dir : str, optional
        Directory to save model checkpoints (unused).
    results_dir : str, optional
        Directory to save results (unused).

    Returns
    -------
    train_loss_list : list of float
        Training loss per epoch.
    train_acc_list : list of float
        Training accuracy per epoch.
    valid_acc_list : list of float
        Validation accuracy per epoch (empty if not validating).
    """

    # setup optimizer
    # optimizer_specs = [# layers
    #          {'params': model.encoder.parameters(), 'lr': lr, 'weight_decay': 0.005},
    #          {'params': model.decoder.parameters(), 'lr':lr, 'weight_decay': 0},
    #          {'params': model.z_mean.parameters(), 'lr': lr, 'weight_decay': 0},
    #          {'params': model.z_log_var.parameters(), 'lr': lr, 'weight_decay': 0},
    #          {'params': model.px_mean.parameters(), 'lr': lr, 'weight_decay': 0},
    #          {'params': model.px_theta.parameters(), 'lr': lr, 'weight_decay': 0},
    #          {'params': model.classifier.parameters(), 'lr':lr, 'weight_decay': 0.005},
    #          # parameters
    #          {'params': model.prototype_vectors, 'lr': lr * 0.8, 'weight_decay': 0.005},
    #          {'params': model.scale, 'lr': lr * 0.8, 'weight_decay': 0.005},
    #          {'params': model.theta, 'lr': lr * 0.8, 'weight_decay': 0.005},
    #          ]
    optimizer_specs = [{'params': model.parameters(), 'lr': lr}]
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(optimizer_specs)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_specs)
    else:
        raise NotImplementedError
    
    if batch_size is None:
        batch_size = 1024 if len(train_X) > 1e5 else 128
    train_loader = scRNAData.assign_dataloader(train_X, train_Y, batch_size)


    # setup loss coef  
    two_step_training = two_step
    edge = 30 if epochs // 2 > 30 else epochs // 2

    coefs = {'crs_ent': 1, 'recon': recon_coef, 
            'kl': 2 if two_step_training else kl_coef,
            'ortho': stage1_ortho_coef if two_step_training else ortho_coef,
            'atomic': 0.0 if two_step_training else atomic_coef,
            }
    print('Init loss coef:', coefs)

    # train
    print('Start training')
    start_time = time.time()
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(epochs+1):
        train_acc, train_loss, train_recon, train_kl, \
            train_ce, train_ortho, train_atomic = _train_model(model = model, 
                                                               dataloader = train_loader,
                                                               optimizer = optimizer,
                                                               coefs = coefs,
                                                               )
        
        # two-stage training: add ortho loss in the second half of training
        if two_step_training:
            if epoch == edge:
                coefs['kl'] = kl_coef
                coefs['ortho'] = ortho_coef
                coefs['atomic'] = atomic_coef
                print('Updated loss coef:', coefs)

        # validate
        if validate_model:
            test_acc = _test_model(model = model, 
                                input = test_X, label = test_Y,
                                coefs = coefs,
                                )
        else:
            test_acc = None

        if (epoch % 10 == 0):
            utils.print_results(epoch, train_acc, train_loss, train_recon, train_kl, \
                            train_ce, train_ortho, train_atomic, is_train=True)
            if validate_model:
                utils.print_results(epoch, test_acc, is_train = False)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(test_acc)
    

    end_time = time.time()
    print('\nFinished training')
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    if validate_model:
        return (train_loss_list, train_acc_list, valid_acc_list)
    else:
        return (train_loss_list, train_acc_list, valid_acc_list)




def _train_model(model, dataloader, optimizer, coefs): 
    model.train()

    n_examples = len(dataloader.dataset)
    n_correct = 0
    n_batches = 0
    total_loss = 0
    total_cross_entropy = 0
    total_recons_loss = 0
    total_kl_loss = 0
    total_orth_loss = 0
    total_atomic_loss = 0

    for i, (sample, label) in enumerate(dataloader):
        input = sample.to(device)
        target = label.to(device)
        # batch_id = b.to(device)

        with torch.enable_grad():
            pred_y, px_mu, px_theta, z_mu, z_logVar, sim_scores = model(input)

            recon_loss, kl_loss, cross_entropy, \
                ortho_loss, atomic_loss = model.loss_function(input, target, pred_y, 
                                                              px_mu, px_theta, z_mu, z_logVar, 
                                                              sim_scores)

            # get prediction
            _, predicted = torch.max(pred_y.data, 1)
            n_correct += (predicted == target).sum().item()

            # update metrics
            total_recons_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_cross_entropy += cross_entropy.item()
            total_orth_loss += ortho_loss.item()
            total_atomic_loss += atomic_loss.item()
            n_batches += 1

        # compute gradient and do SGD step
        loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['recon'] * recon_loss
                    + coefs['kl'] * kl_loss
                    + coefs['ortho'] * ortho_loss
                    + coefs['atomic'] * atomic_loss
                )
        total_loss += loss.item()


        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()

        del input, target, pred_y, predicted, px_mu, px_theta
        torch.cuda.empty_cache()

    train_acc = n_correct / n_examples
    train_loss = total_loss / n_batches
    train_ce = total_cross_entropy / n_batches
    train_recon = total_recons_loss / n_batches
    train_kl = total_kl_loss / n_batches
    train_ortho = total_orth_loss / n_batches
    train_atomic = total_atomic_loss / n_batches

    return train_acc, train_loss, train_recon, train_kl, train_ce, train_ortho, train_atomic


def _test_model(model, input, label, coefs): 
    model.eval()

    input = torch.Tensor(input).to(device)
    target = torch.LongTensor(label).to(device)

    pred_y, px_mu, px_theta, z_mu, z_logVar, sim_scores = model(input)
    # get prediction
    _, predicted = torch.max(pred_y.data, 1)
    n_correct = (predicted == target).sum().item()

    del input, target, pred_y, predicted, px_mu, px_theta
    torch.cuda.empty_cache()

    test_acc = n_correct / len(label)
    return test_acc#, loss.item()


def freeze_modules(model):
    for param in model.parameters():
        param.requires_grad = False


def save_model(model, model_dict, model_dir, results_dir, exp_code=None,
               result_trend=None, save_prototypes=True, accu=None, log=print):
    """Save model artifacts in one call.
    Saves model weights, model config, training trend and prototypes.
    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    model_dict : dict
        Configuration dictionary used to build the model.
    model_dir : str
        Directory for model files.
    results_dir : str
        Directory for results files.
    exp_code : str
        Experiment code prefix for filenames.
    result_trend : object, optional
        Training trend object to save (as _trend.npy).
    save_prototypes : bool, default True
        Whether to save prototypes to results.
    """
    if exp_code is None:
        exp_code = 'protocloud'
    # preserve existing utils.save_model behavior
    utils.save_model(model, model_dir, exp_code)
    utils.save_model_dict(model_dict, model_dir)

    if result_trend is not None and results_dir is not None:
        utils.save_file(result_trend, results_dir, exp_code, '_trend.npy')

    if save_prototypes and results_dir is not None:
        utils.save_file(get_prototypes(model), results_dir, exp_code, '_prototypes.npy')


def load_model(model_dir_or_checkpoint, model=None, exp_code=None, device='cpu'):
    """Load model weights.
    Modes:
        1) (checkpoint + model instance) - backward-compatible:
            load_model(checkpoint_path, model_instance)
        2) load_model(model_dir, exp_code='run1', device=device)
    """
    if model is not None:
        state_dict = torch.load(model_dir_or_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded")
        return model

    if exp_code is None:
        exp_code = 'protocloud'

    model_dict = utils.load_model_dict(model_dir_or_checkpoint, device)
    model = protoCloud(**model_dict).to(device)
    checkpoint_path = os.path.join(model_dir_or_checkpoint, exp_code + '.pth')
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded")
    return model



def get_log_likelihood(model, X=None, Y=None):
    """Compute negative log-likelihood.

    Parameters
    ----------
    model : protoCloud
        Trained model in eval mode.
    X : numpy.ndarray
        Input gene expression matrix.
    Y : numpy.ndarray, optional
        Labels, required for ``celltype_target`` dispersion.

    Returns
    -------
    numpy.ndarray
        Negative log-likelihood per cell, shape ``(n_samples,)``.
    """
    results = []
    if Y is not None:
        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1000)
        with torch.no_grad():
            for i, (sample, label) in enumerate(dataloader):
                input = sample.to(device)
                target = label.to(device)
                output = model.get_log_likelihood(input, target).detach().cpu().numpy()

                results.append(output)
    
    else:
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X)),
                                                  batch_size = 1000)
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                input = sample[0].to(device)
                output = model.get_log_likelihood(input).detach().cpu().numpy()
                results.append(output)

    results = np.concatenate(results, axis=0)
    return results


def get_predictions(model, X):
    """Get top-2 predictions and prototype similarity scores.

    Parameters
    ----------
    model : protoCloud
        Trained model in eval mode.
    X : numpy.ndarray
        Input gene expression matrix.

    Returns
    -------
    pandas.DataFrame
        DataFrame with prediction columns
    """
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X)),
                                             batch_size = 1000)
    top2_pred = {
        'prob1': [],
        'prob2': [],
        'idx1': [],
        'idx2': [],
        'sim_proto': [],
        'sim_score': [],
    }

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim = -1)
        for sample in dataloader:
            input = sample[0].to(device)
            pred, max_sim, proto_idx = model.get_pred(input)
            pred = softmax(pred)

            del input
            top2_probs, top2_idxs = torch.topk(pred, 2)
            max_sim = max_sim.detach().cpu().numpy()
            proto_idx = proto_idx.detach().cpu().numpy()

            prob1 = top2_probs[:,0].detach().cpu().numpy()
            prob2 = top2_probs[:,1].detach().cpu().numpy()
            idx1 = top2_idxs[:,0].detach().cpu().numpy()
            idx2 = top2_idxs[:,1].detach().cpu().numpy()

            top2_pred['prob1'].append(prob1)
            top2_pred['prob2'].append(prob2)
            top2_pred['idx1'].append(idx1)
            top2_pred['idx2'].append(idx2)
            top2_pred['sim_proto'].append(proto_idx)
            top2_pred['sim_score'].append(max_sim)
    
    for p in top2_pred:
        top2_pred[p] = np.concatenate(top2_pred[p], axis=0)
    
    return pd.DataFrame(top2_pred)



def get_latent(model, X):
    """Extract latent mean embeddings.

    Parameters
    ----------
    model : protoCloud
        Trained model in eval mode.
    X : numpy.ndarray
        Input gene expression matrix.

    Returns
    -------
    numpy.ndarray
        Latent embeddings, shape ``(n_samples, latent_dim)``.
    """
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X)),
                                            batch_size = 1000)
    
    latent_embedding = []
    for sample in dataloader:
        input = sample[0].to(device)
        latent = model.get_latent(input).cpu().detach().numpy()
        latent_embedding.append(latent)
        del input, latent

    latent_embedding = np.concatenate(latent_embedding, axis=0)
    return latent_embedding


def get_latent_decode(model, Z):
    """Decode latent vectors to gene expression space.

    Parameters
    ----------
    model : protoCloud
        Trained model.
    Z : numpy.ndarray
        Latent vectors, shape ``(n_samples, latent_dim)``.

    Returns
    -------
    px_mu : numpy.ndarray
        Reconstruction mean.
    px_theta : numpy.ndarray
        Dispersion parameters.
    """
    input = torch.Tensor(Z).to(device)
    px_mu, px_theta = model.get_latent_decode(input)
    px_mu = px_mu.detach().cpu().numpy()
    px_theta = px_theta.detach().cpu().numpy()
    del input
    return px_mu, px_theta


def get_recon(model, X):
    """Get full reconstruction of input data.

    Parameters
    ----------
    model : protoCloud
        Trained model.
    X : numpy.ndarray
        Input gene expression matrix.

    Returns
    -------
    numpy.ndarray
        Reconstructed expression matrix.
    """
    input = torch.Tensor(X).to(device)
    recon, _ = model.get_recon(input)
    recon = recon.detach().cpu().numpy()
    del input
    return recon


def get_prototypes(model):
    """Extract prototype vectors as a numpy array.

    Parameters
    ----------
    model : protoCloud
        Trained model.

    Returns
    -------
    numpy.ndarray
        Prototype vectors, shape ``(num_prototypes, latent_dim)``.
    """
    prototypes = model.get_prototypes.cpu().detach().numpy()
    return prototypes


def get_prototype_cells(model):
    """Get prototype reconstructions in gene expression space.

    Parameters
    ----------
    model : protoCloud
        Trained model.

    Returns
    -------
    numpy.ndarray
        Prototype cells, shape ``(num_prototypes, input_dim)``.
    """
    proto_cells = model.get_prototype_cells()
    proto_cells = proto_cells.detach().cpu().numpy()
    return proto_cells