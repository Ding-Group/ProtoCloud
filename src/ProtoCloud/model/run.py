import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ProtoCloud.data.scRNAdata import *
from ProtoCloud.utils.utils import *
import ProtoCloud.glo as glo
EPS = glo.get_value('EPS')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0


def run_model(model, 
            train_X, train_Y,
            batch_size = 1028,
            epochs = 100, 
            lr = 1e-3, 
            optimizer = "AdamW",
            two_step = True,
            recon_coef = 10, 
            kl_coef = 2,
            ortho_coef = 0.3,
            atomic_coef= 1,
            validate_model = True, test_X = None, test_Y = None,
            model_dir = None, results_dir = None,
            **kwargs):

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
    

    train_loader = scRNAData.assign_dataloader(train_X, train_Y, batch_size)


    # setup loss coef  
    two_step_training = two_step
    edge = 30 if epochs // 2 > 30 else epochs // 2

    coefs = {'crs_ent': 1, 'recon': recon_coef, 'kl': kl_coef,
            'ortho': 0.0 if two_step_training else ortho_coef,
            'atomic': 0.0 if two_step_training else atomic_coef,
            }
    print('loss coef:', coefs)

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
                coefs['ortho'] = ortho_coef
                coefs['atomic'] = atomic_coef

        # validate
        if validate_model:
            test_acc = _test_model(model = model, 
                                input = test_X, label = test_Y,
                                coefs = coefs,
                                )
        else:
            test_acc = None

        if (epoch % 10 == 0):
            print_results(epoch, train_acc, train_loss, train_recon, train_kl, \
                            train_ce, train_ortho, train_atomic, is_train=True)
            print_results(epoch, test_acc, is_train = False)
        
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


def load_model(model_dir, model):
    '''
    Load model from model_dir
    '''
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # model = torch.load(model_dir)
    # model.eval()
    print("Model loaded")



def get_log_likelihood(model, X=None, Y=None):
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


def get_predictions(model, X, test: bool = False):
    """
    Get top 2 predictions from model
        test: if True, use z_mean for prediction
              if False, use z_mean + z_log_var for prediction
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
            pred, max_sim, proto_idx = model.get_pred(input, test)
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
    return top2_pred



def get_latent(model, X):
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
    input = torch.Tensor(Z).to(device)
    px_mu, px_theta = model.get_latent_decode(input)
    px_mu = px_mu.detach().cpu().numpy()
    px_theta = px_theta.detach().cpu().numpy()
    del input
    return px_mu, px_theta


def get_recon(model, X):
    input = torch.Tensor(X).to(device)
    recon, _ = model.get_recon(input)
    recon = recon.detach().cpu().numpy()
    del input
    return recon


def get_prototypes(model):
    prototypes = model.get_prototypes.cpu().detach().numpy()
    return prototypes


def get_prototype_cells(model):
    proto_cells = model.get_prototype_cells()
    proto_cells = proto_cells.detach().cpu().numpy()
    return proto_cells