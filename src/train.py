import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import *
import src.glo as glo
EPS = glo.get_value('EPS')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0


def run_model(model, train_loader, test_loader, args):
    # setup optimizer
    lr = args.lr
    optimizer_specs = \
            [# layers
             {'params': model.encoder.parameters(), 'lr': lr, 'weight_decay': 0.005},
             {'params': model.decoder.parameters(), 'lr':lr, 'weight_decay': 0.005},
             {'params': model.z_mean.parameters(), 'lr': lr, 'weight_decay': 0},
             {'params': model.z_log_var.parameters(), 'lr': lr, 'weight_decay': 0},
             {'params': model.px_mean.parameters(), 'lr': lr, 'weight_decay': 0},
             {'params': model.px_theta.parameters(), 'lr': lr, 'weight_decay': 0},
             {'params': model.classifier.parameters(), 'lr':lr, 'weight_decay': 0.005},
             # parameters
             {'params': model.prototype_vectors, 'lr': lr * 0.8, 'weight_decay': 0.005},
             {'params': model.scale, 'lr': lr * 0.8, 'weight_decay': 0.005},
             {'params': model.theta, 'lr': lr * 0.8, 'weight_decay': 0.005},
             ]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(optimizer_specs)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_specs, eps = EPS)
    else:
        raise NotImplementedError
    
    # setup loss coef   
    coefs = {'crs_ent': 1, 'recon': 1,
                'kl': 1, 'ortho': 0.0, 'atomic': 0.0}
    print('loss coef:', coefs)


    # train
    two_step_training = args.two_step

    print('Start training')
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    start_time = time.time()
    for epoch in range(args.epochs+1):
        train_acc, train_loss, train_recon, train_kl, \
            train_ce, train_ortho, train_atomic = _train_model(model = model, 
                                                               dataloader = train_loader,
                                                               optimizer = optimizer,
                                                               coefs = coefs,
                                                               )
        # two-stage training: add ortho loss in the second half of training
        if two_step_training and epoch == args.epochs // 2:
            coefs['ortho'] = args.ortho_coef
            coefs['atomic'] = 1

        
        if (epoch % 10 == 0):
            print_results(epoch, train_acc, train_loss, train_recon, train_kl, \
                            train_ce, train_ortho, train_atomic, is_train=True)

            
            # validate
            test_acc, val_loss = _test_model(model = model, 
                                         dataloader = test_loader,
                                         coefs = coefs,
                                         )
            print_results(epoch, test_acc, val_loss, is_train = False)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(test_acc)
    
    end_time = time.time()
    print('\nFinished training')

    elapsed_time = end_time - start_time
    print(f"The running time is: {elapsed_time} sec")
    

    # save model to model_dir
    save_model(model, args.model_dir, args.exp_code)
    print('Model saved')

    return train_loss_list, train_acc_list, valid_acc_list




def _train_model(model, dataloader, optimizer, coefs): 
    model.train()

    n_examples = len(dataloader.dataset)
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_recons_loss = 0
    total_kl_loss = 0
    total_orth_loss = 0
    total_atomic_loss = 0

    for i, (sample, label) in enumerate(dataloader):
        input = sample.to(device)
        target = label.to(device)

        with torch.enable_grad():
            pred_y, px_mu, px_theta, z_mu, z_logVar, sim_scores = model(input)

            recon_loss, kl_loss, cross_entropy, ortho_loss, atomic_loss = model.loss_function(input, target, pred_y, px_mu, px_theta, z_mu, z_logVar, sim_scores)

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

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()

        del input, target, pred_y, predicted, px_mu, px_theta
        torch.cuda.empty_cache()

    train_acc = n_correct / n_examples
    train_ce = total_cross_entropy / n_batches
    train_recon = total_recons_loss / n_batches
    train_kl = total_kl_loss / n_batches
    train_ortho = total_orth_loss / n_batches
    train_atomic = total_atomic_loss / n_batches

    return train_acc, loss.item(), train_recon, train_kl, train_ce, train_ortho, train_atomic



def _test_model(model, dataloader, coefs): 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    n_examples = len(dataloader.dataset)
    n_correct = 0

    for i, (sample, label) in enumerate(dataloader):
        input = sample.to(device)
        target = label.to(device)

        pred_y, px_mu, px_theta, z_mu, z_logVar, sim_scores = model(input)

        recon_loss, kl_loss, cross_entropy, \
            ortho_loss, atomic_loss = model.loss_function(input, target, pred_y, px_mu,
                                                         px_theta, z_mu, z_logVar, sim_scores)

        # get prediction
        _, predicted = torch.max(pred_y.data, 1)
        n_correct += (predicted == target).sum().item()

        # compute gradient and do SGD step
        loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['recon'] * recon_loss
                    + coefs['kl'] * kl_loss
                    + coefs['ortho'] * ortho_loss
                    + coefs['atomic'] * atomic_loss
                )

        del input, target, pred_y, predicted, px_mu, px_theta
        torch.cuda.empty_cache()

    test_acc = n_correct / n_examples
    return test_acc, loss.item()



def load_model(args, model, model_dir=None):
    '''
    Load model from model_dir
    '''
    if model_dir == None:
        model_dir = args.model_dir + args.exp_code + '.pth'
        print(model_dir)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    # model = torch.load(model_dir)
    # model.eval()
    print("Model loaded")



def get_predictions(model, X, test: bool):
    """
    Get top 2 predictions from model
        test: if True, use z_mean for prediction
              if False, use z_mean + z_log_var for prediction
    """
    input = torch.Tensor(X).to(device)
    pred, sim_scores = model.get_pred(input, test)

    softmax = torch.nn.Softmax(dim = -1)
    pred = softmax(pred)
    top2_probs, top2_idxs = torch.topk(pred, 2)

    prob1 = top2_probs[:,0].detach().cpu().numpy()
    prob2 = top2_probs[:,1].detach().cpu().numpy()
    idx1 = top2_idxs[:,0].detach().cpu().numpy()
    idx2 = top2_idxs[:,1].detach().cpu().numpy()
    top2_pred = {
        'prob1': prob1,
        'prob2': prob2,
        'idx1': idx1,
        'idx2': idx2
    }

    del input
    return top2_pred, sim_scores



def get_latent(model, X):
    input = torch.Tensor(X).to(device)
    latent = model.get_latent(input).cpu().detach().numpy()
    del input
    return latent



def get_prototypes(model):
    prototypes = model.prototype_vectors.cpu().detach().numpy()
    return prototypes



def get_prototype_cells(model):
    prototype_cells = model.get_prototype_cells().detach().cpu().numpy()
    prototype_cells = (prototype_cells + 1) / 2.0     # scale to [0,1]