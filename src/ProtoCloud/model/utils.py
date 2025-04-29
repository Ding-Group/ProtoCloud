import os
import torch, random, math, pickle
import numpy as np

from ProtoCloud import glo
EPS = glo.get_value('EPS')


def seed_torch(device, seed = 7, msg=True):
    """
    Sets Seed for reproducible experiments.
    """
    if msg:
        print("Global seed set to {}".format(seed))
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



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


def one_hot_encoder(target, n_cls):
    assert torch.max(target).item() <= n_cls

    target = target.view(-1, 1)
    onehot = torch.zeros(target.size(0), n_cls)
    onehot = onehot.to(target.device)
    onehot.scatter_(1, target.long(), 1)

    return onehot


### Model Data Retrieval
#######################################################
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


