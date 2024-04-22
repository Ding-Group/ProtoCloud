from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson
import numpy as np

from src.utils import seed_torch, log_likelihood_nb, one_hot_encoder
import src.glo as glo
glo._init()
glo.set_value('EPS', 1e-16)
glo.set_value('LRP_FILTER_TOP_K', 0.1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

EPS = glo.get_value('EPS')


def form_block(in_dim, out_dim,
                use_bn = True, activation = 'relu',
                bias = False, dropout = 0,
                ):
    """
    Constructs a fully connected layer with bias, batch norm, and then leaky relu activation function
    args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        batch_norm (bool): use the batch norm in the layers, defaults to True
        bias (bool): add bias to the layers
    returns (array): the layers specified
    """
    layers = [nn.Linear(in_dim, out_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    # activation
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    else:
        raise ValueError('activation not recognized')
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layers)

    
class protoCloud(nn.Module):
    def __init__(self, input_dim:int,
                 num_prototypes_per_class:int, 
                 num_classes:int,  
                 latent_dim:int, 
                 raw_input:int, 
                 encoder_layer_sizes: Optional[list] = None,
                 decoder_layer_sizes: Optional[list] = None,
                 activation: Literal['relu', 'leakyrelu'] = 'relu', 
                 use_bias:bool = False,
                 use_dropout:float = 0,
                 use_bn:bool = True,
                 obs_dist: Literal['nb', 'normal'] = 'nb',
                 nb_dispersion: Literal['celltype_target', 'celltype_pred', 'gene'] = "celltype_pred",
                #  n_batch:int = 1,
                 ):
        super(protoCloud, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_classes = num_classes
        self.raw_input = raw_input
        self.activation = activation
        self.use_bn = use_bn
        self.use_bias = bool(use_bias)
        self.use_dropout = use_dropout
        self.obs_dist = None if not self.raw_input else obs_dist
        self.nb_dispersion = nb_dispersion
        self.epsilon = EPS
        # self.n_batch = n_batch
        

        # prototype-class labeled matrix
        self.num_prototypes = self.num_prototypes_per_class * self.num_classes
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        # prototype vectors
        prototype_shape = (self.num_prototypes, self.latent_dim)
        self.prototype_vectors = nn.Parameter(torch.randn(prototype_shape), requires_grad = True)
        # mask
        self.scale = nn.Parameter(torch.ones(1) * 1.0)


        #######################################################
        if encoder_layer_sizes is None:
            self.encoder_layer_sizes = [self.input_dim] + [1024, 512, 256] # [128, 64, 32] 
        else:
            self.encoder_layer_sizes = [self.input_dim] + encoder_layer_sizes
        if decoder_layer_sizes is None:
            # self.latent_dim += self.n_batch
            self.decoder_layer_sizes = [self.latent_dim] + [512, 1024] # [32, 128]
        else:
            self.decoder_layer_sizes = [self.latent_dim] + decoder_layer_sizes
            # self.decoder_layer_sizes[0] += self.n_batch

        # Encoder
        self.encoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(self.encoder_layer_sizes[:-1], self.encoder_layer_sizes[1:])):
            self.encoder.add_module(str(i), form_block(in_dim, out_dim, 
                                    self.use_bn, self.activation, self.use_bias, self.use_dropout))

        self.z_mean = nn.Linear(self.encoder_layer_sizes[-1], latent_dim, bias = True)
        self.z_log_var = nn.Linear(self.encoder_layer_sizes[-1], latent_dim, bias = True)
        
        # Decoder
        self.decoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(self.decoder_layer_sizes[:-1], self.decoder_layer_sizes[1:])):
            self.decoder.add_module(str(i), form_block(in_dim, out_dim, 
                                    self.use_bn, self.activation, self.use_bias, self.use_dropout))
        self.px_mean = nn.Linear(self.decoder_layer_sizes[-1], input_dim, bias = True)

        # likelihood
        self.softmax = nn.Softmax(dim = -1)
        # nb dispersion: gene-specific
        self.px_theta = nn.Sequential(
            nn.Linear(self.decoder_layer_sizes[-1], input_dim, bias = True),
            nn.Softplus())   # output always positive
        # nb dispersion: celltype-specific
        self.theta = nn.Parameter(torch.randn(self.input_dim, self.num_classes))
        
        # Classifier
        self.classifier = nn.Linear(self.num_prototypes, self.num_classes, bias = False)
        self._initialize_weights()



    def forward(self, x, batch_id=None):
        self.lib_size = torch.sum(x, 1, True)

        if self.raw_input:     # raw: 1
            x = torch.log(x + 1)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        z_logVar = self.z_log_var(encode)
        z = self.reparameterize(z_mu, z_logVar)

        sim_scores = self.calc_sim_scores(z)
        pred = self.classifier(sim_scores)

        # # add batch to decoder
        # batch = torch.nn.functional.one_hot(batch_id, num_classes=self.n_batch)
        # print("batch_dim", batch.shape)
        # z = torch.concat((z, batch), axis=1)
        # print("z_dim", z.shape)

        px = self.decoder(z)
        px_mu = self.px_mean(px)

        if self.obs_dist == 'nb':
            px_mu = self.softmax(px_mu) * self.lib_size
            if self.nb_dispersion.startswith('celltype') :
                px_t = self.theta
            elif self.nb_dispersion == 'gene':
                px_t = self.px_theta(px)
                px_t = torch.mean(px_t, 0, True)
            else:
                raise NotImplementedError
            
            px_t = torch.clamp(px_t, min = EPS)
        else:
            px_t = None

        return pred, px_mu, px_t, z_mu, z_logVar, sim_scores

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)

        return mu + std * eps
    

    def loss_function(self, x, target, pred, px_mu, px_theta, z_mu, z_logVar, sim_scores):
        if target.max() >= self.prototype_class_identity.shape[0]:
            print("Max target:", target.max())
            print("Shape of prototype_class_identity:", self.prototype_class_identity.shape)
            raise IndexError("Target index is out of bounds.")


        # Reconstruction loss
        if self.nb_dispersion == 'celltype_target' or self.nb_dispersion == 'gene':            
            # data target or gene
            recon_loss, _ = self.recon_loss(x, target, px_mu, px_theta)
        else: # self.nb_dispersion == 'celltype_pred' 
            softmax_pred = F.softmax(pred, dim=1)
            max_index = torch.multinomial(softmax_pred, 1)
            recon_loss, _ = self.recon_loss(x, max_index, px_mu, px_theta)


        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, target.cpu()])
        index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(as_tuple = True)[1]
        # class-corresponding prototypes' index for each sample in the batch
        index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(x.shape[0], 
                                                                                   self.num_prototypes_per_class)

        # KL divergence loss
        kl_loss, mask = self.kl_divergence_nearest(z_mu, z_logVar, index_prototypes_of_correct_class, sim_scores)
        
        # Classification loss
        classify_loss = F.cross_entropy(pred, target)
        
        # Orthogonality loss
        ortho_loss = self.orthogonal_loss()
        
        # Atomic loss
        atomic_loss = self.atomic_loss(sim_scores, mask)

        return recon_loss, kl_loss, classify_loss, ortho_loss, atomic_loss


    def calc_sim_scores(self, z):
        # pairwise Euclidean distances between z and prototype vectors
        d = torch.cdist(z[:, :self.latent_dim // 2], 
                        self.prototype_vectors[:, :self.latent_dim // 2], p = 2)  ## Batch size x num_prototypes
        sim_scores = self.distance_2_similarity(d)
        return sim_scores
    
    def distance_2_similarity(self, distances):
        # return torch.log((distances + 1) / (distances + self.epsilon))
        return 1.0 / (torch.square(distances * self.scale) + 1.0)   # heavy tail
    
    # def classification_loss(self, pred, target):
    #     # return F.cross_entropy(pred, target)

    #     pred_reshaped = pred.view(-1, self.num_classes, self.prototypes_per_class)
    #     # Apply max pooling across the prototypes for each class
    #     outputs_max, _ = torch.max(pred_reshaped, dim=2)
    #     return F.cross_entropy(outputs_max, targets)


    def recon_loss(self, x, target, px_mu, px_t):
        if self.obs_dist == 'nb':
            if self.nb_dispersion.startswith('celltype'):
                dispersion = F.linear(one_hot_encoder(target, self.num_classes), self.theta)
                dispersion = torch.exp(dispersion)

            elif self.nb_dispersion == 'gene':
                dispersion = px_t
            else:
                raise NotImplementedError
            
            ll = -log_likelihood_nb(x, px_mu, dispersion)
            recon_loss = torch.mean(torch.sum(ll, dim = -1))
            recon_loss = recon_loss / self.input_dim * self.latent_dim / 2.0 # scale nb loss down

        else:
            # x = F.normalize(x, dim = 0)
            recon_loss = torch.nn.functional.mse_loss(px_mu, x, reduction = "mean")
            dispersion = None 

        return recon_loss, dispersion


    def kl_divergence_nearest(self, mu, logVar, nearest_pt, sim_scores):
        kl_loss = torch.zeros(sim_scores.shape).to(device) 

        for i in range(self.num_prototypes_per_class):
            p = torch.distributions.Normal(mu, torch.exp(logVar / 2))
            p_v = self.prototype_vectors[nearest_pt[:, i], :]     # all class prototype i vector
            q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
            kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim = -1)
            kl_loss[np.arange(sim_scores.shape[0]), nearest_pt[:, i]] = kl

        kl_loss = kl_loss * sim_scores    # element-wise scale by similarity scores
        mask = kl_loss > 0 # prototypes contributes

        # prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, target.cpu()])
        # print(torch.equal(kl_loss > 0, prototypes_of_correct_class.to(device))) # True
        # print(torch.sum(kl_loss > 0), torch.sum(kl_loss == 0), target)
        # exit()

        kl_loss = torch.sum(kl_loss, dim = -1) / (torch.sum(sim_scores * mask, dim = -1))
        kl_loss = torch.mean(kl_loss)

        return kl_loss, mask
    

    def orthogonal_loss(self):
        s_loss = 0
        for k in range(self.num_classes):
            p_k = self.prototype_vectors[k*self.num_prototypes_per_class : (k+1)*self.num_prototypes_per_class, :]
            p_k_mean = torch.mean(p_k, dim = 0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(device))
            s_loss += torch.norm(s_matrix, p = 2)
        
        # L1 regularization
        sparsity = 1.0 / torch.numel(self.encoder[0][0].weight) * torch.norm(self.encoder[0][0].weight, 1)

        return s_loss / self.num_classes + sparsity


    def atomic_loss(self, sim_scores, mask):
        attraction = torch.mean(torch.max(sim_scores * mask, 1).values)
        repulsion = torch.mean(torch.max(sim_scores * torch.logical_not(mask), 1).values)
        # repulsion = torch.sum(torch.mean(sim_scores * torch.logical_not(mask), 1))
        return repulsion - attraction


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
    
    
    # def _initialize_weights(self):
    #     '''
    #     initialize weights for all layers
    #     '''
    #     for m_name, m in self.named_modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, -0.08, 0.08)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0.001)
    #         else:
    #             pass
    #     self.set_last_layer_incorrect_connection(incorrect_strength = -0.5)


    def _initialize_weights(self):
        '''
        initialize weights for vae
        '''
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.001)

        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.001)

        self.set_last_layer_incorrect_connection(incorrect_strength = -0.5)



    # get results helper functions
    #######################################################
    @property
    def get_prototypes(self):
        return self.prototype_vectors
    
    def get_prototype_cells(self):
        px_mu, px_theta = self.get_latent_decode(self.prototype_vectors)
        # sample 100 and take avg for each
        proto_cells = torch.zeros(self.num_prototypes, self.input_dim)
        for i in range(self.num_classes):
            x_mu = px_mu[i*self.num_prototypes_per_class : (i+1)*self.num_prototypes_per_class, :]
            for j in range(self.num_prototypes_per_class):
                t = px_t[:,i]
                mu = x_mu[j]
                proto_cells[i*self.num_prototypes_per_class + j, :] = torch.mean(model.sample_recon(mu, t, 100), axis=0)
        return proto_cells


    def get_pred(self, x, test = False):
        self.eval()
        if self.raw_input:     # raw: 1
            x = torch.log(x + 1)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        if test:
            z = z_mu
        else:
            z_logVar = self.z_log_var(encode)
            z = self.reparameterize(z_mu, z_logVar)

        sim_scores = self.calc_sim_scores(z)
        pred = self.classifier(sim_scores)

        return pred, sim_scores
    

    def get_latent(self, x):
        self.eval()
        if self.raw_input:
            x = torch.log(x + 1)
        encode = self.encoder(x)
        z_mu = self.z_mean(encode)

        return z_mu

    
    def get_latent_decode(self, z):
        px = self.decoder(z)
        px_mu = self.px_mean(px)
        px_t = self.px_theta(px)

        if self.obs_dist == 'nb':
            px_mu = self.softmax(px_mu) * self.num_prototypes
            if self.nb_dispersion.startswith('celltype'):
                px_t = self.theta
            elif self.nb_dispersion == 'gene':
                px_t = self.px_theta(px)
                px_t = torch.mean(px_t, 0, True)
            else:
                raise NotImplementedError
            
            px_t = torch.clamp(px_t, min = EPS)
        else:
            px_t = None

        return px_mu, px_t


    def get_recon(self, x):
        self.eval()
        if self.raw_input:     # raw: 1
            x = torch.log(x + 1)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        z_logVar = self.z_log_var(encode)
        z = self.reparameterize(z_mu, z_logVar)

        px_mu, px_t = self.get_latent_decode(z)
        return px_mu, px_t


    def get_log_likelihood(self, input, target):
        if self.obs_dist != 'nb':
            raise NotImplementedError
        self.eval()
        
        n_sample = 5
        ll_value = 0
        for i in range(n_sample):
            with torch.no_grad():
                seed_torch(torch.device(device), seed = i)

                pred, px_mu, px_t, _, _, _ = self.forward(input)

                if self.nb_dispersion == 'celltype_target':            
                    # data target
                    dispersion = F.linear(one_hot_encoder(target, self.num_classes), self.theta)
                    dispersion = torch.exp(dispersion)
                elif self.nb_dispersion == 'celltype_pred': 
                    # pred target
                    softmax_pred = F.softmax(pred, dim=1)
                    max_index = torch.multinomial(softmax_pred, 1)
                    dispersion = F.linear(one_hot_encoder(max_index, self.num_classes), self.theta)
                    dispersion = torch.exp(dispersion)
                elif self.nb_dispersion == 'gene':
                    dispersion = px_t
                else:
                    raise NotImplementedError
                
                ll = -log_likelihood_nb(input, px_mu, dispersion)
                ll = torch.sum(ll, dim = -1)
                ll_value += ll

                del px_mu, px_t, dispersion, ll
                torch.cuda.empty_cache()

        return ll_value / n_sample


    def sample_recon(self, px_mu, px_t, sample_size):
        concentration = px_t
        rate = px_t / px_mu
        # Gamma(alpha, beta: rate = 1/scale)
        gamma_d = Gamma(concentration=concentration, rate=rate)
        p_means = gamma_d.rsample((sample_size,))
        l_train = torch.clamp(p_means, max=1e8)
        counts = Poisson(l_train).sample()  # (n_samples, n_cells, n_vars)
        return counts
