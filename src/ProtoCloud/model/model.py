from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson
import numpy as np
import pandas as pd

import ProtoCloud.glo as glo
glo.set_value('EPS', 1e-16)
glo.set_value('LRP_FILTER_TOP_K', 0.1)

from ..utils import seed_torch, log_likelihood_nb, one_hot_encoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

EPS = glo.get_value('EPS')


def form_block(in_dim, out_dim,
                use_bn = True, activation = 'relu',
                bias = False, dropout = 0,
                ):
    """Construct a sequential block of Linear, BatchNorm, activation, and dropout.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    use_bn : bool, optional
        Whether to include BatchNorm1d, by default True.
    activation : {'relu', 'leakyrelu'}, optional
        Activation function type, by default 'relu'.
    bias : bool, optional
        Whether to add bias to the Linear layer, by default False.
    dropout : float, optional
        Dropout rate. Set to 0 to disable, by default 0.

    Returns
    -------
    torch.nn.Sequential
        The composed layer block.

    Raises
    ------
    ValueError
        If ``activation`` is not recognized.
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
    """
    ProtoCloud, a self-explaining deep generative model trained end-to-end to embed cells into a structured, low-dimensional space organized around cell type-specific prototypes.

    Parameters
    ----------
    input_dim : int
        Number of input genes.
    num_prototypes_per_class : int
        Number of prototype vectors per cell type.
    num_classes : int
        Number of cell types.
    latent_dim : int
        Dimensionality of the latent space.
    raw_input : int
        1 if input is raw counts (applies log1p), 0 if log-normalized.
    encoder_layer_sizes : list of int, optional
        Hidden layer sizes for the encoder, by default [1024, 512, 256].
    decoder_layer_sizes : list of int, optional
        Hidden layer sizes for the decoder, by default [512, 1024].
    activation : {'relu', 'leakyrelu'}, optional
        Activation function, by default 'relu'.
    use_bias : bool, optional
        Whether to use bias in Linear layers, by default False.
    use_dropout : float, optional
        Dropout rate, by default 0.
    use_bn : bool, optional
        Whether to use batch normalization, by default True.
    obs_dist : {'nb', 'normal'}, optional
        Observation distribution for reconstruction, by default 'nb'.
    nb_dispersion : {'celltype_target', 'celltype_pred', 'gene'}, optional
        How to model negative binomial dispersion, by default
        'celltype_target'.
    """

    def __init__(self, input_dim:int,
                 num_classes:int,
                 num_prototypes_per_class: int = 6,
                 latent_dim: int = 20,
                 raw_input:int = 1,
                 encoder_layer_sizes: Optional[list] = None,
                 decoder_layer_sizes: Optional[list] = None,
                 activation: Literal['relu', 'leakyrelu'] = 'relu', 
                 use_bias:bool = False,
                 use_dropout:float = 0,
                 use_bn:bool = True,
                 obs_dist: Literal['nb', 'normal'] = 'nb',
                 nb_dispersion: Literal['celltype_target', 'celltype_pred', 'gene'] = "celltype_target",
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
        """Forward pass through the full model.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression of shape ``(batch_size, input_dim)``.
        batch_id : optional
            Batch information (currently unused).

        Returns
        -------
        pred : torch.Tensor
            Classification logits, shape ``(batch_size, num_classes)``.
        px_mu : torch.Tensor
            Reconstruction mean, shape ``(batch_size, input_dim)``.
        px_t : torch.Tensor or None
            Dispersion parameters for NB distribution, or None.
        z_mu : torch.Tensor
            Latent mean, shape ``(batch_size, latent_dim)``.
        z_logVar : torch.Tensor
            Latent log-variance, shape ``(batch_size, latent_dim)``.
        sim_scores : torch.Tensor
            Prototype similarity scores, shape ``(batch_size, num_prototypes)``.
        """
        self.lib_size = torch.sum(x, 1, True)

        if self.raw_input:     # raw: 1
            x = torch.log1p(x)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        z_logVar = self.z_log_var(encode)
        z = self.reparameterize(z_mu, z_logVar)

        sim_scores = self.calc_sim_scores(z)
        pred = self.classifier(sim_scores)

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
        """Sample from the latent distribution using the reparameterization trick.

        Computes ``z = mu + exp(logvar / 2) * epsilon`` where epsilon is
        sampled from a standard normal.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        logvar : torch.Tensor
            Log-variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector, same shape as ``mu``.
        """
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)

        return mu + std * eps
    

    def loss_function(self, x, target, pred,
                      px_mu, px_theta,
                      z_mu, z_logVar,
                      sim_scores):
        """Compute all loss components for training.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression, shape ``(batch_size, input_dim)``.
        target : torch.Tensor
            Ground truth class labels, shape ``(batch_size,)``.
        pred : torch.Tensor
            Classification logits from the forward pass.
        px_mu : torch.Tensor
            Reconstruction mean from the forward pass.
        px_theta : torch.Tensor
            Dispersion parameters from the forward pass.
        z_mu : torch.Tensor
            Latent mean from the forward pass.
        z_logVar : torch.Tensor
            Latent log-variance from the forward pass.
        sim_scores : torch.Tensor
            Prototype similarity scores from the forward pass.

        Returns
        -------
        recon_loss : torch.Tensor
            Reconstruction loss (NB log-likelihood or MSE).
        kl_loss : torch.Tensor
            KL divergence to class prototypes.
        classify_loss : torch.Tensor
            Cross-entropy classification loss.
        ortho_loss : torch.Tensor
            Orthogonality loss for prototype separation.
        atomic_loss : torch.Tensor
            Attraction/repulsion loss for prototype assignment.
        """
        if target.max() >= self.prototype_class_identity.shape[0]:
            print("Max target:", target.max())
            print("Shape of prototype_class_identity:", self.prototype_class_identity.shape)
            raise IndexError("Target index is out of bounds.")

        # Reconstruction loss
        if self.nb_dispersion == 'celltype_target' or self.nb_dispersion == 'gene':            
            recon_loss, _ = self.recon_loss(x, target, px_mu, px_theta)
        else: # self.nb_dispersion == 'celltype_pred' 
            softmax_pred = F.softmax(pred, dim=1)
            max_index = torch.multinomial(softmax_pred, 1)
            recon_loss, _ = self.recon_loss(x, max_index, px_mu, px_theta)


        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, target]).to(device) 
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
        atomic_loss = self.atomic_loss(sim_scores, prototypes_of_correct_class)

        return recon_loss, kl_loss, classify_loss, ortho_loss, atomic_loss


    def calc_sim_scores(self, z):
        """Compute similarity scores between latent embeddings and prototypes.

        Uses the first half of latent dimensions to compute pairwise
        Euclidean distances, then converts to similarities via the
        Cauchy kernel.

        Parameters
        ----------
        z : torch.Tensor
            Latent embeddings, shape ``(batch_size, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Similarity scores, shape ``(batch_size, num_prototypes)``.
        """
        # pairwise Euclidean distances between z and prototype vectors
        d = torch.cdist(z[:, :self.latent_dim // 2], 
                        self.prototype_vectors[:, :self.latent_dim // 2], p = 2)  ## Batch size x num_prototypes
        sim_scores = self.distance_2_similarity(d)
        return sim_scores


    def distance_2_similarity(self, distances):
        """Convert distances to similarities using a Cauchy kernel.

        Computes ``1 / (scale^2 * d^2 + 1)``, yielding values in [0, 1].

        Parameters
        ----------
        distances : torch.Tensor
            Pairwise distance values.

        Returns
        -------
        torch.Tensor
            Similarity scores in [0, 1], same shape as input.
        """
        # return torch.log((distances + 1) / (distances + self.epsilon))
        return 1.0 / (torch.square(distances * self.scale) + 1.0)   # heavy tail


    def recon_loss(self, x, target, px_mu, px_t):
        """Compute reconstruction loss.

        Uses negative binomial log-likelihood for count data or MSE
        for normalized data.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth expression, shape ``(batch_size, input_dim)``.
        target : torch.Tensor
            Class labels, used for cell-type-specific dispersion.
        px_mu : torch.Tensor
            Predicted reconstruction mean.
        px_t : torch.Tensor or None
            Dispersion parameters.

        Returns
        -------
        loss : torch.Tensor
            Scalar reconstruction loss.
        dispersion : torch.Tensor or None
            Computed dispersion values, or None for normal distribution.
        """
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
        """Compute KL divergence to nearest class prototypes.

        The first half of the latent space is regularized toward prototype
        distributions (weight 5); the second half toward a standard
        normal (weight 1). Losses are weighted by similarity scores.

        Parameters
        ----------
        mu : torch.Tensor
            Latent means, shape ``(batch_size, latent_dim)``.
        logVar : torch.Tensor
            Latent log-variances, shape ``(batch_size, latent_dim)``.
        nearest_pt : torch.Tensor
            Prototype indices for correct class, shape
            ``(batch_size, num_prototypes_per_class)``.
        sim_scores : torch.Tensor
            Similarity scores, shape ``(batch_size, num_prototypes)``.

        Returns
        -------
        kl_loss : torch.Tensor
            Scalar KL divergence loss.
        mask : torch.Tensor
            Boolean mask indicating contributing prototypes.
        """
        kl_loss = torch.zeros(sim_scores.shape).to(device)
        half_latent = self.latent_dim // 2

        for i in range(self.num_prototypes_per_class):
            p_v = self.prototype_vectors[nearest_pt[:, i], :]     # all class prototype i vector

            kl1 = torch.distributions.kl.kl_divergence(
                torch.distributions.Normal(mu[:, :half_latent], torch.exp(logVar[:, :half_latent] / 2)),
                torch.distributions.Normal(p_v[:, :half_latent], torch.ones(p_v[:, :half_latent].shape).to(device))
                )

            kl2 = torch.distributions.kl.kl_divergence(
                torch.distributions.Normal(mu[:, half_latent:], torch.exp(logVar[:, half_latent:] / 2)),
                # torch.distributions.Normal(torch.zeros_like(p_v[:, half_latent:]).to(device), torch.ones_like(p_v[:, half_latent:]).to(device))
                torch.distributions.Normal(torch.zeros_like(p_v[:, half_latent:]), torch.ones_like(p_v[:, half_latent:]).to(device))
                )

            kl = torch.mean(kl1 * 5 + kl2, dim=-1)
            kl_loss[np.arange(sim_scores.shape[0]), nearest_pt[:, i]] = kl

        kl_loss = kl_loss * sim_scores    # element-wise scale by similarity scores
        mask = kl_loss > 0 # prototypes contributes
        kl_loss = torch.sum(kl_loss, dim = -1) / (torch.sum(sim_scores * mask, dim = -1))
        kl_loss = torch.mean(kl_loss)

        return kl_loss, mask


    def orthogonal_loss(self):
        """Compute orthogonality loss for prototype diversity.

        Encourages prototypes within the same class to be orthogonal
        in the first half of the latent space. Also includes L1
        sparsity regularization on the first encoder layer weights.

        Returns
        -------
        torch.Tensor
            Scalar orthogonality + sparsity loss.
        """
        s_loss = 0
        for k in range(self.num_classes):
            # p_k = self.prototype_vectors[k*self.num_prototypes_per_class : (k+1)*self.num_prototypes_per_class, :]
            p_k = self.prototype_vectors[k*self.num_prototypes_per_class : (k+1)*self.num_prototypes_per_class, :self.latent_dim//2]

            p_k_mean = torch.mean(p_k, dim = 0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(device))
            s_loss += torch.norm(s_matrix, p = 2)
        
        # # L1 regularization
        sparsity = 1.0 / torch.numel(self.encoder[0][0].weight) * torch.norm(self.encoder[0][0].weight, 1)

        return s_loss / self.num_classes + sparsity


    def atomic_loss(self, sim_scores, mask):
        """Compute attraction/repulsion loss for prototype assignment.

        Encourages high similarity to correct-class prototypes
        (attraction) and low similarity to incorrect-class prototypes
        (repulsion).

        Parameters
        ----------
        sim_scores : torch.Tensor
            Prototype similarities, shape ``(batch_size, num_prototypes)``.
        mask : torch.Tensor
            Boolean mask for correct-class prototypes.

        Returns
        -------
        torch.Tensor
            Scalar loss (repulsion - attraction).
        """
        attraction = torch.mean(torch.max(sim_scores * mask, 1).values)
        repulsion = torch.mean(torch.max(sim_scores * torch.logical_not(mask), 1).values)
        # repulsion = torch.sum(torch.mean(sim_scores * torch.logical_not(mask), 1).values)
        return repulsion - attraction



    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """Initialize classifier weights based on prototype-class identity.

        Sets weights to 1 for correct class connections and
        ``incorrect_strength`` for incorrect class connections.

        Parameters
        ----------
        incorrect_strength : float
            Weight for incorrect class connections (e.g., -0.5).
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)



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
        """torch.Tensor : Prototype vectors of shape ``(num_prototypes, latent_dim)``."""
        return self.prototype_vectors
    
    def get_prototype_cells(self):
        """Decode prototype vectors to gene expression space.

        Samples 100 times from the reconstruction distribution for each
        prototype and returns the mean counts.

        Returns
        -------
        torch.Tensor
            Prototype cells in gene space, shape
            ``(num_prototypes, input_dim)``.
        """
        px_mu, px_theta = self.get_latent_decode(self.prototype_vectors)
        # sample 100 and take avg for each
        proto_cells = torch.zeros(self.num_prototypes, self.input_dim)
        for i in range(self.num_classes):
            x_mu = px_mu[i*self.num_prototypes_per_class : (i+1)*self.num_prototypes_per_class, :]
            for j in range(self.num_prototypes_per_class):
                t = px_theta[:,i]
                mu = x_mu[j]
                proto_cells[i*self.num_prototypes_per_class + j, :] = torch.mean(self.sample_recon(mu, t, 100), axis=0)
        return proto_cells


    def max_sim_score(self, sim_scores):
        """Find the maximum similarity score per class and the best-matching prototype.

        Parameters
        ----------
        sim_scores : torch.Tensor
            Similarity scores, shape ``(batch_size, num_prototypes)``.

        Returns
        -------
        max_sim : torch.Tensor
            Maximum similarity to nearest prototype, shape ``(batch_size,)``.
        nearest_proto_idx : torch.Tensor
            Index of the nearest prototype within its class, shape ``(batch_size,)``.
        """
        sim_reshaped = sim_scores.view(-1, self.num_classes, self.num_prototypes_per_class)

        # max sim to a prototype for each class
        max_sim_per_cls, max_proto_indices_per_class = torch.max(sim_reshaped, dim=2)
        # Find the nearest class for each cell
        max_sim, nearest_cls_idx = torch.max(max_sim_per_cls, dim=1)
        # The indices of the nearest prototypes
        nearest_proto_idx = max_proto_indices_per_class[range(sim_reshaped.shape[0]), nearest_cls_idx]
        
        return max_sim, nearest_proto_idx


    def get_pred(self, x, test=False):
        """Get classification predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression, shape ``(batch_size, input_dim)``.
        test : bool, optional
            If True, use latent mean only (deterministic); otherwise
            sample via reparameterization, by default False.

        Returns
        -------
        pred : torch.Tensor
            Classification logits, shape ``(batch_size, num_classes)``.
        max_sim : torch.Tensor
            Maximum similarity to nearest prototype.
        proto_idx : torch.Tensor
            Index of the nearest prototype.
        """
        self.eval()
        if self.raw_input:     # raw: 1
            x = torch.log1p(x)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        if test:
            z = z_mu
        else:
            z_logVar = self.z_log_var(encode)
            z = self.reparameterize(z_mu, z_logVar)

        sim_scores = self.calc_sim_scores(z)
        pred = self.classifier(sim_scores)
        max_sim, proto_idx = self.max_sim_score(sim_scores)

        return pred, max_sim, proto_idx
    

    def get_latent(self, x):
        """Encode input to latent mean (deterministic, no sampling).

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression, shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Latent means, shape ``(batch_size, latent_dim)``.
        """
        self.eval()
        if self.raw_input:
            x = torch.log1p(x)
        encode = self.encoder(x)
        z_mu = self.z_mean(encode)

        return z_mu


    def get_latent_decode(self, z):
        """Decode latent vectors to gene expression space.

        Parameters
        ----------
        z : torch.Tensor
            Latent embeddings, shape ``(batch_size, latent_dim)``.

        Returns
        -------
        px_mu : torch.Tensor
            Reconstruction mean, shape ``(batch_size, input_dim)``.
        px_t : torch.Tensor or None
            Dispersion parameters, or None for normal distribution.
        """
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
        """Reconstruct input via full encode-decode pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression, shape ``(batch_size, input_dim)``.

        Returns
        -------
        px_mu : torch.Tensor
            Reconstruction mean.
        px_t : torch.Tensor or None
            Dispersion parameters, or None for normal distribution.
        """
        self.eval()
        if self.raw_input:     # raw: 1
            x = torch.log1p(x)

        encode = self.encoder(x)
        z_mu = self.z_mean(encode)
        z_logVar = self.z_log_var(encode)
        z = self.reparameterize(z_mu, z_logVar)

        px_mu, px_t = self.get_latent_decode(z)
        return px_mu, px_t



    def get_log_likelihood(self, input, target=None):
        """Compute negative log-likelihood averaged over multiple samples.

        Samples 5 times with different random seeds and averages the
        negative binomial log-likelihood across samples.

        Parameters
        ----------
        input : torch.Tensor
            Input gene expression, shape ``(batch_size, input_dim)``.
        target : torch.Tensor, optional
            Ground truth labels, required when ``nb_dispersion`` is
            ``'celltype_target'``.

        Returns
        -------
        torch.Tensor
            Mean negative log-likelihood per cell, shape ``(batch_size,)``.

        Raises
        ------
        NotImplementedError
            If ``obs_dist`` is not ``'nb'`` or if ``target`` is required
            but not provided.
        """
        if self.obs_dist != 'nb':
            raise NotImplementedError
        elif self.nb_dispersion == 'celltype_target' and target is None:
            print("Provide target label for log-likelihood calculation due to your choice of dispersion")
            raise NotImplementedError
        self.eval()
        
        n_sample = 5
        ll_value = 0
        for i in range(n_sample):
            with torch.no_grad():
                seed_torch(torch.device(device), seed = i, msg=False)

                pred, px_mu, px_t, _, _, _ = self.forward(input)

                if self.nb_dispersion == 'celltype_target':            
                    # data target
                    assert target is not None
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
        """Sample from the negative binomial reconstruction distribution.

        Uses a Gamma-Poisson compound to generate count samples.

        Parameters
        ----------
        px_mu : torch.Tensor
            Reconstruction mean.
        px_t : torch.Tensor
            Dispersion (concentration) parameter.
        sample_size : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor
            Sampled counts, shape ``(sample_size, ..., input_dim)``.
        """
        concentration = px_t
        rate = px_t / px_mu
        # Gamma(alpha, beta: rate = 1/scale)
        gamma_d = Gamma(concentration=concentration, rate=rate)
        p_means = gamma_d.rsample((sample_size,))
        l_train = torch.clamp(p_means, max=1e8)
        counts = Poisson(l_train).sample()  # (n_samples, n_cells, n_vars)
        return counts
