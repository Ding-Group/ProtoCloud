import os

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .model.model import protoCloud
from .model.train import run_model, get_predictions, get_log_likelihood
from .model.calibrator import simCalibration
from .data.scRNAdata import scRNAData
from .utils.utils import (
    seed_torch, save_model, save_model_dict, load_model_dict,
    data_info_saver, data_info_loader, process_prediction_file,
    get_cls_threshold, identify_TypeError, get_threshold, makedir,
)


class ProtoCloudModel:
    """High-level API for training and predicting with ProtoCloud.

    Parameters
    ----------
    **kwargs
        Architecture hyperparameters passed to the protoCloud constructor.
        Supported keys: latent_dim, num_prototypes_per_class, activation,
        use_bn, obs_dist, nb_dispersion, use_bias, use_dropout,
        encoder_layer_sizes, decoder_layer_sizes, raw_input.
    """

    def __init__(self, **kwargs):
        self.latent_dim = kwargs.get('latent_dim', 20)
        self.num_prototypes_per_class = kwargs.get('num_prototypes_per_class', 6)
        self.activation = kwargs.get('activation', 'relu')
        self.use_bn = kwargs.get('use_bn', 1)
        self.obs_dist = kwargs.get('obs_dist', 'nb')
        self.nb_dispersion = kwargs.get('nb_dispersion', 'celltype_target')
        self.use_bias = kwargs.get('use_bias', 0)
        self.use_dropout = kwargs.get('use_dropout', 0.0)
        self.encoder_layer_sizes = kwargs.get('encoder_layer_sizes', None)
        self.decoder_layer_sizes = kwargs.get('decoder_layer_sizes', None)
        self.raw_input = kwargs.get('raw_input', 1)

        self._model = None
        self._model_dict = None
        self._cell_encoder = None
        self._gene_names = None
        self._calibrator = None
        self._cls_threshold = None
        self._device = None
        self._train_idx = None
        self._load_dir = None

    def fit_model(self, adata,
                  celltype_col='celltype',
                  count_layer='counts',
                  test_ratio=0.1,
                  data_balance=True,
                  seed=7,
                  epochs=100,
                  lr=1e-3,
                  batch_size=None,
                  optimizer='AdamW',
                  two_step=True,
                  recon_coef=10,
                  kl_coef=2,
                  ortho_coef=0.3,
                  stage1_ortho_coef=0.0,
                  atomic_coef=1,
                  validate=True,
                  **kwargs):
        """Train a ProtoCloud model on an AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix with gene expression. Must have
            ``adata.var['gene_name']`` and cell type labels in
            ``adata.obs[celltype_col]``.
        celltype_col : str
            Column name in ``adata.obs`` containing cell type labels.
        count_layer : str
            Layer name in ``adata.layers`` for raw counts. Falls back
            to ``adata.X`` if the layer does not exist.
        test_ratio : float
            Fraction of data held out for validation.
        data_balance : bool
            Whether to augment rare cell types via oversampling.
        seed : int
            Random seed for reproducibility.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int or None
            Mini-batch size. Auto-selected if None.
        optimizer : str
            Optimizer type ('Adam' or 'AdamW').
        two_step : bool
            Whether to use two-stage curriculum training.
        recon_coef : float
            Reconstruction loss coefficient.
        kl_coef : float
            KL divergence loss coefficient.
        ortho_coef : float
            Orthogonality loss coefficient.
        stage1_ortho_coef : float
            Stage 1 orthogonality loss coefficient for two-step training.
        atomic_coef : float
            Atomic loss coefficient.
        validate : bool
            Whether to evaluate on a held-out validation set.
        **kwargs
            Additional keyword arguments forwarded to ``run_model``.

        Returns
        -------
        tuple
            (train_loss_list, train_acc_list, valid_acc_list) from training.
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed_torch(self._device, seed=seed)

        gene_names = adata.var['gene_name'].values
        celltypes = adata.obs[celltype_col].values

        continue_training = self._model is not None

        if continue_training:
            # gene alignment for continue training
            X = self._align_genes(adata, count_layer)
            assert self._cell_encoder is not None, "cell_encoder missing for continue training"
            two_step = False
        else:
            # first training: extract counts directly
            X = _extract_counts(adata, count_layer)
            input_dim = X.shape[1]
            num_classes = len(np.unique(celltypes))

            self._model_dict = {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'num_prototypes_per_class': self.num_prototypes_per_class,
                'latent_dim': self.latent_dim,
                'raw_input': self.raw_input,
                'encoder_layer_sizes': self.encoder_layer_sizes,
                'decoder_layer_sizes': self.decoder_layer_sizes,
                'activation': self.activation,
                'use_bias': self.use_bias,
                'use_dropout': self.use_dropout,
                'use_bn': self.use_bn,
                'obs_dist': self.obs_dist,
                'nb_dispersion': self.nb_dispersion,
            }
            self._model = protoCloud(**self._model_dict).to(self._device)

            _, self._cell_encoder = scRNAData._label_encoder(np.unique(celltypes))
            self._gene_names = gene_names

        # train/test split
        if test_ratio > 0:
            train_idx, test_idx = train_test_split(
                range(X.shape[0]), test_size=test_ratio, shuffle=True,
                random_state=seed,
            )
        else:
            train_idx = np.arange(X.shape[0])
            test_idx = np.array([], dtype=int)

        self._train_idx = np.asarray(train_idx)

        train_X = X[train_idx]
        train_Y_str = celltypes[train_idx]
        train_Y = self._cell_encoder.transform(train_Y_str)

        test_X = X[test_idx] if len(test_idx) > 0 else None
        test_Y_str = celltypes[test_idx] if len(test_idx) > 0 else None
        test_Y = self._cell_encoder.transform(test_Y_str) if test_Y_str is not None else None

        if data_balance:
            train_X, train_Y = _augment_rares(train_X, train_Y, self._cell_encoder)

        validate_model = validate and test_X is not None
        result = run_model(
            self._model, train_X, train_Y,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            two_step=two_step,
            recon_coef=recon_coef,
            kl_coef=kl_coef,
            ortho_coef=ortho_coef,
            stage1_ortho_coef=stage1_ortho_coef,
            atomic_coef=atomic_coef,
            validate_model=validate_model,
            test_X=test_X,
            test_Y=test_Y,
            **kwargs,
        )

        # post-training calibration
        predicted = get_predictions(self._model, train_X)
        predicted = process_prediction_file(
            predicted,
            model_encoder=self._cell_encoder,
            label=self._cell_encoder.inverse_transform(train_Y),
        )
        self._cls_threshold = get_cls_threshold(predicted)

        self._calibrator = simCalibration()
        self._calibrator.fit(
            predicted['sim_score'].values,
            predicted['label'].values,
            predicted['pred1'].values,
        )

        return result

    def predict_model(self, adata, count_layer='counts'):
        """Predict cell types for new data.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix to predict. Must have
            ``adata.var['gene_name']``.
        count_layer : str
            Layer name in ``adata.layers`` for raw counts.

        Returns
        -------
        anndata.AnnData
            The input adata with predictions added to ``adata.obs``:
            ``pc_prediction``, ``pc_sim_score``, ``pc_certainty``,
            ``pc_log_likelihood`` (if obs_dist == 'nb').
        """
        assert self._model is not None, "Model not trained. Call fit_model first."

        X = self._align_genes(adata, count_layer)

        predicted = get_predictions(self._model, X)

        # decode predicted indices to cell type names
        pred_labels = self._cell_encoder.inverse_transform(predicted['idx1'].values)
        sim_scores = predicted['sim_score'].values

        # calibrated certainty
        calibrated_certainty = self._calibrator.predict_proba(sim_scores, pred_labels)

        # certainty category using cls_threshold
        certainty = np.full(len(pred_labels), 'certain', dtype=object)
        for _, row in self._cls_threshold.iterrows():
            label = row['label']
            threshold = row['certainty_threshold']
            mask = (pred_labels == label) & (sim_scores < threshold)
            certainty[mask] = 'ambiguous'
        # cells with labels not in cls_threshold default to 'certain'

        adata.obs['pc_prediction'] = pred_labels
        adata.obs['pc_sim_score'] = sim_scores
        adata.obs['pc_certainty'] = certainty
        adata.obs['pc_calibrated_certainty'] = calibrated_certainty

        # log-likelihood for negative binomial models
        if self.obs_dist == 'nb':
            Y_encoded = self._cell_encoder.transform(pred_labels)
            ll = get_log_likelihood(self._model, X=X, Y=Y_encoded)
            adata.obs['pc_log_likelihood'] = ll

        return adata

    def save_model(self, model_dir):
        """Save all model artifacts to a directory.

        Parameters
        ----------
        model_dir : str
            Directory path to save model files.
        """
        assert self._model is not None, "No model to save."
        makedir(model_dir)

        # ensure trailing separator for functions that use string concatenation
        model_dir_sep = model_dir if model_dir.endswith(os.sep) else model_dir + os.sep

        save_model(self._model, model_dir, 'protocloud')
        save_model_dict(self._model_dict, model_dir_sep)
        data_info_saver(self._cell_encoder, model_dir, 'cell_encoder')
        data_info_saver(self._gene_names, model_dir, 'gene_names')
        data_info_saver(self._cls_threshold, model_dir, 'cls_threshold')
        self._calibrator.save(model_dir_sep)

    @classmethod
    def load(cls, model_dir, device=None):
        """Load a saved ProtoCloudModel from a directory.

        Parameters
        ----------
        model_dir : str
            Directory containing saved model artifacts.
        device : str or None
            Device to load the model onto. Auto-detected if None.

        Returns
        -------
        ProtoCloudModel
            Loaded model ready for prediction.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        instance = cls()
        instance._device = torch.device(device)

        # load model architecture dict and weights
        instance._model_dict = load_model_dict(model_dir, device)
        instance._model = protoCloud(**instance._model_dict).to(instance._device)
        checkpoint_path = os.path.join(model_dir, 'protocloud.pth')
        state_dict = torch.load(checkpoint_path, map_location=instance._device)
        instance._model.load_state_dict(state_dict)
        instance._model.eval()

        # load metadata
        instance._cell_encoder = data_info_loader('cell_encoder', model_dir)
        instance._gene_names = np.array(data_info_loader('gene_names', model_dir))
        instance._cls_threshold = data_info_loader('cls_threshold', model_dir)

        model_dir_sep = model_dir if model_dir.endswith(os.sep) else model_dir + os.sep
        instance._calibrator = simCalibration.load(model_dir_sep)

        # sync architecture hyperparameters from model_dict
        instance.latent_dim = instance._model_dict.get('latent_dim', 20)
        instance.num_prototypes_per_class = instance._model_dict.get('num_prototypes_per_class', 6)
        instance.activation = instance._model_dict.get('activation', 'relu')
        instance.use_bn = instance._model_dict.get('use_bn', True)
        instance.obs_dist = instance._model_dict.get('obs_dist', 'nb')
        instance.nb_dispersion = instance._model_dict.get('nb_dispersion', 'celltype_target')
        instance.use_bias = instance._model_dict.get('use_bias', False)
        instance.use_dropout = instance._model_dict.get('use_dropout', 0.0)
        instance.encoder_layer_sizes = instance._model_dict.get('encoder_layer_sizes', None)
        instance.decoder_layer_sizes = instance._model_dict.get('decoder_layer_sizes', None)
        instance.raw_input = instance._model_dict.get('raw_input', 1)

        return instance

    def _align_genes(self, adata, count_layer='counts'):
        """Align adata genes to the model's gene space and return dense counts.

        Parameters
        ----------
        adata : anndata.AnnData
            Input data with possibly different gene ordering/set.
        count_layer : str
            Layer name for raw counts.

        Returns
        -------
        numpy.ndarray
            Dense count matrix aligned to ``self._gene_names``,
            shape ``(n_cells, n_model_genes)``, dtype float32.
        """
        current_genes = adata.var['gene_name'].values
        model_genes = self._gene_names

        share_genes = [g for g in model_genes if g in current_genes]
        n_shared = len(share_genes)
        n_model = len(model_genes)
        print(f"\tShared genes: {n_shared}/{n_model} "
              f"({n_shared / n_model * 100:.1f}%)")

        gene_mask = [g in share_genes for g in model_genes]
        var_indices = [np.where(current_genes == g)[0][0]
                       for g in share_genes if (current_genes == g).any()]

        # get source matrix
        if count_layer in adata.layers:
            source = adata.layers[count_layer]
        else:
            source = adata.X

        new_shape = (source.shape[0], n_model)
        aligned = scRNAData._resize_and_fill(source, new_shape, gene_mask, var_indices)

        # convert to dense float32
        if sparse.issparse(aligned):
            X = np.asarray(aligned.todense(), dtype=np.float32)
        else:
            X = np.asarray(aligned, dtype=np.float32)

        return X


def _extract_counts(adata, count_layer):
    """Extract count matrix from adata as dense float32 numpy array.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.
    count_layer : str
        Layer name to use for counts.

    Returns
    -------
    numpy.ndarray
        Dense count matrix, dtype float32.
    """
    if count_layer in adata.layers:
        source = adata.layers[count_layer]
    else:
        source = adata.X

    if sparse.issparse(source):
        return np.asarray(source.todense(), dtype=np.float32)
    return np.asarray(source, dtype=np.float32)


def _augment_rares(X, Y, cell_encoder):
    """Oversample rare cell types via multinomial sampling.

    Fixed version of ``scRNAData.augment_rares`` that accepts the
    encoder as a parameter instead of referencing ``self``.

    Parameters
    ----------
    X : numpy.ndarray
        Gene expression matrix, shape ``(n_cells, n_genes)``.
    Y : numpy.ndarray
        Numeric cell type labels, shape ``(n_cells,)``.
    cell_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder for decoding cell type names.

    Returns
    -------
    augmented_X : numpy.ndarray
        Expression matrix with synthetic samples appended.
    augmented_Y : numpy.ndarray
        Labels including synthetic samples.
    """
    from torch.distributions import Multinomial

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
    print(f"\tRare cell types: {len(rares)}")
    print(f"\tRare cell types: {[cell_encoder.inverse_transform([c])[0] for c in rares]}")
    print(f"\tTotal samples to add: {num_sample}")

    new_X = torch.zeros((X.shape[0] + num_sample, X.shape[1]))
    new_Y = torch.zeros(Y.shape[0] + num_sample, dtype=torch.long)
    new_X[:X.shape[0]] = torch.from_numpy(X)
    new_Y[:Y.shape[0]] = torch.from_numpy(Y.astype(np.int64))

    start_idx = X.shape[0]
    for c in rares:
        rare = X[Y == c]
        n_add = min_type_num - np.sum(Y == c)
        rates = torch.from_numpy(rare)
        idx = np.tile(np.arange(rare.shape[0]), n_add // rare.shape[0] + 1)
        rates = rates[np.random.choice(idx, n_add, replace=False)]

        for i in range(n_add):
            rate_i = rates[i, :] + 0.01
            n_i = torch.sum(rate_i)
            p = rate_i.to(torch.float64) / n_i

            u_i = np.random.randint(low=50, high=100)
            new_umi = np.ceil(n_i * u_i / 100).type(torch.int32)
            new_X[start_idx + i] = Multinomial(total_count=int(new_umi), probs=p).sample()

        new_Y[start_idx:start_idx + n_add] = torch.full((1, n_add), c)
        start_idx += n_add

        del rare, rates

    return new_X.numpy(), new_Y.numpy()
