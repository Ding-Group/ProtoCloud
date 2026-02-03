# from torchvision import datasets
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anndata
from sklearn.preprocessing import MinMaxScaler
import copy

from .lrp_general import *
from ..utils import *
import ProtoCloud.glo as glo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0
# from PIL import Image
# from settings import *

EPS = glo.get_value('EPS')
LRP_FILTER_TOP_K = glo.get_value('LRP_FILTER_TOP_K')


class  Modulenotfounderror(Exception):
  pass


### LRP parameters
lrp_params_def1 = {
    'linear_eps': EPS,
    'linear_gamma': 0.1,
    'apply_filter': True,
    'linear_ignorebias': True,
    }

# relu_wrapper_fct, batchnorm1d_wrapper_fct
# linearlayer_eps_wrapper_fct, linearlayer_gamma_wrapper_fct
# linearlayer_Alpha1Beta0_wrapper_fct, linearlayer_Alpha2Beta1_wrapper_fct
rule_map = {
    'encoder': linearlayer_Alpha1Beta0_wrapper_fct, 
    'z_mean': linearlayer_eps_wrapper_fct,
    # 'encoder': linearlayer_abs_wrapper_fct,
    # 'z_mean': linearlayer_abs_wrapper_fct,
    'z_log_var': linearlayer_eps_wrapper_fct,
    'classifier': linearlayer_Alpha1Beta0_wrapper_fct 
}



def convert_protocloud_to_lrp(target_model, trained_model, 
                              lrp_params, rule_map, 
                              default_rule=linearlayer_Alpha1Beta0_wrapper_fct):
    """
    Args:
        target_model: lrp model
        trained_model: pretrained original model
        lrp_params: {'linear_eps': 1e-6, ...}
        rule_map: e.g. rule_map = {
                        'encoder': linearlayer_Alpha1Beta0_wrapper_fct, 
                        'z_mean': linearlayer_eps_wrapper_fct}
    """
    # --- Encoder (Sequential: Linear -> BN -> ReLU) ---
    target_model.encoder = nn.Sequential()
    for i, block in enumerate(trained_model.encoder):
        src_linear = block[0]
        src_bn = block[1]
        src_relu = block[2]
        
        # 1.1 Fuse BN into linear
        fused_linear = bnafterlinear_overwrite_intolinear(src_linear, src_bn)
        # 1.2 Wrap Linear
        wrapped_linear = get_lrpwrapperformodule(
            fused_linear, 
            lrp_params, 
            rule_map.get('encoder', default_rule)
        )
        # 1.3 Wrap ReLU
        wrapped_relu = zeroparam_wrapper_class(copy.deepcopy(src_relu), autogradfunction = relu_wrapper_fct)
        # 1.4 Identity BN
        wrapped_identity_bn = zeroparam_wrapper_class(resetbn(src_bn), autogradfunction = batchnorm1d_wrapper_fct)

        # 1.5 Sequential
        new_block = nn.Sequential(
            wrapped_linear,
            wrapped_identity_bn,
            wrapped_relu
        )
        target_model.encoder.add_module(str(i), new_block)

    # --- Linear only ---
    target_model.z_mean = get_lrpwrapperformodule(copy.deepcopy(trained_model.z_mean), 
                                                  lrp_params, rule_map.get('z_mean', default_rule))
    target_model.z_log_var = get_lrpwrapperformodule(copy.deepcopy(trained_model.z_log_var), 
                                                  lrp_params, rule_map.get('z_mean', default_rule))

    # sim_layer_instance = PrototypeSimilarityLRP(trained_model)
    # target_model.sim_layer = sim_layer_instance
    
    # --- Decoder (Sequential: Linear -> BN -> ReLU) ---
    target_model.decoder = nn.Sequential()
    for i, block in enumerate(trained_model.decoder):
        src_linear = block[0]
        src_bn = block[1]
        src_relu = block[2]
        
        # 1.1 Fuse BN into linear
        fused_linear = bnafterlinear_overwrite_intolinear(src_linear, src_bn)
        # 1.2 Wrap Linear
        wrapped_linear = get_lrpwrapperformodule(
            fused_linear, 
            lrp_params, 
            rule_map.get('encoder', default_rule)
        )
        # 1.3 Wrap ReLU
        wrapped_relu = zeroparam_wrapper_class(copy.deepcopy(src_relu), autogradfunction = relu_wrapper_fct)
        # 1.4 Identity BN
        wrapped_identity_bn = zeroparam_wrapper_class(resetbn(src_bn), autogradfunction = batchnorm1d_wrapper_fct)
        
        # 1.5 Sequential
        new_block = nn.Sequential(
            wrapped_linear,
            wrapped_identity_bn,
            wrapped_relu
        )
        target_model.decoder.add_module(str(i), new_block)

    # --- 4. 处理 Heads (输出层) ---
    target_model.px_mean = get_lrpwrapperformodule(copy.deepcopy(trained_model.px_mean), 
                                                  lrp_params, rule_map.get('px_mean', default_rule))
    
    # 4.2 px_theta (Linear -> Softplus)
    target_model.px_theta = nn.Sequential(
        get_lrpwrapperformodule(copy.deepcopy(trained_model.px_theta[0]), 
                                lrp_params, rule_map.get('px_theta', default_rule)),
        copy.deepcopy(trained_model.px_theta[1]) # Softplus 通常不需要 LRP wrap，或者简单透传
    )
    target_model.softmax = copy.deepcopy(trained_model.softmax)

    # 4.4 Classifier (Linear)
    target_model.classifier = get_lrpwrapperformodule(copy.deepcopy(trained_model.classifier),
                                                      lrp_params, rule_map.get('classifier', default_rule))

    ############# Copy parameters for non-wrapped layers ###############
    for param_name, param in trained_model.named_parameters():
        *module_path, param_attr = param_name.split('.')
        target_module = target_model
        for module_name in module_path:
            target_module = getattr(target_module, module_name)
        setattr(target_module, param_attr, nn.Parameter(param.clone().detach()))
        
    print("ProtoCloud LRP Model Conversion Complete.")



def x_prp1(model_wrapped, input, idx, cls_protos):
    """Modified x_prp: masked sim scores to closest prototype per class"""
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        ## backward参数: https://blog.csdn.net/sinat_28731575/article/details/90342082
        if model_wrapped.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        zx_mean = model_wrapped.encoder(x)
        z_mu = model_wrapped.z_mean(zx_mean)

        # R_zx = model_wrapped.calc_sim_scores(z_mu)
        R_zx = sim_score_eps_wrapper_fct.apply(z_mu, model_wrapped, 1e-9)
        _, max_proto_indices_per_class = torch.max(R_zx, dim=1)
        
        # R_zx.backward(torch.ones_like(R_zx))
        contrastive_signal = torch.full_like(R_zx, -1.0) # -1/R_zx.size(1)
        contrastive_signal[:, cls_protos] = 1/6
        contrastive_signal[:, idx] = 1.0
        # contrastive_signal = torch.full_like(R_zx, -1)
        # contrastive_signal[:, :6] = 1.0
        R_zx.backward(contrastive_signal)
        
        
        rel = x.grad.data.to('cpu').detach().numpy()    # gradient of input
        
        mask = (max_proto_indices_per_class == idx).to('cpu').numpy()
        if sum(mask) > 0:
            rel = rel[mask]

    rel_mean = np.mean(rel, axis = 0)   # [1, 3346]
    del x, zx_mean, z_mu, R_zx#, rel
    torch.cuda.empty_cache()
    return rel_mean, sum(mask)




def x_lrp(model, input):
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        if model.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        zx_mean = model.encoder(x)
        zx_mean = model.z_mean(zx_mean)
        # s_px = model.calc_sim_scores(zx_mean)
        s_px = sim_score_eps_wrapper_fct.apply(zx_mean, model, 1e-9)
        R_yx = model.classifier(s_px)

        R_yx.backward(torch.ones_like(R_yx))    # backward through encoder
        rel = x.grad.data.to('cpu').detach().numpy()  # gradient of input [n, 3346]
        # print("\tLRP:", np.max(rel), np.min(rel))
    
    rel_mean = np.mean(rel, axis = 0)
    del x, zx_mean, s_px, R_yx
    torch.cuda.empty_cache()
    return rel_mean



def z_recon(model, input):
    cell = 0
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        if model.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        z = model.encoder(x)
        zx_mean = model.z_mean(z)
        zx_mean = torch.nn.Parameter(zx_mean, requires_grad=True) 
        px = model.decoder(zx_mean)
        px_mu = model.px_mean(px)

        px_mu.backward(torch.ones_like(px_mu))    # backward through encoder
        rel = zx_mean.grad.data.to('cpu').detach().numpy()  # gradient of input

    rel_sum = np.sum(rel, axis = 0)   # [1, latent]
    return rel_sum



def normalize_max_abs(relevance_scores, epsilon=1e-9):
    """
    Symmetric scaling: maintains the relative proportion of positive and negative values.
    """
    max_abs_val = np.max(np.abs(relevance_scores))
    return relevance_scores / (max_abs_val + epsilon)


def generate_PRP_explanations(model,              # model_wrapped
                              train_X, train_Y, 
                              data,
                              epsilon, 
                              num_classes, prototypes_per_class,
                              prp_path=None, exp_code=None, 
                              pretrain_model_pth=None,
                              **kwargs):
    model.eval()

    gene_names = data.gene_names
    cell_types = data.cell_encoder.classes_
    n_genes = len(gene_names)

    global_prp_scores = np.zeros((num_classes, prototypes_per_class, n_genes))
    global_counts = np.zeros((num_classes, prototypes_per_class))

    print(f"Generating PRP explanations")

    if pretrain_model_pth is not None:
        model_dir = os.path.dirname(pretrain_model_pth)
        print(f"Loading previous prototype states from {model_dir}")
        load_path = os.path.join(model_dir, "prototype_checkpoint.npy")
        if os.path.exists(load_path):
            checkpoint = np.load(load_path, allow_pickle=True).item()
            # Ensure dimension match (prevent mixing different experiments)
            if all(gene_names == checkpoint['gene_names']):
                global_prp_scores = checkpoint['prp_scores']
            else:
                print("Warning: Gene mismatch in checkpoint. Starting from scratch.")
        else:
            print(f"Warning: Pre-train path provided but prototype_checkpoint.npy not found. Starting from scratch.")

    # ------------------------------------------------------------------
    # 2. Update Prototype-level PRP Scores
    scaler = MinMaxScaler()
    for c in range(num_classes):
        idx = np.where(train_Y == c)[0]
        
        for pno in range(prototypes_per_class):
            global_p_idx = c*prototypes_per_class + pno
            cls_protos = slice(c*prototypes_per_class, c*prototypes_per_class+prototypes_per_class)
            
            if len(idx) > 0:
                try:
                    p_rel_score, p_count = x_prp1(model, train_X[idx, :], global_p_idx, cls_protos)
                    print(f"Class {c} Proto {pno}: Count = {p_count}, min={np.min(p_rel_score):.6f}, max={np.max(p_rel_score):.6f}")
                except Exception as e:
                    print(f"Error computing x_prp1 for Class {c} Proto {pno}: {e}")
                    p_rel_score = np.zeros(n_genes)
                    p_count = 0

            # if no new data, adopt previous rel score from pre-trained model
            if p_count == 0 and pretrain_model_pth is not None:
                updated_score = global_prp_scores[c, pno]
            else:
                updated_score = p_rel_score

            # update global state
            global_prp_scores[c, pno] = updated_score
            global_counts[c, pno] = p_count
            
        
        # save celltype corresponding PRP genes (scaled)
        if np.sum(global_counts[c]) > 0:
            print(cell_types[c], global_prp_scores[c].shape)
            # class_prp = scaler.fit_transform(global_prp_scores[c].T).T
            class_prp = normalize_max_abs(global_prp_scores[c])
            path = prp_path + cell_types[c].replace("/", " OR ") + "_"+exp_code+"_relgenes.npy"
            save_file(class_prp, save_path=path)

    # ------------------------------------------------------------------
    # 3. Calculate Weighted relevance at Cell Type Level
    print("Calculating weighted cell-type PRP scores...")
    celltype_weighted_scores = []
    for c in range(num_classes):
        counts_c = global_counts[c]     # (n_prototypes,)
        scores_c = global_prp_scores[c] # (n_prototypes, n_genes)
        total_c = np.sum(counts_c)

        if total_c > 0:
            # Weighted average: sum(score_i * count_i) / sum(count_i)
            weighted_avg = np.dot(scores_c.T, counts_c) / total_c
        else:
            # no new data for this cell type, use zero (or previous score)
            weighted_avg = scores_c.mean(axis=0)
        celltype_weighted_scores.append(weighted_avg)
    celltype_weighted_scores = np.stack(celltype_weighted_scores) # (n_classes, n_genes)
    celltype_weighted_scores = pd.DataFrame(celltype_weighted_scores,
                                            index=cell_types,
                                            columns=gene_names)
    path = os.path.join(prp_path, "celltype_PRP.csv")
    save_file(celltype_weighted_scores, save_path=path)

    # ------------------------------------------------------------------
    # 4. Save in AnnData (Prototype Level)
    X_flat = global_prp_scores.reshape(-1, n_genes)
    obs_data = []
    for c in range(num_classes):
        for p in range(prototypes_per_class):
            obs_data.append({
                "celltype": cell_types[c],
                "prototype_idx": p,
                "count": global_counts[c, p],
                "unique_id": f"{cell_types[c]}_P{p}"
            })
    adata_proto = anndata.AnnData(
        X=X_flat,
        obs=pd.DataFrame(obs_data),
        var=pd.DataFrame(index=gene_names)
    )
    
    path = os.path.join(prp_path, f"{data.dataset_name}_prp.h5ad")
    adata_proto.write(path)
    print(f"Saved Prototype AnnData to {path}")
    
    checkpoint = {"cell_types": cell_types,
                  "prp_scores": global_prp_scores,
                  "gene_names": gene_names,
                  "counts": global_counts,
                  }
    return checkpoint



def generate_LRP_explanations(model,              # model_wrapped
                              train_X, train_Y, 
                              data, epsilon, 
                              num_classes, lrp_path=None, **kwargs):
    model.eval()

    gene_names = data.gene_names
    cell_types = data.cell_encoder.classes_
    n_genes = len(gene_names)

    global_prp_scores = np.zeros((num_classes, n_genes))
    global_counts = np.zeros((num_classes))

    print(f"Generating z_mu LRP explanations")
    # ------------------------------------------------------------------
    # 2. Update Prototype-level LRP Scores
    for c in range(num_classes):
        idx = np.where(train_Y == c)[0]
        
        if len(idx) > 0:
            try:
                p_rel_score = x_lrp(model, train_X[idx, :])
                print(f"Class {c}: Count = {len(idx)}, min={np.min(p_rel_score):.6f}, max={np.max(p_rel_score):.6f}")
            except Exception as e:
                print(f"Error computing x_lrp for Class {c}: {e}")
                p_rel_score = np.zeros(n_genes)
                
            global_prp_scores[c] = normalize_max_abs(p_rel_score)
    
    path = lrp_path + "zmu_lrp_relgenes.npy"
    save_file(global_prp_scores, save_path=path)

