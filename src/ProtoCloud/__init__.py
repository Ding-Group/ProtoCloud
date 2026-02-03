from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ProtoCloud")
except PackageNotFoundError:
    __version__ = "unknown"

from ProtoCloud import glo
from ProtoCloud.model import protoCloud

from ProtoCloud import model
from ProtoCloud import data
from ProtoCloud import prp
from ProtoCloud import utils
from ProtoCloud import viz

from ProtoCloud.data import (CustomDataset, EPS, scRNAData,)
from ProtoCloud.glo import (get_value, set_value, )
from ProtoCloud.model import (simCalibration, EPS, device,
                              form_block, freeze_modules, get_latent,
                              get_latent_decode, get_log_likelihood,
                              get_predictions, get_prototype_cells,
                              get_prototypes, get_recon, load_model,
                              num_workers, protoCloud, run_model,)
from ProtoCloud.prp import (EPS, LRP_FILTER_TOP_K, Modulenotfounderror,
                            add_epsilon_fn, batchnorm1d_wrapper_fct,
                            bnafterlinear_overwrite_intolinear,
                            convert_protocloud_to_lrp, device, gamma_fn,
                            generate_LRP_explanations,
                            generate_PRP_explanations, get_lrpwrapperformodule,
                            identity_fn, linearlayer_Alpha1Beta0_wrapper_fct,
                            linearlayer_Alpha2Beta1_wrapper_fct,
                            linearlayer_abs_wrapper_fct,
                            linearlayer_eps_wrapper_fct,
                            linearlayer_gamma_wrapper_fct, lrp_backward,
                            lrp_params_def1,
                            lrplookupnotfounderror, negposlinear,
                            normalize_max_abs, num_workers,
                            oneparam_wrapper_class, poslinear, posneglinear,
                            relevance_filter, relu_wrapper_fct, resetbn,
                            rule_map, safe_divide, sim_score_eps_wrapper_fct,
                            tensorlist_todict, twoparam_wrapper_class, x_lrp,
                            x_prp1, z_recon, zeroparam_wrapper_class,)
from ProtoCloud.utils import (EPS, all_to_coo, calculate_batch_entropy,
                              compute_threshold, data_info_loader,
                              data_info_saver, get_avg_expression,
                              get_cls_threshold, get_custom_exp_code,
                              get_dotplot, get_threshold, identify_TypeError,
                              load_file, load_model_dict, load_var_names,
                              log_likelihood_nb, log_likelihood_normal,
                              makedir, minmax_scale_matrix, model_metrics,
                              mutual_genes, one_hot_encoder, print_results,
                              process_prediction_file, rank_HRG, save_file,
                              save_model, save_model_dict,
                              save_model_w_condition, save_prototype_cells,
                              seed_torch,)
from ProtoCloud.viz import (EPS, device, get_UMAP, get_color, map_color,
                            num_workers, plot_2latent,
                            plot_all_class_lrp_dist, plot_cell_likelihoods,
                            plot_confusion_matrix, plot_distance_to_prototypes,
                            plot_epoch_trend, plot_gene_expression,
                            plot_gene_rel_VS_expr, plot_gene_relevance,
                            plot_latent_embedding, plot_lrp_dist,
                            plot_marker_venn_diagram, plot_outlier_heatmap,
                            plot_prediction_summary, plot_protocorr_heatmap,
                            plot_protos_2latent, plot_protos_umap,
                            plot_prp_dist, plot_top_gene_PRP_dotplot,
                            plot_umap,)

__all__ = ["glo", "protoCloud", "model", "scRNAData", "prp", "utils", "viz"]   
# __all__ = ['simCalibration', 'CustomDataset', 'EPS', 'LRP_FILTER_TOP_K',
#            'Modulenotfounderror', 'add_epsilon_fn', 'all_to_coo',
#            'batchnorm1d_wrapper_fct', 'bnafterlinear_overwrite_intolinear',
#            'calculate_batch_entropy', 'calibrator', 'compute_threshold',
#            'convert_protocloud_to_lrp', 'data', 'data_info_loader',
#            'data_info_saver', 'device', 'form_block', 'freeze_modules',
#            'gamma_fn', 'generate_LRP_explanations',
#            'generate_PRP_explanations', 'get_UMAP', 'get_avg_expression',
#            'get_cls_threshold', 'get_color', 'get_custom_exp_code',
#            'get_dotplot', 'get_latent', 'get_latent_decode',
#            'get_log_likelihood', 'get_lrpwrapperformodule', 'get_predictions',
#            'get_prototype_cells', 'get_prototypes', 'get_recon',
#            'get_threshold', 'get_value', 'glo', 'identify_TypeError',
#            'identity_fn', 'linearlayer_Alpha1Beta0_wrapper_fct',
#            'linearlayer_Alpha2Beta1_wrapper_fct',
#            'linearlayer_abs_wrapper_fct', 'linearlayer_eps_wrapper_fct',
#            'linearlayer_gamma_wrapper_fct', 'load_file', 'load_model',
#            'load_model_dict', 'load_var_names', 'log_likelihood_nb',
#            'log_likelihood_normal', 'lrp_backward',
#            'lrp_params_def1', 'lrplookupnotfounderror', 'makedir', 'map_color',
#            'minmax_scale_matrix', 'model', 'model_metrics', 'mutual_genes',
#            'negposlinear', 'normalize_max_abs', 'num_workers',
#            'one_hot_encoder', 'oneparam_wrapper_class', 'plot', 'plot_2latent',
#            'plot_all_class_lrp_dist', 'plot_cell_likelihoods',
#            'plot_confusion_matrix', 'plot_distance_to_prototypes',
#            'plot_epoch_trend', 'plot_gene_expression', 'plot_gene_rel_VS_expr',
#            'plot_gene_relevance', 'plot_latent_embedding', 'plot_lrp_dist',
#            'plot_marker_venn_diagram', 'plot_outlier_heatmap',
#            'plot_prediction_summary', 'plot_protocorr_heatmap',
#            'plot_protos_2latent', 'plot_protos_umap', 'plot_prp_dist',
#            'plot_top_gene_PRP_dotplot', 'plot_umap', 'poslinear',
#            'posneglinear', 'print_results', 'process_prediction_file',
#            'protoCloud', 'prp', 'rank_HRG', 'relevance_filter',
#            'relu_wrapper_fct', 'resetbn', 'rule_map', 'run_model',
#            'safe_divide', 'save_file', 'save_model', 'save_model_dict',
#            'save_model_w_condition', 'save_prototype_cells', 'scRNAData',
#            'scRNAdata', 'seed_torch', 'set_value', 'sim_score_eps_wrapper_fct',
#            'tensorlist_todict', 'train', 'twoparam_wrapper_class', 'utils',
#            'viz', 'x_lrp', 'x_prp1', 'z_recon', 'zeroparam_wrapper_class']
