from ProtoCloud.viz import plot

from ProtoCloud.viz.plot import (EPS, device, get_UMAP, get_color, map_color,
                                 num_workers, plot_2latent,
                                 plot_all_class_lrp_dist,
                                 plot_cell_likelihoods, plot_confusion_matrix,
                                 plot_distance_to_prototypes, plot_epoch_trend,
                                 plot_gene_expression, plot_gene_rel_VS_expr,
                                 plot_gene_relevance, plot_latent_embedding,
                                 plot_lrp_dist, plot_marker_venn_diagram,
                                 plot_outlier_heatmap, plot_prediction_summary,
                                 plot_protocorr_heatmap, plot_protos_2latent,
                                 plot_protos_umap, plot_prp_dist,
                                 plot_top_gene_PRP_dotplot, plot_umap,)

__all__ = ['EPS', 'device', 'get_UMAP', 'get_color', 'map_color',
           'num_workers', 'plot', 'plot_2latent', 'plot_all_class_lrp_dist',
           'plot_cell_likelihoods', 'plot_confusion_matrix',
           'plot_distance_to_prototypes', 'plot_epoch_trend',
           'plot_gene_expression', 'plot_gene_rel_VS_expr',
           'plot_gene_relevance', 'plot_latent_embedding', 'plot_lrp_dist',
           'plot_marker_venn_diagram', 'plot_outlier_heatmap',
           'plot_prediction_summary', 'plot_protocorr_heatmap',
           'plot_protos_2latent', 'plot_protos_umap', 'plot_prp_dist',
           'plot_top_gene_PRP_dotplot', 'plot_umap']
