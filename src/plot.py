from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.pyplot import show
from matplotlib.collections import QuadMesh
from matplotlib_venn import venn2
# import venn

from src.utils import *
import src.glo as glo
EPS = glo.get_value('EPS')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

#TODO: refer to Tangram: https://github.com/broadinstitute/Tangram/blob/master/tangram/plot_utils.py#L450


### plot functions
def plot_epoch_trend(epochs, results_dir, plot_dir, exp_code, **kwargs):
    # epochs, results_dir, plot_dir, exp_code, **kwargs
    # print(result_file)
    trend = load_file(results_dir, exp_code, '_trend.npy')
    
    epochs = range(0, epochs+1)
    save_path = plot_dir + exp_code + '_'

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, trend[1], label='Training Accuracy', color='#308695')
    if kwargs["model_validation"]:
        plt.plot(epochs, trend[2], label='Validation Accuracy', color='#E69D45')
    # plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + 'Accuracy trend.png', bbox_inches='tight')
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(epochs, trend[0], 'b-o')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    # plt.title('Training Loss vs. Epochs')
    plt.grid(True)
    plt.savefig(save_path + 'Training loss.png', bbox_inches='tight')
    plt.close()
    print("\tTrend plots saved")




def plot_cell_likelihoods(args, ll):
    # Calculate the average log-likelihood
    avg_ll = np.mean(ll)

    # Plot a histogram of the log-likelihoods
    plt.hist(ll, bins=50)
    plt.xlabel("Likelihood\nAverage likelihood: {:0.3f}".format(avg_ll))
    plt.ylabel("Frequency")
    # plt.title("Per-cell Likelihood Histogram")

    save_path = args.plot_dir + args.exp_code + '_' + 'likelihood.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("\tLikelihood plot saved")



def plot_confusion_matrix(args):
    predicted = load_file(args.results_dir, args.exp_code, '_pred.csv')
    orig_y = predicted['label']
    pred_y = predicted['pred1']

    # same labels
    if args.pretrain_model_pth is None:
        celltype_order = np.unique(pred_y)
        cm = confusion_matrix(orig_y, pred_y, labels = celltype_order)
        rep = classification_report(orig_y, pred_y, output_dict = True)
        macro_f1 = rep["macro avg"]["f1-score"]
        total_count = np.sum(cm, axis = 1)
        accuracy = np.trace(cm) / np.sum(cm)
        print('Accuracy={:0.3f}; Macro F1={:0.3f}'.format(accuracy, macro_f1))

        cm = cm.astype('float') / (cm.sum(axis = 1)[:, np.newaxis] + EPS) * 100
        cm = pd.DataFrame(cm, index = celltype_order, columns = celltype_order)
        cm["Total Count"] = total_count # add total count column

    
    # different labels
    else:
        model_labels = np.unique(pred_y)
        data_labels = np.unique(orig_y)

        mapping = pd.DataFrame(0, index=data_labels, columns=model_labels)
        for o in data_labels:
            mapping.loc[o] = predicted[predicted['label'] == o]['pred1'].value_counts()
        mapping.fillna(0, inplace=True)

        total_count = mapping.sum(axis = 1)
        cm = mapping / (mapping.sum(axis = 0) + EPS) * 100
        cm["Total Count"] = total_count
        

    # plot
    figsize = (cm.shape[1]/3 + 3, cm.shape[0]/3 + 4)
    f, ax = plt.subplots(1, 1, figsize = figsize)
    sns.heatmap(cm, 
        annot=True,
        annot_kws={"fontsize":7},
        vmin = 0,
        vmax = 100,
        fmt='.0f',
        cmap = "Blues",
        square=True if mapping.shape[0] == mapping.shape[1] else False,
        linewidths = 0.5,
        cbar=False,
        )

    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    # title = " ".join([args.model_name, 'Confusion Matrix on', args.dataset_name])
    # plt.title(title, fontsize = 20)

    plt.tight_layout()
    plt.ylabel('Data label')
    if args.pretrain_model_pth is None:
        plt.xlabel('Predicted label\naccuracy={:0.3f}; Macro F1={:0.3f}'.format(accuracy, macro_f1))
    else:
        plt.xlabel('Predicted label')

    # plt.show()
    plt.savefig(args.plot_dir + args.exp_code + '_cm.png', bbox_inches='tight')
    plt.close()
    print("\tConfusion matrix saved")
    



def plot_latent_embedding(latent_embedding, proto_embedding=None, 
                        pred=None, orig=None, proto_classes=None,
                        plot_dim: Optional[int] = None,
                        prototypes_per_class: Optional[int] = 6,
                        all_color = None,
                        path: Optional[str] = None, 
                        **kwargs):
    """
    Plot the latent embedding of the data.
    args:
        latent_embedding
            The data that we are finding an embedding for, shape (n,d)
        proto_embedding
            The prototypes of the model, shape (k,d)
        
    """ 
    if plot_dim != 2:
        path_ending = "_umap.png"
        umap_kwargs = {'n_neighbors': 10, 'min_dist': 0.05}
        embedding = umap.UMAP(**umap_kwargs).fit_transform(np.concatenate((latent_embedding, proto_embedding), axis = 0))
        latent_embedding = embedding[:latent_embedding.shape[0], :]
        proto_embedding = embedding[latent_embedding.shape[0]:, :] if proto_embedding is not None else None
    else:
        path_ending = "_2latent.png"
        latent_embedding = latent_embedding[:, 0:2]
        proto_embedding = proto_embedding[:, 0:2] if proto_embedding is not None else None


    # color cells by groundtruth unless DNE use model labels
    if orig is not None:
        cell_labels = np.unique(orig)
        y = orig
    else:
        cell_labels = np.unique(pred)
        y = pred
        print("\tNo ground truth, color cell embedding by prediction")
    # if plot prototypes, use model labels
    if proto_embedding is not None:
        model_labels = proto_classes
    else:
        model_labels = []
    all_labels = set(set(list(np.unique(model_labels))+list(np.unique(cell_labels))))
    
    if all_color is None:
        cmap = plt.cm.nipy_spectral
        norm = plt.Normalize(0, len(all_labels)-1)
        all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
        all_color = {}
        for i, c in enumerate(all_labels):
            all_color[c] = all_color_list[i]


    f, ax = plt.subplots(1, figsize = (14, 10))
    # plot latent embeddings, color according to orig labels
    for label in cell_labels:
        ax.scatter(
            *latent_embedding[y == label, :].T,
            s=3, # point size
            alpha=0.5,
            color = all_color[label],
            label = label,
            )
        x_center = np.mean(latent_embedding[y == label, 0])
        y_center = np.mean(latent_embedding[y == label, 1])
        ax.text(x_center*1.02, y_center*1.05,
                label, 
                color = all_color[label],
                fontsize = 6,
                style = 'oblique', 
                horizontalalignment='center',
                verticalalignment='top',
                wrap=True,
                bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
                )
    
    if proto_embedding is not None:
        # plot prototype embeddings, color according to prototype labels
        for t, label in enumerate(model_labels):
            count = prototypes_per_class * t
            for i in range(prototypes_per_class):
                ax.scatter(
                    *proto_embedding[count+i, :].T,
                    s=60,
                    linewidth=0.7,
                    edgecolors="k",
                    marker="o",
                    alpha=0.8,
                    color = all_color[label],
                )
            # add class label text at the center of each prototype embeddings
            x_center = np.mean(proto_embedding[count : count+prototypes_per_class, 0])
            y_center = np.mean(proto_embedding[count : count+prototypes_per_class, 1])
            ax.text(x_center*1.02, y_center*1.05,
                    label, 
                    color = all_color[label],
                    fontsize = 8,
                    style = "italic", 
                    horizontalalignment='center',
                    verticalalignment='top',
                    wrap=True,
                    bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
                    )

    legend_handles = []
    for label, color in all_color.items():
        handle = mlines.Line2D([], [], color=color, linestyle='None', marker='o', markersize=20, label=label)
        legend_handles.append(handle)
    ncol = 1 if len(all_labels) < 20 else len(all_labels)//20
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=ncol)

    ax = plt.gca()
    ax.set_xticks([])   # Hide the x and y axis ticks
    ax.set_yticks([])
    
    if path is None:
        plt.show()
    else:
        path += path_ending
        plt.subplots_adjust(right=0.7)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    print("\tUMAP embedding saved")


# def plot_umap_embedding(args, data, plot_dim=None, proto=None):
#     """
#     Plot the umap embedding of the data, color and label based on y.
#     args:
#         latent (np.array): The data that we are finding an embedding for, (n,d)
#         proto (np.array): The prototypes of the model, (k,d)
#         y (np.array): The labels of the data, (n,)
#         data (Data): The scRNAData object that contains the cell encoder and cell color
#     """ 
#     # latent, proto, orig, pred, args: latent_dim, prototypes_per_class, plot_dir, exp_code
#     # plot_dim=2: 2latent, proto=None: no prototypes
#     latent = load_file(args.results_dir, args.exp_code, '_latent.npy')[:, :args.latent_dim//2]
#     proto = load_file(args.results_dir, args.exp_code, '_prototypes.npy')[:, :args.latent_dim//2]
#     predicted = load_file(args.results_dir, args.exp_code, '_pred.csv')
#     y = predicted['label'] if 'label' in predicted.columns else predicted['pred1']   # actual label


#     umap_kwargs = {'n_neighbors': 10, 'min_dist': 0.05}

#     embedding = umap.UMAP(**umap_kwargs).fit_transform(np.concatenate((latent, proto), axis = 0))
#     latent_embedding = embedding[:latent.shape[0], :]
#     proto_embedding = embedding[latent.shape[0]:, :]

#     f, ax = plt.subplots(1, figsize = (14, 10))

#     if args.pretrain_model_pth is not None:
#         model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
#         model_labels = model_encoder.classes_
#     else:
#         model_labels = data.cell_encoder.classes_
#     print("model_labels: ", model_labels)
#     if data.Y is not None:
#         cell_labels = data.cell_encoder.classes_
#         print("cell_labels: ", cell_labels)
#     else:
#         cell_labels = model_labels
#         print("\tNo ground truth, color cell embedding by prediction")
#     all_labels = set(set(list(np.unique(model_labels))+list(np.unique(cell_labels))))
    
#     cmap = plt.cm.nipy_spectral
#     norm = plt.Normalize(0, len(all_labels)-1)
#     all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
#     all_color = {}
#     for i, c in enumerate(all_labels):
#         all_color[c] = all_color_list[i]

#     # plot latent embeddings, color according to orig labels
#     for label in cell_labels:
#         ax.scatter(
#             *latent_embedding[y == label, :].T,
#             s=3, # point size
#             alpha=0.5,
#             color = all_color[label],
#             label = label,
#             )
#         x_center = np.mean(latent_embedding[y == label, 0])
#         y_center = np.mean(latent_embedding[y == label, 1])
#         ax.text(x_center*1.02, y_center*1.05,
#                 label, 
#                 color = all_color[label],
#                 fontsize = 6,
#                 style = 'oblique', 
#                 horizontalalignment='center',
#                 verticalalignment='top',
#                 wrap=True,
#                 bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
#                 )
    
#     # plot prototype embeddings, color according to prototype labels
#     for t, label in enumerate(model_labels):
#         count = args.prototypes_per_class * t
#         for i in range(args.prototypes_per_class):
#             ax.scatter(
#                 *proto_embedding[count+i, :].T,
#                 s=60,
#                 linewidth=0.7,
#                 edgecolors="k",
#                 marker="o",
#                 alpha=0.8,
#                 color = all_color[label],
#             )
#         # add class label text at the center of each prototype embeddings
#         x_center = np.mean(proto_embedding[count : count+args.prototypes_per_class, 0])
#         y_center = np.mean(proto_embedding[count : count+args.prototypes_per_class, 1])
#         ax.text(x_center*1.02, y_center*1.05,
#                 label, 
#                 color = all_color[label],
#                 fontsize = 8,
#                 style = "italic", 
#                 horizontalalignment='center',
#                 verticalalignment='top',
#                 wrap=True,
#                 bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
#                 )
#     # title = " ".join([args.model_name, 'UMAP embedding on', args.dataset_name])
#     # ax.set_title(title)

#     legend_handles = []
#     for label, color in all_color.items():
#         handle = mlines.Line2D([], [], color=color, linestyle='None', marker='o', markersize=20, label=label)
#         legend_handles.append(handle)
#     ncol = 1 if len(all_labels) < 20 else len(all_labels)//20
#     ax.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=ncol)

#     ax = plt.gca()
#     ax.set_xticks([])   # Hide the x and y axis ticks
#     ax.set_yticks([])
    
#     path = args.plot_dir + args.exp_code + '_umap.png'
#     plt.subplots_adjust(right=0.7)
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     print("\tUMAP embedding saved")


# def plot_two_latents_embedding(args, data):
#     latent_embedding = load_file(args.results_dir, args.exp_code, '_latent.npy')[:, 0:2]
#     proto_embedding = load_file(args.results_dir, args.exp_code, '_prototypes.npy')[:, 0:2]
#     predicted = load_file(args.results_dir, args.exp_code, '_pred.csv')
#     y = predicted['label'] if 'label' in predicted.columns else predicted['pred1'] 

#     f, ax = plt.subplots(1, figsize = (14, 10))

#     if args.pretrain_model_pth is not None:
#         model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
#         model_labels = model_encoder.classes_
#     else:
#         model_labels = data.cell_encoder.classes_
#     if data.Y is not None:
#         cell_labels = data.cell_encoder.classes_
#         # print("cell_labels: ", cell_labels)
#     else:
#         cell_labels = model_labels
#         print("\tNo ground truth, color cell embedding by prediction")
#     all_labels = set(set(list(np.unique(model_labels))+list(np.unique(cell_labels))))
    
#     cmap = plt.cm.nipy_spectral
#     norm = plt.Normalize(0, len(all_labels)-1)
#     all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
#     all_color = {}
#     for i, c in enumerate(all_labels):
#         all_color[c] = all_color_list[i]
    
#     # plot latent embeddings, color according to orig labels
#     for label in cell_labels:
#         ax.scatter(
#             *latent_embedding[y == label, :].T,
#             s=5, # point size
#             alpha=0.5,
#             color = all_color[label],
#             label = label,
#             )
#         # x_center = np.mean(latent_embedding[y == label, 0])
#         # y_center = np.mean(latent_embedding[y == label, 1])
#         # ax.text(x_center*1.02, y_center*1.05,
#         #         label, 
#         #         # color = all_color[label],
#         #         fontsize = 12,
#         #         style = 'oblique', 
#         #         horizontalalignment='center',
#         #         verticalalignment='top',
#         #         wrap=True,
#         #         bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
#         #         )
    
#     # plot prototype embeddings, color according to prototype labels
#     for t, label in enumerate(model_labels):
#         count = args.prototypes_per_class * t
#         for i in range(args.prototypes_per_class ):
#             ax.scatter(
#                 *proto_embedding[count+i, :].T,
#                 s=60,
#                 linewidth=0.7,
#                 edgecolors="k",
#                 marker="o",
#                 color = all_color[label],
#             )
#         # # add class label text at the center of each prototype embeddings
#         # x_center = np.mean(proto_embedding[count:count+args.prototypes_per_class, 0])
#         # y_center = np.mean(proto_embedding[count:count+args.prototypes_per_class, 1])
#         # ax.text(x_center*1.02, y_center*1.05,
#         #         label, 
#         #         # color = all_color[label],
#         #         fontsize = 14,
#         #         style = "italic", 
#         #         horizontalalignment='center',
#         #         verticalalignment='top',
#         #         wrap=True,
#         #         bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
#         #         )
        
#     # title = "First 2 latents embedding on " + args.dataset_name
#     # ax.set_title(title)

#     legend_handles = []
#     for label, color in all_color.items():
#         handle = mlines.Line2D([], [], color=color, linestyle='None', marker='o', markersize=20, label=label)
#         legend_handles.append(handle)
#     ncol = 1 if len(all_labels) < 20 else len(all_labels)//20
#     ax.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=ncol)

#     ax = plt.gca()
#     ax.set_xticks([])   # Hide the x and y axis ticks
#     ax.set_yticks([])
    
#     path = args.plot_dir + args.exp_code + '_2latent.png'
#     plt.subplots_adjust(right=0.7)
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     print("\tLatent2 embedding saved")



def plot_protocorr_heatmap(args, data):
    # prototype correlation heatmap
    proto = load_file(args.results_dir, args.exp_code, "_prototypes.npy")

    df = pd.DataFrame(proto).T.corr()
    plt.figure(figsize = (8,8))
    ax = sns.heatmap(df,
            cmap = "Blues",
            square=True,
            cbar=False,
            )    
    ax.hlines(range(0, args.num_prototypes, args.prototypes_per_class), *ax.get_xlim(), color='white')
    ax.vlines(range(0, args.num_prototypes, args.prototypes_per_class), *ax.get_ylim(), color='white')
    # Celltype label for each prototype group
    celltypes = data.cell_encoder.classes_
    plt.xticks(np.arange(args.prototypes_per_class//2, args.num_prototypes, args.prototypes_per_class), 
                celltypes)
    plt.yticks(np.arange(args.prototypes_per_class//2, args.num_prototypes, args.prototypes_per_class), 
                celltypes)

    plt.ylabel('Prototypes', fontsize = 12)
    plt.xlabel('Prototypes', fontsize = 12)
    # title = " ".join([args.model_name, 'Prototype Correlation on', args.dataset_name])
    # plt.title(title)

    save_path = args.plot_dir + args.exp_code + '_protoCorr.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("\tPrototype correlation heatmap saved")



def plot_distance_to_prototypes(args, data):
    latents = load_file(args.results_dir, args.exp_code, '_latent.npy')
    protos = load_file(args.results_dir, args.exp_code, '_prototypes.npy')
    predicted = load_file(args.results_dir, args.exp_code, '_pred.csv')
    pred = predicted['pred1'].values

    classes = data.cell_encoder.classes_
    all_distances = {}
    variances = {}
    means = {}
    class_colors = {}

    palette = sns.color_palette("hsv", len(classes))
    plt.figure(figsize=(20, 5))
    for i, c in enumerate(classes):
        latent = latents[pred == c, :args.latent_dim // 2]
        proto = protos[i * args.prototypes_per_class: (i + 1) * args.prototypes_per_class, :args.latent_dim // 2]
        distances = np.abs(scipy.spatial.distance.cdist(latent, proto))
        distances = np.min(distances, axis=1)
        all_distances[c] = distances
        variances[c] = np.var(distances)
        means[c] = np.mean(distances)
        
        sns.kdeplot(distances, label=c, fill=False, color=palette[i], lw=2, alpha=0.8)
        class_colors[c] = palette[i]
    # # label class with min/max variance
    # minvar_class = min(variances, key=variances.get)
    # plt.annotate(minvar_class, 
    #              xy=(-1.5, 1.5), color=class_colors[minvar_class])
    # maxmean_class = max(means, key=means.get)
    # plt.annotate(maxmean_class, 
    #              xy=(np.max(means[maxmean_class])*1.25, 0.75), color=class_colors[maxmean_class])

    ncol = 1 if len(classes) < 15 else len(classes)//15
    plt.legend(title='Celltype', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol)

    ax = plt.gca()
    ax.spines['top'].set_visible(False) # Hide the x and y axis ticks
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])   # Hide the x and y axis ticks
    ax.set_yticks([])
    title = args.dataset_name + ' Distance Distribution to Prototypes'
    # plt.title(title)
    plt.xlabel('Distance to Prototype')
    plt.ylabel('Density')

    path = args.plot_dir + args.exp_code + '_distanceDist.png'
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("\tDistance distribution to prototypes saved")

# Call the function with appropriate arguments
# plot_distance_to_prototypes(args, data)





# PRP & LRP related plots
#######################################################
def plot_prp_dist(celltypes, gene_names, 
                num_classes, prototypes_per_class,
                prp_path, exp_code, **kwargs):
    """
    Plot the histogram of PRP scores of all genes for each cell type
    """
    filename = '_' + exp_code + '_relgenes.npy'

    for i in range(num_classes):
        file1 = celltypes[i].replace("/", " OR ") + filename
        if os.path.exists(prp_path + file1):
            proto_rel = np.load(prp_path + file1, allow_pickle = True)
        else:
            print(f"\t{prp_path + file1} does not exist.")
            continue

        # Create a figure with 6 subplots (one for each prototype) in one row
        fig, axes = plt.subplots(prototypes_per_class, 1, 
                                 figsize=(20, 5*prototypes_per_class))

        for p in range(prototypes_per_class):
            gene_rel = proto_rel[p, :]
            ax = axes[p]

            # Plot per prototype on subplot
            ax.plot(list(range(len(gene_rel))), gene_rel)
            ax.set_title("{}: Prototype {} Gene Relevance Score".format(celltypes[i], p), fontsize=10)
            ax.set_ylabel('Relevance Score', fontsize=8)
            ax.set_xlabel('Gene Indices', fontsize=8)

            # Identify and annotate top genes
            df = pd.DataFrame(data=gene_rel).T
            top_genes_indices = df.sum().nlargest(20).index.tolist()
            top_genes = gene_names[top_genes_indices].tolist()

            for j, value in enumerate(top_genes_indices):
                ax.annotate(top_genes[j],
                            (value, gene_rel[value]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=14)
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False) # Hide the x and y axis ticks
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])   # Hide the x and y axis ticks
            ax.set_yticks([])
        
        plt.savefig(prp_path + celltypes[i].replace("/", " OR ") + '_prpRelavance.png', bbox_inches='tight')
        plt.close()
    print("\tPRP relavance genes saved")



def plot_lrp_dist(celltypes, gene_names, num_classes, lrp_path, exp_code, **kwargs):
    """
    Plot the histogram of LRP scores of all genes for each cell type
    """
    filename = '_' + exp_code + '_relgenes.npy'

    df_top_genes = pd.DataFrame()

    for i in range(num_classes):
        file1 = celltypes[i].replace("/", " OR ") + filename
        if os.path.exists(lrp_path + file1):
            gene_rel = np.load(lrp_path + file1, allow_pickle = True)
        else:
            print(f"\t{lrp_path + file1} does not exist.")
            continue

        plt.figure(figsize = (20, 5))
        plt.plot(list(range(0, len(gene_rel))), gene_rel)

        title = "{}: Gene LRP Relavance Score".format(celltypes[i])
        plt.title(title, fontsize = 20)
        plt.tight_layout()
        plt.ylabel('Relevance Score', fontsize = 16)
        plt.xlabel('Gene Indices', fontsize = 16)

        df = pd.DataFrame(data = gene_rel).T
        top_genes_indices = df.sum().nlargest(20).index.tolist()
        top_genes = gene_names[top_genes_indices].tolist()

        # Adding annotation on top of the bars in the graph
        for j, value in enumerate(top_genes_indices):
            plt.annotate(top_genes[j],
                        (value, gene_rel[value]),
                        textcoords="offset points",
                        xytext = (0, 10),
                        ha='center',
                        fontsize = 14)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False) # Hide the x and y axis ticks
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])   # Hide the x and y axis ticks
        ax.set_yticks([])
    
        plt.savefig(lrp_path + celltypes[i].replace("/", " OR ") + '_lrpRelavance.png', bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(data = gene_rel).T
        top_genes_indices = df.sum().nlargest(50).index.tolist()
        top_genes = gene_names[top_genes_indices].tolist()
        df_top_genes = pd.concat([df_top_genes, pd.Series(top_genes, name = celltypes[i])], axis = 0)

    # Save DataFrame to csv
    save_file(df_top_genes, lrp_path, exp_code, '_LRPgenes')
    print("\tPRP relavance genes saved")
        


def plot_top_gene_PRP_dotplot(celltypes, gene_names, num_classes,
                    prototypes_per_class, 
                    results_dir, exp_code,
                    num_protos:int = 1,
                    top_num_genes:int = 10,   # num of top rel genes from each class
                    celltype_specific = True, # True: non-overlapping top rel genes
                    save_markers = False, # save top rel markers
                    **kwargs):
    num_protos = prototypes_per_class if num_protos > prototypes_per_class else num_protos
    prp_path = results_dir + "prp/"
    filename = '_' + exp_code + "_relgenes.npy"

    # Identify top k genes from each class
    all_rel = []
    top_genes = []
    for i in range(len(celltypes)):
        proto_rel = np.load(prp_path + celltypes[i].replace("/", " OR ") + filename, allow_pickle=True)
        cls_top_genes = []
        for p in range(num_protos):
            gene_rel = proto_rel[p, :]
            df = pd.DataFrame(data=gene_rel).T
            tgi = df.sum().nlargest(top_num_genes).index.tolist()
            cls_top_genes = mutual_genes(cls_top_genes, gene_names[tgi].tolist(), False)
            all_rel.append(gene_rel)
        top_genes = mutual_genes(top_genes, cls_top_genes, celltype_specific)
    all_rel = np.stack(all_rel, axis = 0) 
    print(len(top_genes), all_rel.shape)
    if len(top_genes) == 0:
        print("No cell-type specific marker found")
        return
    
    if save_markers:
        if celltype_specific:
            filename = results_dir + exp_code+ "_novel_markers.txt"
        else:
            filename = results_dir + exp_code + "_novel_markers_full.txt"
        with open(filename, 'w') as file:
            for item in top_genes:
                file.write(f"{item}\n")
        print(f"\tTop rel genes saved to {filename}")
    
    
    all_rel = minmax_scale_matrix(all_rel)
    
    lrp = anndata.AnnData(X=all_rel)
    lrp.var["gene_name"] = gene_names
    lrp.var_names = gene_names
    if num_protos != 1:
        lrp.obs['row'] = [f"{x}_{i+1}" for x in celltypes for i in range(num_protos)]
    else:
        lrp.obs['row'] = celltypes
    lrp.var.index = gene_names
    
    
    file_path = prp_path + exp_code + "_celltype_order.txt"
    if os.path.exists(file_path):
        celltype_order = pd.read_csv(file_path, header=None).iloc[:,0].tolist()
        if num_protos != 1:
            orders = [f"{x}_{i+1}" for x in celltype_order for i in range(num_protos)]
        else:
            orders = [gene for gene in celltype_order if gene in lrp.obs['row'].values]
        # print("shared_celltypes:", len(orders))
    else:
        orders = None
    
    if num_protos == 1:
        if celltype_specific:
            path = results_dir + "plots/" + exp_code + "_PRP_dotplot.png"
        else:
            path = results_dir + "plots/" + exp_code + "_PRP_dotplot_full.png"
    else:
        path = results_dir + "plots/" + exp_code + "_PRP_L_dotplot.png"
    get_dotplot(lrp[:,top_genes], top_genes, groupby="row", celltype_order=orders, path=path)
    print("\tTop rel genes PRP dotplot saved")



def plot_outlier_heatmap(celltypes, gene_names,
                        num_classes,
                        lrp_path, exp_code, **kwargs):
    filename1 = "_" + exp_code + "_relgenes.npy"
    filename2 = "_" + exp_code + "_lrp.npy"

    for i in range(num_classes):
        # get top genes
        sum_gene_rel = np.load(lrp_path + celltypes[i].replace("/", " OR ") + filename1, allow_pickle = True)
        df = pd.DataFrame(data = sum_gene_rel).T
        marker_idx = df.sum().nlargest(20).index.tolist()
        # get cellwise lrp values
        gene_rel = np.load(lrp_path + celltypes[i].replace("/", " OR ") + filename2, allow_pickle = True)
        df = pd.DataFrame(gene_rel, columns = gene_names).iloc[:, marker_idx].T

        ax = sns.clustermap(df,
                    cmap = "Blues_r",
                    cbar_pos = None,
                    row_cluster = False,
                    dendrogram_ratio = (0, .2)
                    )

        plt.ylabel('Top LRP Genes')
        plt.xlabel('Cell Index')
        ax = plt.gca()
        ax.set_xticks([])   # Hide the x and y axis ticks
        ax.set_yticks([])
        # plt.title(celltypes[i] + ' Outlier Heatmap')
        plt.tight_layout()
        plt.savefig(lrp_path + celltypes[i].replace("/", " OR ") + '_outlier_heatmap.png', bbox_inches='tight')
        plt.close()

    print("\tCelltype outlier heatmap saved")


def plot_marker_venn_diagram(adata, num_classes,
                        prp_path, exp_code, **kwargs):
    """
    Plot the venn diagram of top HVG and top prp genes for each cell type
    """
    celltypes = np.unique(adata.obs['celltype'])
    gene_names = adata.var['gene_name']

    # top HVG genes
    sc.pp.log1p(adata)
    adata.uns['log1p']["base"] = None
    sc.tl.rank_genes_groups(adata, groupby = 'celltype', use_raw = False, n_genes = 50, method = 'wilcoxon')
    top_HVG = adata.uns['rank_genes_groups']['names']

    # top lrp marker genes
    markers = {}
    for i in range(num_classes):
        file1 = celltypes[i].replace("/", " OR ") + "_" + exp_code + "_relgenes.npy"
        gene_rel = np.load(prp_path + file1, allow_pickle=True)
        df = pd.DataFrame(data=gene_rel)
        top_genes_indices = df.sum().nlargest(20).index.tolist()
        markers[celltypes[i]] = set(gene_names[top_genes_indices])

    col = 4
    row = math.ceil(num_classes / col)
    fig, axs = plt.subplots(row, col, figsize=(16, 3 * row))

    # Iterate over each subplot
    for i in range(num_classes):
        ax = axs[i // col][i % col]
        venn2([markers[celltypes[i]], set(top_HVG[celltypes[i]])],
            set_labels = ('Relevant Genes', 'HVG'),
            set_colors = ('purple', 'skyblue'),
            ax = ax)
        ax.set_title(celltypes[i])
        ax = plt.gca()
        ax.set_xticks([])   # Hide the x and y axis ticks
        ax.set_yticks([])
        

    # plt.suptitle('Overlapping Marker Gene Venn Diagram')
    plt.tight_layout()
    plt.savefig(prp_path + 'DE_marker_venn.png', bbox_inches='tight')
    plt.close()

    print("\tDE marker genes venn diagram saved")

