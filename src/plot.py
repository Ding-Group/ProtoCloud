import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import umap
import seaborn as sns
import matplotlib.pyplot as plt
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
def plot_epoch_trend(args):
    # print(result_file)
    trend = load_file(args, '_trend.npy')
    
    epochs = range(0, args.epochs+1)
    save_path = args.plot_dir + args.exp_code + '_'

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, trend[1], label='Training Accuracy', color='#308695')
    plt.plot(epochs, trend[2], label='Validation Accuracy', color='#E69D45')
    plt.title('Accuracy vs. Epochs')
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
    plt.title('Training Loss vs. Epochs')
    plt.grid(True)
    plt.savefig(save_path + 'Training loss.png', bbox_inches='tight')
    plt.close()
    print("Trend plots saved")





def plot_cell_likelihoods(args, ll):
    # Calculate the average log-likelihood
    avg_ll = np.mean(ll)

    # Plot a histogram of the log-likelihoods
    plt.hist(ll, bins=50)
    plt.xlabel("Likelihood\nAverage likelihood: {:0.3f}".format(avg_ll))
    plt.ylabel("Frequency")
    plt.title("Per-cell Likelihood Histogram")

    save_path = args.plot_dir + args.exp_code + '_' + 'likelihood.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Likelihood plot saved")



def plot_confusion_matrix(args, target_names):
    misclass_rate, rep, cm = load_file(args, '_metrics.npy')
    macro_f1 = rep["macro avg"]["f1-score"]
    total_count = np.sum(cm, axis = 1) # actual cell type count
    accuracy = 1 - misclass_rate
    # normalize
    cm = cm.astype('float') / (cm.sum(axis = 1)[:, np.newaxis] + EPS)

    cm = pd.DataFrame(cm, index = target_names, columns = target_names)
    cm["Total Count"] = total_count # add total count column

    # plot
    n = 15 if len(target_names) >= 15 else 8
    f, ax = plt.subplots(1, 1, figsize = (n,n))

    sns.heatmap(cm, 
        annot=True,
        vmin = 0,
        vmax = 1,
        fmt='.2f',
        cmap = "Blues",
        square=True,
        yticklabels=target_names,
        linewidths = 0.5,
        cbar=False,
        annot_kws={"size": 35 / np.sqrt(len(target_names))},
        )

    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    title = " ".join([args.model, 'Confusion Matrix on', args.dataset])
    plt.title(title, fontsize = 20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label\naccuracy={:0.3f}; Macro F1={:0.3f}'.format(accuracy, macro_f1), fontsize = 16)

    # plt.show()
    plt.savefig(args.plot_dir + args.exp_code + '_cm.png', bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved")


def plot_umap_embedding(args, y, data):
    """
    Plot the umap embedding of the data, color and label based on y. Call matplotlib.pylot.show() after to display plot.
    args:
        latent (np.array): The data that we are finding an embedding for, (n,d)
        proto (np.array): The prototypes of the model, (k,d)
        y (np.array): The labels of the data, (n,)
        data (Data): The scRNAData object that contains the cell encoder and cell color
    """ 
    latent = load_file(args, '_latent.npy')
    proto = load_file(args, '_prototypes.npy')

    umap_kwargs = {'n_neighbors': 10, 'min_dist': 0.05}

    # embedding = umap.UMAP(**umap_kwargs).fit_transform(latent)
    embedding = umap.UMAP(**umap_kwargs).fit_transform(np.concatenate((latent, proto), axis = 0))
    latent_embedding = embedding[:latent.shape[0], :]
    proto_embedding = embedding[latent.shape[0]:, :]
    cell_labels = data.cell_encoder.classes_

    f, ax = plt.subplots(1, figsize = (14, 10))

    types = np.unique(y)
    cmap = plt.cm.nipy_spectral
    norm = plt.Normalize(np.min(types), np.max(types))
    cell_color = cmap(norm(types)) if data.cell_color is None else data.cell_color

    # plot latent embeddings
    for type in types:
        ax.scatter(
            *latent_embedding[type == y, :].T,
            s=1, # point size
            alpha=0.5,
            color =  cell_color[type],
            label = cell_labels[type],
            )
    for type in types:
        count = args.prototypes_per_class*type
        # plot prototype embeddings
        for i in range(0,args.prototypes_per_class):
            ax.scatter(
                *proto_embedding[count+i, :].T,
                s=50,
                linewidth=0.7,
                edgecolors="k",
                marker="o",
                alpha=0.8,
                color = cell_color[type],
            )
        # add class label text at the center of each prototype embeddings
        x_center = np.mean(proto_embedding[count:count+i, 0])
        y_center = np.mean(proto_embedding[count:count+i, 1])
        ax.text(x_center*1.02, y_center*1.05,
                cell_labels[type], 
                color = cell_color[type],
                fontsize = 8,
                style = "italic", 
                horizontalalignment='center',
                verticalalignment='top',
                wrap=True,
                bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
                )
        
    title = " ".join([args.model, 'UMAP embedding on', args.dataset])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    path = args.plot_dir + args.exp_code + '_umap.png'
    plt.subplots_adjust(right=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("UMAP embedding saved")


def plot_protocorr_heatmap(args):
    proto = load_file(args, "_prototypes.npy")
    # prototype correlation heatmap
    df = pd.DataFrame(proto).T.corr()
    # df.columns = celltypes

    plt.figure(figsize = (8,8))
    ax = sns.heatmap(df,
            cmap = "Blues",
            square=True,
            cbar=False,
            )

    ax.hlines(range(0, args.num_prototypes, args.prototypes_per_class), *ax.get_xlim(), color='white')
    ax.vlines(range(0, args.num_prototypes, args.prototypes_per_class), *ax.get_ylim(), color='white')
    plt.ylabel('Prototypes', fontsize = 14)
    plt.xlabel('Prototypes', fontsize = 14)
    title = " ".join([args.model, 'Prototype Correlation on', args.dataset])
    plt.title(title)
    
    save_path = args.plot_dir + args.exp_code + '_protoCorr.png'
    plt.savefig(save_path)
    plt.close()
    print("Prototype correlation heatmap saved")



def plot_misclassified_umap(args, data):
    """
    Plot the umap embedding of the data, color and label based on y. Call matplotlib.pylot.show() after to display plot.
    args:
        latent (np.array): The data that we are finding an embedding for, (n,d)
        proto (np.array): The prototypes of the model, (k,d)
        y (np.array): The labels of the data, (n,)
        data (Data): The scRNAData object that contains the cell encoder and cell color
    """ 
    latent = load_file(args, '_latent.npy')
    proto = load_file(args, '_prototypes.npy')
    predictions = load_file(args, '_pred.csv')
    umap_kwargs = {'n_neighbors': 10, 'min_dist': 0.05}

    # only plot those that are misclassified
    actual_y = predictions['celltype']
    pred1 = predictions['pred1']
    pred2 = predictions['pred2']

    latent = latent[actual_y != pred1]

    # embedding = umap.UMAP(**umap_kwargs).fit_transform(latent)
    embedding = umap.UMAP(**umap_kwargs).fit_transform(np.concatenate((latent, proto), axis = 0))
    latent_embedding = embedding[:latent.shape[0], :]
    proto_embedding = embedding[latent.shape[0]:, :]
    cell_labels = data.cell_encoder.classes_

    f, ax = plt.subplots(1, figsize = (14, 10))

    types = np.unique(actual_y)
    cmap = plt.cm.nipy_spectral
    norm = plt.Normalize(np.min(types), np.max(types))
    cell_color = cmap(norm(types)) if data.cell_color is None else data.cell_color

    # plot latent embeddings
    for type in types:
        ax.scatter(
            *latent_embedding[type == actual_y, :].T,
            s=10, # point size
            alpha=0.5,
            color =  cell_color[type],
            label = cell_labels[type],
            )
    for type in types:
        count = args.prototypes_per_class*type
        # plot prototype embeddings
        for i in range(0,args.prototypes_per_class):
            ax.scatter(
                *proto_embedding[count+i, :].T,
                s=50,
                linewidth=0.7,
                edgecolors="k",
                marker="o",
                alpha=0.8,
                color = cell_color[type],
            )
        # add class label text at the center of each prototype embeddings
        x_center = np.mean(proto_embedding[count:count+i, 0])
        y_center = np.mean(proto_embedding[count:count+i, 1])
        ax.text(x_center*1.02, y_center*1.05,
                cell_labels[type], 
                color = cell_color[type],
                fontsize = 8,
                style = "italic", 
                horizontalalignment='center',
                verticalalignment='top',
                wrap=True,
                bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
                )
        
    title = " ".join([args.model, 'UMAP embedding on', args.dataset])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    path = args.plot_dir + args.exp_code + '_umap.png'
    plt.subplots_adjust(right=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("UMAP embedding saved")



# LRP related plots
#######################################################
def plot_lrp_dist(data, args):
    """
    Plot the histogram of LRP scores of all genes for each cell type
    """
    lrp_path = args.lrp_path
    celltypes = data.cell_encoder.classes_
    gene_names = pd.Series(data.gene_names)
    filename = '_' + args.exp_code + '_relgenes.npy'

    df_top_genes = pd.DataFrame()

    # Open the file in write mode and create/overwrite the existing file
    # with open(lrp_path+args.exp_code+'_LRPgenes.txt', 'w') as file:
    for i in range(args.num_classes):
        file1 = celltypes[i] + filename
        gene_rel = np.load(lrp_path + file1, allow_pickle = True) # genes

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
            plt.annotate(top_genes[j], # this is the text
                        (value, gene_rel[value]), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext = (0, 10), # distance from text to points (x,y)
                        ha='center', # horizontal alignment can be left, right or center
                        fontsize = 12)
    
        plt.savefig(lrp_path + celltypes[i] + '_lrpRelavance.png', bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(data = gene_rel).T
        top_genes_indices = df.sum().nlargest(50).index.tolist()
        top_genes = gene_names[top_genes_indices].tolist()
        # file.write(celltypes[i] + "\n")
        # [file.write(item + '\t') for item in top_genes]
        # file.write('\n')
        df_top_genes = pd.concat([df_top_genes, pd.Series(top_genes, name = celltypes[i])], axis = 0)

    # Save DataFrame to csv
    save_file(df_top_genes, args, '_LRPgenes', lrp_path)
        


def plot_top_gene_heatmap(data, args):
    """
    Plot the heatmap of top lrp genes across all cell type
    """
    lrp_path = args.lrp_path
    celltypes = data.cell_encoder.classes_
    gene_names = pd.Series(data.gene_names)
    filename = '_' + args.exp_code + '_relgenes.npy'

    all_marker_idx = []
    for i in range(args.num_classes):
        file1 = celltypes[i] + filename
        gene_rel = np.load(lrp_path + file1, allow_pickle = True) # genes
        
        df = pd.DataFrame(data = gene_rel).T
        all_marker_idx += df.sum().nlargest(20).index.tolist()

    all_marker_idx = list(set(all_marker_idx))

    marker_value = []
    for i in range(args.num_classes):
        file1 = celltypes[i]+ filename
        gene_rel = np.load(lrp_path + file1, allow_pickle = True) # genes
        marker_value.append(gene_rel[all_marker_idx])

    all_marker_df = pd.DataFrame(marker_value, index = celltypes, columns = gene_names[all_marker_idx])

    plt.figure(figsize = (len(all_marker_idx) // 6, args.num_classes // 3))
    ax = sns.heatmap(all_marker_df,
            linewidths = 0.5,
            cmap = "Blues_r")
    
    plt.xlabel('Gene Name', fontsize = 14)
    plt.ylabel('Cell Types', fontsize = 14)
    title = "Cell Type - Marker Gene Heatmap"
    plt.title(title, fontsize = 20)
    plt.tight_layout()

    plt.savefig(lrp_path + 'celltype_markergene_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Cell type - marker gene heatmap saved")


def plot_outlier_heatmap(data, args):
    lrp_path = args.lrp_path
    celltypes = data.cell_encoder.classes_
    gene_names = pd.Series(data.gene_names)
    filename1 = "_" + args.exp_code + "_relgenes.npy"
    filename2 = "_" + args.exp_code + "_lrp.npy"

    for i in range(args.num_classes):
        # get top genes
        sum_gene_rel = np.load(lrp_path + celltypes[i] + filename1, allow_pickle = True)
        df = pd.DataFrame(data = sum_gene_rel).T
        marker_idx = df.sum().nlargest(20).index.tolist()
        # get cellwise lrp values
        gene_rel = np.load(lrp_path + celltypes[i] + filename2, allow_pickle = True)
        df = pd.DataFrame(gene_rel, columns = gene_names).iloc[:, marker_idx].T

        ax = sns.clustermap(df,
                    linewidths = .1,
                    cmap = "Blues_r",
                    cbar_pos = None,
                    row_cluster = False,
                    dendrogram_ratio = (0, .2)
                    )

        plt.ylabel('Top LRP Genes')
        plt.xlabel('Cell Index')
        plt.title(celltypes[i] + ' Outlier Heatmap')
        plt.tight_layout()
        plt.savefig(lrp_path + celltypes[i] + '_outlier_heatmap.png', bbox_inches='tight')
        plt.close()

    print("Celltype outlier heatmap saved")


def plot_marker_venn_diagram(adata, args):
    """
    Plot the venn diagram of top DE genes and top lrp genes for each cell type
    """
    lrp_path = args.lrp_path
    num_classes = args.num_classes
    celltypes = np.unique(adata.obs['celltype'])
    gene_names = adata.var['gene_name']

    # top DE genes
    adata.uns['log1p']["base"] = None
    sc.pp.log1p(adata.layers["counts"])
    sc.tl.rank_genes_groups(adata, groupby = 'celltype', use_raw = False, layer = 'counts', n_genes = 50, method = 'wilcoxon')
    top_de_genes = adata.uns['rank_genes_groups']['names']

    # top lrp marker genes
    markers = {}
    for i in range(num_classes):
        file1 = celltypes[i] + "_" + args.exp_code + "_relgenes.npy"
        gene_rel = np.load(lrp_path + file1, allow_pickle=True)
        df = pd.DataFrame(data = gene_rel).T
        top_genes_indices = df.sum().nlargest(20).index
        markers[celltypes[i]] = set(gene_names[top_genes_indices])

    col = 4
    row = math.ceil(num_classes / col)
    fig, axs = plt.subplots(row, col, figsize=(16, 3 * row))

    # Iterate over each subplot
    for i in range(num_classes):
        ax = axs[i // col][i % col]
        venn2([markers[celltypes[i]], set(top_de_genes[celltypes[i]])],
            set_labels = ('Marker Genes', 'DE Genes'),
            set_colors = ('purple', 'skyblue'),
            ax = ax)
        ax.set_title(celltypes[i])
        
    plt.suptitle('Overlapping Marker Gene Venn Diagram')
    plt.tight_layout()
    plt.savefig(lrp_path + 'DE_marker_venn.png', bbox_inches='tight')
    plt.close()

    print("DE marker genes venn diagram saved")

