import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
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
    # epochs, plot_dir, exp_code, **kwargs
    # print(result_file)
    trend = load_file(args, '_trend.npy')
    
    epochs = range(0, args.epochs+1)
    save_path = args.plot_dir + args.exp_code + '_'

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, trend[1], label='Training Accuracy', color='#308695')
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
    predicted = load_file(args, '_pred.csv')
    orig_y = predicted['celltype']
    pred_y = predicted['idx1']

    unique_types = np.unique(np.concatenate((orig_y, pred_y)))
    cm = confusion_matrix(orig_y, pred_y, labels = unique_types)
    rep = classification_report(orig_y, pred_y, output_dict = True)
    macro_f1 = rep["macro avg"]["f1-score"]
    total_count = np.sum(cm, axis = 1)
    accuracy = np.trace(cm) / np.sum(cm)

    cm = cm.astype('float') / (cm.sum(axis = 1)[:, np.newaxis] + EPS) # normalize
    cm = pd.DataFrame(cm, index = unique_types, columns = unique_types)
    cm["Total Count"] = total_count # add total count column

    # plot
    n = 15 if len(unique_types) >= 15 else 8
    f, ax = plt.subplots(1, 1, figsize = (n,n))

    sns.heatmap(cm, 
        annot=True,
        vmin = 0,
        vmax = 1,
        fmt='.2f',
        cmap = "Blues",
        square=True,
        yticklabels=unique_types,
        linewidths = 0.5,
        cbar=False,
        annot_kws={"size": 35 / np.sqrt(len(unique_types))},
        )

    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    # title = " ".join([args.model, 'Confusion Matrix on', args.dataset])
    # plt.title(title, fontsize = 20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label\naccuracy={:0.3f}; Macro F1={:0.3f}'.format(accuracy, macro_f1), fontsize = 16)

    # plt.show()
    plt.savefig(args.plot_dir + args.exp_code + '_cm.png', bbox_inches='tight')
    plt.close()
    print("\tConfusion matrix saved")
    print('Accuracy={:0.3f}; Macro F1={:0.3f}'.format(accuracy, macro_f1))



def plot_umap_embedding(args, data):
    """
    Plot the umap embedding of the data, color and label based on y.
    args:
        latent (np.array): The data that we are finding an embedding for, (n,d)
        proto (np.array): The prototypes of the model, (k,d)
        y (np.array): The labels of the data, (n,)
        data (Data): The scRNAData object that contains the cell encoder and cell color
    """ 
    latent = load_file(args, '_latent.npy')[:, :args.latent_dim//2]
    proto = load_file(args, '_prototypes.npy')[:, :args.latent_dim//2]
    predicted = load_file(args, '_pred.csv')
    y = predicted['celltype'] if 'celltype' in predicted.columns else predicted['idx1']   # actual label


    umap_kwargs = {'n_neighbors': 10, 'min_dist': 0.05}

    embedding = umap.UMAP(**umap_kwargs).fit_transform(np.concatenate((latent, proto), axis = 0))
    latent_embedding = embedding[:latent.shape[0], :]
    proto_embedding = embedding[latent.shape[0]:, :]

    f, ax = plt.subplots(1, figsize = (14, 10))

    model_labels = data.cell_encoder.classes_
    cell_labels = np.unique(y)
    all_labels = set(set(list(np.unique(model_labels))+list(np.unique(cell_labels))))
    
    cmap = plt.cm.nipy_spectral
    norm = plt.Normalize(0, len(all_labels)-1)
    all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
    all_color = {}
    for i, c in enumerate(all_labels):
        all_color[c] = all_color_list[i]

    # plot latent embeddings, color according to orig labels
    for label in cell_labels:
        ax.scatter(
            *latent_embedding[y == label, :].T,
            s=3, # point size
            alpha=0.5,
            color = all_color[label],
            label = label,
            )
        # if args.model_data is not None:
        #     x_center = np.mean(latent_embedding[y == label, 0])
        #     y_center = np.mean(latent_embedding[y == label, 1])
        #     ax.text(x_center*1.02, y_center*1.05,
        #             label, 
        #             # color = all_color[label],
        #             fontsize = 12,
        #             style = 'oblique', 
        #             horizontalalignment='center',
        #             verticalalignment='top',
        #             wrap=True,
        #             bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
        #             )
    
    # plot prototype embeddings, color according to prototype labels
    for t, label in enumerate(model_labels):
        count = args.prototypes_per_class * t
        for i in range(args.prototypes_per_class):
            ax.scatter(
                *proto_embedding[count+i, :].T,
                s=60,
                linewidth=0.7,
                edgecolors="k",
                marker="o",
                alpha=0.8,
                color = all_color[label],
            )
        # # add class label text at the center of each prototype embeddings
        # x_center = np.mean(proto_embedding[count : count+args.prototypes_per_class, 0])
        # y_center = np.mean(proto_embedding[count : count+args.prototypes_per_class, 1])
        # ax.text(x_center*1.02, y_center*1.05,
        #         label, 
        #         # color = all_color[label],
        #         fontsize = 14,
        #         style = "italic", 
        #         horizontalalignment='center',
        #         verticalalignment='top',
        #         wrap=True,
        #         bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
        #         )
        
    # title = " ".join([args.model, 'UMAP embedding on', args.dataset])
    # ax.set_title(title)
    ncol = 1 if len(all_labels) < 20 else len(all_labels)//20
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=ncol)
    ax = plt.gca()
    ax.set_xticks([])   # Hide the x and y axis ticks
    ax.set_yticks([])
    
    path = args.plot_dir + args.exp_code + '_umap.png'
    plt.subplots_adjust(right=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("\tUMAP embedding saved")


def plot_two_latents_embedding(args, data):
    latent_embedding = load_file(args, '_latent.npy')[:, 0:2]
    proto_embedding = load_file(args, '_prototypes.npy')[:, 0:2]
    predicted = load_file(args, '_pred.csv')
    y = predicted['celltype'] if 'celltype' in predicted.columns else predicted['idx1'] 

    f, ax = plt.subplots(1, figsize = (14, 10))

    model_labels = data.cell_encoder.classes_
    cell_labels = np.unique(y)
    all_labels = set(set(list(np.unique(model_labels))+list(np.unique(cell_labels))))
    
    cmap = plt.cm.nipy_spectral
    norm = plt.Normalize(0, len(all_labels)-1)
    all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
    all_color = {}
    for i, c in enumerate(all_labels):
        all_color[c] = all_color_list[i]
    
    # plot latent embeddings, color according to orig labels
    for label in cell_labels:
        ax.scatter(
            *latent_embedding[y == label, :].T,
            s=5, # point size
            alpha=0.5,
            color = all_color[label],
            label = label,
            )
        # x_center = np.mean(latent_embedding[y == label, 0])
        # y_center = np.mean(latent_embedding[y == label, 1])
        # ax.text(x_center*1.02, y_center*1.05,
        #         label, 
        #         # color = all_color[label],
        #         fontsize = 12,
        #         style = 'oblique', 
        #         horizontalalignment='center',
        #         verticalalignment='top',
        #         wrap=True,
        #         bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
        #         )
    
    # plot prototype embeddings, color according to prototype labels
    for t, label in enumerate(model_labels):
        count = args.prototypes_per_class * t
        for i in range(args.prototypes_per_class ):
            ax.scatter(
                *proto_embedding[count+i, :].T,
                s=60,
                linewidth=0.7,
                edgecolors="k",
                marker="o",
                color = all_color[label],
            )
        # # add class label text at the center of each prototype embeddings
        # x_center = np.mean(proto_embedding[count:count+args.prototypes_per_class, 0])
        # y_center = np.mean(proto_embedding[count:count+args.prototypes_per_class, 1])
        # ax.text(x_center*1.02, y_center*1.05,
        #         label, 
        #         # color = all_color[label],
        #         fontsize = 14,
        #         style = "italic", 
        #         horizontalalignment='center',
        #         verticalalignment='top',
        #         wrap=True,
        #         bbox=dict(boxstyle='round,pad=0.05', fc='w', lw=0, alpha=0.8),
        #         )
        
    # title = "First 2 latents embedding on " + args.dataset
    # ax.set_title(title)
    ncol = 1 if len(all_labels) < 20 else len(all_labels)//20
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=ncol)
    ax = plt.gca()
    ax.set_xticks([])   # Hide the x and y axis ticks
    ax.set_yticks([])
    
    path = args.plot_dir + args.exp_code + '_2latent.png'
    plt.subplots_adjust(right=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("\tLatent2 embedding saved")



def plot_protocorr_heatmap(args, data):
    # prototype correlation heatmap
    proto = load_file(args, "_prototypes.npy")

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
                celltypes, rotation=45)
    plt.yticks(np.arange(args.prototypes_per_class//2, args.num_prototypes, args.prototypes_per_class), 
                celltypes, rotation=90)

    plt.ylabel('Prototypes', fontsize = 14)
    plt.xlabel('Prototypes', fontsize = 14)
    # title = " ".join([args.model, 'Prototype Correlation on', args.dataset])
    # plt.title(title)

    save_path = args.plot_dir + args.exp_code + '_protoCorr.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("\tPrototype correlation heatmap saved")



def plot_distance_to_prototypes(args, data):
    latents = load_file(args, '_latent.npy')
    protos = load_file(args, '_prototypes.npy')
    predicted = load_file(args, '_pred.csv')
    pred = predicted['idx1'].values

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
    title = args.dataset + ' Distance Distribution to Prototypes'
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
def plot_prp_dist(data, args):
    """
    Plot the histogram of PRP scores of all genes for each cell type
    """
    prp_path = args.prp_path
    celltypes = data.cell_encoder.classes_
    gene_names = pd.Series(data.gene_names)
    filename = '_' + args.exp_code + '_relgenes.npy'

    for i in range(args.num_classes):
        file1 = celltypes[i].replace("/", " OR ") + filename
        proto_rel = np.load(prp_path + file1, allow_pickle = True)

        # Create a figure with 6 subplots (one for each prototype) in one row
        fig, axes = plt.subplots(args.prototypes_per_class, 1, 
                                 figsize=(20, 5*args.prototypes_per_class))

        for p in range(args.prototypes_per_class):
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
        file1 = celltypes[i].replace("/", " OR ") + filename
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
        # file.write(celltypes[i] + "\n")
        # [file.write(item + '\t') for item in top_genes]
        # file.write('\n')
        df_top_genes = pd.concat([df_top_genes, pd.Series(top_genes, name = celltypes[i])], axis = 0)

    # Save DataFrame to csv
    save_file(df_top_genes, args, '_LRPgenes', lrp_path)
    print("\tPRP relavance genes saved")
        


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
        file1 = celltypes[i].replace("/", " OR ") + filename
        gene_rel = np.load(lrp_path + file1, allow_pickle = True) # genes
        
        df = pd.DataFrame(data = gene_rel).T
        all_marker_idx += df.sum().nlargest(20).index.tolist()

    all_marker_idx = list(set(all_marker_idx))

    marker_value = []
    for i in range(args.num_classes):
        file1 = celltypes[i].replace("/", " OR ") + filename
        gene_rel = np.load(lrp_path + file1, allow_pickle = True) # genes
        marker_value.append(gene_rel[all_marker_idx])

    all_marker_df = pd.DataFrame(marker_value, index = celltypes, columns = gene_names[all_marker_idx])

    plt.figure(figsize = (len(all_marker_idx) // 6, args.num_classes // 3))
    ax = sns.heatmap(all_marker_df,
            linewidths = 0.5,
            cmap = "Blues_r")
    
    plt.xlabel('Gene Name', fontsize = 14)
    plt.ylabel('Cell Types', fontsize = 14)
    # title = "Cell Type - Marker Gene Heatmap"
    # plt.title(title, fontsize = 20)
    plt.tight_layout()

    plt.savefig(lrp_path + 'celltype_markergene_heatmap.png', bbox_inches='tight')
    plt.close()
    print("\tCell type - marker gene heatmap saved")


def plot_outlier_heatmap(data, args):
    lrp_path = args.lrp_path
    celltypes = data.cell_encoder.classes_
    gene_names = pd.Series(data.gene_names)
    filename1 = "_" + args.exp_code + "_relgenes.npy"
    filename2 = "_" + args.exp_code + "_lrp.npy"

    for i in range(args.num_classes):
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


def plot_marker_venn_diagram(adata, args):
    """
    Plot the venn diagram of top DE genes and top lrp genes for each cell type
    """
    lrp_path = args.lrp_path
    num_classes = args.num_classes
    celltypes = np.unique(adata.obs['celltype'])
    gene_names = adata.var['gene_name']

    # top DE genes
    sc.pp.log1p(adata)
    adata.uns['log1p']["base"] = None
    sc.tl.rank_genes_groups(adata, groupby = 'celltype', use_raw = False, n_genes = 50, method = 'wilcoxon')
    top_de_genes = adata.uns['rank_genes_groups']['names']

    # top lrp marker genes
    markers = {}
    for i in range(num_classes):
        file1 = celltypes[i].replace("/", " OR ") + "_" + args.exp_code + "_relgenes.npy"
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
        ax = plt.gca()
        ax.set_xticks([])   # Hide the x and y axis ticks
        ax.set_yticks([])
        

    # plt.suptitle('Overlapping Marker Gene Venn Diagram')
    plt.tight_layout()
    plt.savefig(lrp_path + 'DE_marker_venn.png', bbox_inches='tight')
    plt.close()

    print("\tDE marker genes venn diagram saved")

