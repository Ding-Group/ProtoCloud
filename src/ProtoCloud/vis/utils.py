def get_UMAP(latent, n_neighbors=10, min_dist=0.2):
    umap_kwargs = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
    latent_embedding = umap.UMAP(**umap_kwargs).fit_transform(latent)
    print("UMAP computed")
    return latent_embedding

def get_tsne(latents):
    tsne = TSNE(n_components=2, random_state=0)
    transformed_data = tsne.fit_transform(latents)
    return transformed_data

def plot_umap(latent_embedding, all_labels, s=1, a=1, type_color=None, ax=None, show_legend=False):
    mapped_color = map_color(all_labels, type_color)
    
    if ax is None:
        f, ax = plt.subplots(1, figsize = (14, 10))

    ax.scatter(
        *latent_embedding.T,
        s=s, # point size
        linewidth=0,
        alpha=a,
        color = mapped_color,
        )
    ax.set_ylabel('UMAP1')
    ax.set_xlabel('UMAP2')

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        
def plot_protos_umap(proto_embedding, model_labels, type_color=None,
                    prototypes_per_class = 6, s=25, a=1,
                    ax=None, show_legend=False):
    model_labels = [x for x in model_labels for _ in range(prototypes_per_class)]
    mapped_color = map_color(model_labels, type_color)

    if ax is None:
        f, ax = plt.subplots(1, figsize = (14, 10))

    ax.scatter(proto_embedding[:, 0], proto_embedding[:, 1],
                s=s, alpha=a,
                linewidth=1,
                edgecolors="k",
                marker="o",
                color=mapped_color)

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))




def plot_tsne(transformed_data, all_labels, type_color=None, s=1, a=1, ax=None, show_legend=False):
    mapped_color = map_color(all_labels, type_color)

    if ax is None:
        f, ax = plt.subplots(1, figsize = (14, 10))

    # plot latent embeddings
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1],
                s=s, alpha=a,
                color=mapped_color)

    ax.set_ylabel('tSNE_1')
    ax.set_xlabel('tSNE_2')

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


def plot_protos_tsne(proto_embedding, model_labels, type_color=None,
                    prototypes_per_class = 6, s=25, a=1,
                    ax=None, show_legend=False):
    model_labels = [x for x in model_labels for _ in range(prototypes_per_class)]
    mapped_color = map_color(model_labels, type_color)

    if ax is None:
        f, ax = plt.subplots(1, figsize = (14, 10))

    ax.scatter(proto_embedding[:, 0], proto_embedding[:, 1],
                s=s, alpha=a,
                linewidth=1,
                edgecolors="k",
                marker="o",
                color=mapped_color)

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        
import matplotlib.lines as mlines

def get_color(all_labels, cmap=None):
    cmap = plt.cm.nipy_spectral if cmap is None else cmap
    norm = plt.Normalize(0, len(all_labels)-1)
    all_color_list = [cmap(i) for i in np.linspace(0, 1, len(all_labels))]
    all_color = {}
    for i, c in enumerate(all_labels):
        all_color[c] = all_color_list[i]
    return all_color


def map_color(labels, type_color):
    mapped_color = [type_color[label] for label in labels]
    return mapped_color


def plot_2latent(transformed_data, all_labels, type_color=None, s=1, a=1, ax=None, show_legend=False,
                 axis_x = 'Latent_1', axis_y ='Latent_2'):
    mapped_color = map_color(all_labels, type_color)

    if ax is None:
        f, ax = plt.subplots(1, figsize = (14, 10))

    # plot latent embeddings
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1],
                s=s, alpha=a,
                color=mapped_color)

    ax.set_xlabel(axis_x)
    ax.set_ylabel(axis_y)
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


def plot_protos_2latent(proto_embedding, model_labels, type_color=None,
                    prototypes_per_class = 6, s=25, a=1,
                    ax=None, show_legend=False):
    model_labels = [x for x in model_labels for _ in range(prototypes_per_class)]
    mapped_color = map_color(model_labels, type_color)

    ax.scatter(proto_embedding[:, 0], proto_embedding[:, 1],
                s=s, alpha=a,
                linewidth=1,
                edgecolors="k",
                marker="o",
                color=mapped_color)

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))