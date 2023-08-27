import os

import numpy as np
from scipy.stats import fisher_exact
from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering, linkage
from sklearn.cluster import KMeans
import polars as pl
import matplotlib.pyplot as plt


def get_motif_occurences(hits_df):
    occ_df = (
        hits_df
        .pivot(index="peak_id", columns="motif_name", values="count", maintain_order=False)
        .sort(["peak_id"])
    )

    motif_names = occ_df.columns[1:]

    num_peaks = occ_df.height
    num_motifs = len(motif_names)

    occ_mat = np.zeros((num_peaks, num_motifs), dtype=np.int16)
    for i, m in enumerate(motif_names):
        occ_mat[:,i] = occ_df.get_column(m).to_numpy()

    occ_bin = (occ_mat > 0).astype(np.int32)
    coocc = occ_bin.T @ occ_bin

    return occ_df, occ_mat, occ_bin, coocc, motif_names


def cooccurrence_sigs(coocc, num_peaks):
    num_motifs = coocc.shape[0]
    nlps = np.full((num_motifs, num_motifs), np.nan)

    for i in range(num_motifs):
        for j in range(i):
            cont_table = np.array([[0, 0], [0, 0]])
            cont_table[0,0] = coocc[i,j]
            cont_table[1,0] = coocc[i,i] - coocc[i,j]
            cont_table[0,1] = coocc[j,j] - coocc[i,j]
            cont_table[1,1] = num_peaks - coocc[i,i] - coocc[j,j] + coocc[i,j]

            pval = fisher_exact(cont_table, alternative="greater").pvalue
            nlp = -np.log10(pval)

            nlps[i,j] = nlp

    return nlps


def cluster_matrix_indices(matrix):
    """
    Clusters matrix using k-means. Always clusters on the first
    axis. Returns the indices needed to optimally order the matrix
    by clusters.
    
    Adapted from: https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/vizutils.py#L100-L129
    """
    if len(matrix) == 1:
        # Don't cluster at all
        return np.array([0])

    num_clusters = min(max(5, len(matrix) // 4), len(matrix))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_assignments = kmeans.fit_predict(matrix)

    # Perform hierarchical clustering on the cluster centers to determine optimal ordering
    kmeans_centers = kmeans.cluster_centers_
    cluster_order = leaves_list(
        optimal_leaf_ordering(linkage(kmeans_centers, method="centroid"), kmeans_centers)
    )

    # Order the peaks so that the cluster assignments follow the optimal ordering
    cluster_inds = []
    for cluster_id in cluster_order:
        cluster_inds.append(np.where(cluster_assignments == cluster_id)[0])
    cluster_inds = np.concatenate(cluster_inds)

    return cluster_inds


def plot_score_distributions(hits_df, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    
    bins = 20
    for name, data in hits_df.groupby("motif_name"):
        scores = data.get_column("hit_score").to_numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.hist(scores, bins=15)

        ax.set_title(f"Distribution of {name} hit scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count per bin")

        output_path = os.path.join(plot_dir, f"{name}".png)
        plt.savefig(output_path, dpi=300)


def plot_homotypic_densities(occ_mat, motif_names, plot_dir):
    """
    Plots a CDF of number of motif hits per peak, for each motif.
    Adapted from: https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/vizutils.py#L161-L174
    """
    os.makedirs(plot_dir, exist_ok=True)

    for i, k in enumerate(motif_names):
        counts = occ_mat[:, i]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        bins = np.concatenate([np.arange(np.max(counts)), [np.inf]])
        ax.hist(counts, bins=bins, density=True, histtype="step", cumulative=True)

        ax.set_title(f"Cumulative distribution of {k} hit counts per peak")
        ax.set_xlabel("Number of hits in peak")
        ax.set_ylabel("Cumulative fraction of peaks")

        plt.show()

        fig.tight_layout()
        
        output_path = os.path.join(plot_dir, f"{k}".png)
        plt.savefig(output_path, dpi=300)


def plot_frac_peaks(occ_bin, motif_names, plot_path):
    """
    Adapted from https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/run_finemo.py#L267-L280
    """
    num_peaks, num_motifs = occ_bin.shape
    frac_peaks_with_motif = occ_bin.mean(axis=0) 

    labels = np.array(motif_names)
    sorted_inds = np.flip(np.argsort(frac_peaks_with_motif))
    frac_peaks_with_motif = frac_peaks_with_motif[sorted_inds]
    labels = labels[sorted_inds]
    
    fig, ax = plt.subplots(figsize=(15, 20))
    ax.bar(np.arange(num_motifs), frac_peaks_with_motif)

    ax.set_title("Fraction of peaks with each motif")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    plt.xticks(rotation=90)

    plt.savefig(plot_path)

    
def plot_occurrence(occ_mat, motif_names, peak_order, motif_order, plot_path):
    matrix = occ_mat[peak_order, motif_order]

    motif_names_x =  np.array(motif_names)[motif_order]

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    im = ax.imshow(matrix)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("cbarlabel", rotation=-90)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names_x, rotation=90)
    ax.set_xlabel("Motif")
    ax.set_ylabel("Peak")

    fig.tight_layout()
    plt.savefig(plot_path, dpi=300)


def plot_cooccurrence_counts(coocc, motif_names, motif_order, plot_path):
    matrix = coocc[motif_order, motif_order]

    motif_names = np.array(motif_names)[motif_order]

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    ax.imshow(matrix)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, rotation=90)
    ax.set_yticks(np.arange(len(motif_names)))
    ax.set_yticklabels(motif_names)

    # Annotate heatmap
    for i in range(matrix.shape[0]):
        for j in range(i):
            text = f"{matrix[i,j]}"
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    plt.savefig(plot_path, dpi=300)


def plot_cooccurrence_sigs(coocc_nlp, motif_names, motif_order, plot_path):
    matrix = coocc_nlp[motif_order, motif_order]

    motif_names = np.array(motif_names)[motif_order]

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    ax.imshow(matrix)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, rotation=90)
    ax.set_yticks(np.arange(len(motif_names)))
    ax.set_yticklabels(motif_names)

    # Annotate heatmap
    for i in range(matrix.shape[0]):
        for j in range(i):
            text = f"{matrix[i,j]:.1f}"
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    plt.savefig(plot_path, dpi=300)