import os

import numpy as np
from scipy.stats import fisher_exact
from scipy.cluster import hierarchy
# from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering, linkage
# from sklearn.cluster import KMeans
import polars as pl
import matplotlib.pyplot as plt


def get_motif_occurences(hits_df):
    occ_df = (
        hits_df
        .pivot(index="peak_id", columns="motif_name", values="count", 
               aggregate_function="sum", maintain_order=False)
        .fill_null(0)
        .sort(["peak_id"])
    )
    # print(occ_df) ####

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
    nlps = np.zeros((num_motifs, num_motifs))
    lors = np.zeros((num_motifs, num_motifs))

    for i in range(num_motifs):
        for j in range(i):
            cont_table = np.array([[0, 0], [0, 0]])
            cont_table[0,0] = coocc[i,j]
            cont_table[1,0] = coocc[i,i] - coocc[i,j]
            cont_table[0,1] = coocc[j,j] - coocc[i,j]
            cont_table[1,1] = num_peaks - coocc[i,i] - coocc[j,j] + coocc[i,j]

            res = fisher_exact(cont_table, alternative="greater")
            pval = res.pvalue
            odds_ratio = res.statistic
            # print(pval) ####
            nlp = np.clip(-np.log10(pval), None, 300)
            lor = np.log10(odds_ratio)

            nlps[i,j] = nlps[j,i] = nlp
            lors[i,j] = lors[j,i] = lor

    return nlps, lors


def seqlet_recall(hits_df, peaks_df, seqlets_df, scale_scores, modisco_half_width):
    if scale_scores:
        score_col = "hit_score_scaled"
    else:
        score_col = "hit_score_unscaled"

    hits_filtered = (
        hits_df
        .with_columns(pl.col('peak_id').cast(pl.UInt32))
        .join(
            peaks_df.lazy(), on="peak_id", how="inner"
        )
        .filter(
            ((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= 0) 
            & ((pl.col("end_untrimmed") - pl.col("peak_region_start")) <= (2 * modisco_half_width))
        )
        .select(
            chr=pl.col("chr"),
            start_untrimmed=pl.col("start_untrimmed"),
            end_untrimmed=pl.col("end_untrimmed"),
            is_revcomp=pl.col("strand") == '-',
            motif_name=pl.col("motif_name"),
            score=pl.col(score_col),
            # start_offset=(pl.col("start_untrimmed") - pl.col("peak_region_start") - modisco_half_width).abs(),
            # ends_offset=(pl.col("end_untrimmed") - pl.col("peak_region_start") - modisco_half_width).abs(),
        )
        .unique(subset=["chr", "start_untrimmed", "motif_name", "is_revcomp"])
        # .filter(
        #     (pl.col("start_offset") < modisco_half_width) & (pl.col("ends_offset") < modisco_half_width)
        # )
    )

    seqlet_counts_df = seqlets_df.group_by("motif_name").agg(pl.count()).collect()
    seqlet_counts = {r["motif_name"]: r["count"] for r in seqlet_counts_df.iter_rows(named=True)}
    # print(seqlet_counts_df) ####
    # print(seqlet_counts) ####
    
    overlaps_df = (
        hits_filtered.join(
            seqlets_df, 
            on=["chr", "start_untrimmed", "end_untrimmed", "is_revcomp", "motif_name"],
            how="left"
        )
        .with_columns(pl.col("seqlet_indicator").fill_null(strategy="zero"))
        .collect()
    )

    nonoverlaps_df = (
        seqlets_df.join(
            hits_filtered, 
            on=["chr", "start_untrimmed", "end_untrimmed", "is_revcomp", "motif_name"],
            how="anti"
        )
        .collect()
    )
    
    overlaps_by_motif = overlaps_df.partition_by("motif_name", as_dict=True)
    recalls = {}
    # seqlet_counts = {}
    for k, v in overlaps_by_motif.items():
        # num_seqlets = v.height
        recall_data = (
            v.lazy()
            .sort("score", descending=True)
            .select(seqlet_recall=pl.cumsum("seqlet_indicator"))
            .collect()
            .get_column("seqlet_recall")
            .to_numpy() / seqlet_counts[k]
        )

        recalls[k] = recall_data
        # seqlet_counts[k] = num_seqlets

    return recalls, overlaps_df, nonoverlaps_df, seqlet_counts


# def cluster_matrix_indices(matrix):
#     """
#     Clusters matrix using k-means. Always clusters on the first
#     axis. Returns the indices needed to optimally order the matrix
#     by clusters.
    
#     Adapted from: https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/vizutils.py#L100-L129
#     """
#     if len(matrix) == 1:
#         # Don't cluster at all
#         return np.array([0])

#     num_clusters = min(max(5, len(matrix) // 4), len(matrix))
    
#     # Perform k-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     cluster_assignments = kmeans.fit_predict(matrix)

#     # Perform hierarchical clustering on the cluster centers to determine optimal ordering
#     kmeans_centers = kmeans.cluster_centers_
#     cluster_order = leaves_list(
#         optimal_leaf_ordering(linkage(kmeans_centers, method="centroid"), kmeans_centers)
#     )

#     # Order the peaks so that the cluster assignments follow the optimal ordering
#     cluster_inds = []
#     for cluster_id in cluster_order:
#         cluster_inds.append(np.where(cluster_assignments == cluster_id)[0])
#     cluster_inds = np.concatenate(cluster_inds)

#     return cluster_inds

def order_rows(matrix):
    linkage = hierarchy.linkage(matrix, method='average', metric='euclidean',optimal_ordering=False)
    dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendrogram['leaves']
    
    return order


def plot_score_distributions(hits_df, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    
    for name, data in hits_df.groupby("motif_name"):
        scores = data.get_column("hit_score_scaled").to_numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.hist(scores, bins=15)

        ax.set_title(f"Distribution of {name} hit scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count per bin")

        output_path = os.path.join(plot_dir, f"{name}.png")
        plt.savefig(output_path, dpi=300)

        plt.close(fig)


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
        
        output_path = os.path.join(plot_dir, f"{k}.png")
        plt.savefig(output_path, dpi=300)

        plt.close(fig)


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

    plt.close(fig)

    
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

    plt.close(fig)


def plot_cooccurrence_counts(coocc, motif_names, motif_order, plot_path):
    matrix = coocc[np.ix_(motif_order, motif_order)]

    motif_names = np.array(motif_names)[motif_order]

    # plt.figure(figsize=(15,15))

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    ax.imshow(matrix)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, size=4, rotation=90)
    ax.set_yticks(np.arange(len(motif_names)))
    ax.set_yticklabels(motif_names, size=4)

    # Annotate heatmap
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            text = f"{matrix[i,j]:.1e}"
            ax.text(j, i, text, ha="center", va="center", size=1.5)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=600)


def plot_cooccurrence_lors(coocc_lor, motif_names, motif_order, plot_path):
    matrix = coocc_lor[np.ix_(motif_order, motif_order)]

    motif_names = np.array(motif_names)[motif_order]

    # plt.figure(figsize=(15,15))

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    ax.imshow(matrix)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, size=4, rotation=90)
    ax.set_yticks(np.arange(len(motif_names)))
    ax.set_yticklabels(motif_names, size=4)

    # Annotate heatmap
    for i in range(matrix.shape[0]):
        for j in range(i):
            text = f"{matrix[i,j]:.1e}"
            ax.text(j, i, text, ha="center", va="center", size=1.5)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=600)


def plot_cooccurrence_sigs(coocc_nlp, motif_names, motif_order, plot_path):
    matrix = coocc_nlp[np.ix_(motif_order, motif_order)]

    motif_names = np.array(motif_names)[motif_order]

    # plt.figure(figsize=(15,15))

    fig, ax = plt.subplots()
    
    # Plot the heatmap
    ax.imshow(matrix)

    # Set axes on heatmap
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, size=4, rotation=90)
    ax.set_yticks(np.arange(len(motif_names)))
    ax.set_yticklabels(motif_names, size=4)

    # Annotate heatmap
    for i in range(matrix.shape[0]):
        for j in range(i):
            text = f"{matrix[i,j]:.1e}"
            ax.text(j, i, text, ha="center", va="center", size=1.5)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=600)

    plt.close(fig)


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

    plt.close(fig)


def plot_modisco_recall(seqlet_recalls, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    labels = list(seqlet_recalls.keys())
    recalls = [seqlet_recalls[l][-1] for l in labels]

    fig, ax = plt.subplots(figsize=(15, 20))
    ax.bar(labels, recalls)

    ax.set_title("Total Modisco seqlet recall")
    plt.xticks(rotation=90)

    plt.savefig(os.path.join(plot_dir, f"overall.png"))

    plt.close(fig)

    for k, v in seqlet_recalls.items():
        x = np.arange(v.shape[0])
        
        plt.plot(x, v)
        plt.title(f"{k} Modisco seqlet recall")
        plt.xlabel("Hit rank")
        plt.ylabel("Recall")
        
        output_path = os.path.join(plot_dir, f"{k}.png")
        plt.savefig(output_path, dpi=300)

        plt.close()