import os
import warnings
import importlib

import numpy as np
from scipy.stats import fisher_exact
from scipy.cluster import hierarchy
# from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering, linkage
# from sklearn.cluster import KMeans
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from matplotlib import patches
from jinja2 import Template

from . import templates


def abbreviate_motif_name(name):
    group, motif = name.split(".")

    if group == "pos_patterns":
        group_short = "+"
    elif group == "neg_patterns":
        group_short = "-"

    motif_num = motif.split("_")[1]

    return f"{group_short}/{motif_num}"


def get_motif_occurences(hits_df, motif_names):
    occ_df = (
        hits_df
        .collect()
        .pivot(index="peak_id", columns="motif_name", values="count", aggregate_function="sum")
        .fill_null(0)
        .with_columns(total=pl.sum_horizontal(*motif_names))
        .sort(["peak_id"])
    )
    # # print(occ_df) ####

    # motif_names = occ_df.columns[1:]

    num_peaks = occ_df.height
    num_motifs = len(motif_names)

    occ_mat = np.zeros((num_peaks, num_motifs), dtype=np.int16)
    for i, m in enumerate(motif_names):
        occ_mat[:,i] = occ_df.get_column(m).to_numpy()

    occ_bin = (occ_mat > 0).astype(np.int32)
    coocc = occ_bin.T @ occ_bin

    # return occ_df, occ_mat, occ_bin, coocc, motif_names

    return occ_df, coocc


def plot_hit_distributions(occ_df, motif_names, plot_dir):
    motifs_dir = os.path.join(plot_dir, "motif_hit_distributions")
    os.makedirs(motifs_dir, exist_ok=True)

    for m in motif_names:
        fig, ax = plt.subplots(figsize=(6, 2))

        unique, counts = np.unique(occ_df.get_column(m), return_counts=True)
        freq = counts / counts.sum()
        num_bins = np.amax(unique) + 1
        x = np.arange(num_bins)
        y = np.zeros(num_bins)
        y[unique] = freq
        ax.bar(x, y)

        output_path = os.path.join(motifs_dir, f"{m}.png")
        plt.savefig(output_path, dpi=300)

        plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4))

    unique, counts = np.unique(occ_df.get_column("total"), return_counts=True)
    freq = counts / counts.sum()
    num_bins = np.amax(unique) + 1
    x = np.arange(num_bins)
    y = np.zeros(num_bins)
    y[unique] = freq
    ax.bar(x, y)

    ax.set_xlabel("Motifs per peak")
    ax.set_ylabel("Frequency")

    output_path = os.path.join(plot_dir, "total_hit_distribution.png")
    plt.savefig(output_path, dpi=300)

    plt.close(fig)


def plot_peak_motif_indicator_heatmap(peak_hit_counts, motif_names, output_path):
    """
    Plots a simple indicator heatmap of the motifs in each peak.
    """
    cov_norm = 1 / np.sqrt(np.diag(peak_hit_counts))
    matrix = peak_hit_counts * cov_norm[:,None] * cov_norm[None,:]
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]
    # motif_keys_x = motif_keys
    
    # Cluster matrix by motifs
    # matrix_t = np.transpose(peak_hit_counts)
    # inds = cluster_matrix_indices(matrix_t, max(5, len(matrix_t) // 4))
    # matrix = np.transpose(matrix_t[inds])
    # motif_keys_x =  np.array(motif_keys)[inds]
    # motif_keys_y = np.array(motif_keys)

    # num_motifs=len(motif_keys)
    # fig_width = max(5, num_motifs)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the heatmap
    ax.imshow(matrix, interpolation="nearest", aspect="auto", cmap="Greens")

    # Set axes on heatmap
    ax.set_yticks(np.arange(len(motif_keys)))
    ax.set_yticklabels(motif_keys)
    ax.set_xticks(np.arange(len(motif_keys)))
    ax.set_xticklabels(motif_keys, rotation=90)
    ax.set_xlabel("Motif i")
    ax.set_ylabel("Motif j")

    # fig.tight_layout()
    plt.savefig(output_path, dpi=300)

    plt.close()
    


# def cooccurrence_sigs(coocc, num_peaks):
#     num_motifs = coocc.shape[0]
#     nlps = np.zeros((num_motifs, num_motifs))
#     lors = np.zeros((num_motifs, num_motifs))

#     for i in range(num_motifs):
#         for j in range(i):
#             cont_table = np.array([[0, 0], [0, 0]])
#             cont_table[0,0] = coocc[i,j]
#             cont_table[1,0] = coocc[i,i] - coocc[i,j]
#             cont_table[0,1] = coocc[j,j] - coocc[i,j]
#             cont_table[1,1] = num_peaks - coocc[i,i] - coocc[j,j] + coocc[i,j]

#             res = fisher_exact(cont_table, alternative="greater")
#             pval = res.pvalue
#             odds_ratio = res.statistic
#             # print(pval) ####
#             nlp = np.clip(-np.log10(pval), None, 300)
#             lor = np.log10(odds_ratio)

#             nlps[i,j] = nlps[j,i] = nlp
#             lors[i,j] = lors[j,i] = lor

#     return nlps, lors


def get_cwms(regions, positions_df, motif_width):
    idx_df = (
        positions_df
        .select(
            peak_idx=pl.col("peak_id"),
            start_idx=pl.col("start_untrimmed") - pl.col("peak_region_start"),
            is_revcomp=pl.col("is_revcomp")
        )
    )
    peak_idx = idx_df.get_column('peak_idx').to_numpy()
    start_idx = idx_df.get_column('start_idx').to_numpy()
    is_revcomp = idx_df.get_column("is_revcomp").to_numpy().astype(bool)

    # print(peak_idx.dtype) ####
    # print(start_idx.dtype) ####
    # print(is_revcomp.dtype) ####

    row_idx = peak_idx[:,None,None]
    pos_idx = start_idx[:,None,None] + np.zeros((1,1,motif_width), dtype=int)
    pos_idx[~is_revcomp,:,:] += np.arange(motif_width)[None,None,:]
    pos_idx[is_revcomp,:,:] += np.arange(motif_width)[None,None,::-1]
    nuc_idx = np.zeros((peak_idx.shape[0],4,1), dtype=int)
    nuc_idx[~is_revcomp,:,:] += np.arange(4)[None,:,None]
    nuc_idx[is_revcomp,:,:] += np.arange(4)[None,::-1,None]

    # print(row_idx.dtype) ####
    # print(nuc_idx.dtype) ####
    # print(pos_idx.dtype) ####

    seqs = regions[row_idx, nuc_idx, pos_idx]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        cwms = seqs.mean(axis=0)

    return cwms


def seqlet_recall(regions, hits_df, peaks_df, seqlets_df, motif_names, modisco_half_width, motif_width):
    hits_df = (
        hits_df
        .with_columns(pl.col('peak_id').cast(pl.UInt32))
        .join(
            peaks_df.lazy(), on="peak_id", how="inner"
        )
        # .unique(subset=["chr", "start", "motif_name", "strand"])
        .select(
            chr=pl.col("chr"),
            start_untrimmed=pl.col("start_untrimmed"),
            end_untrimmed=pl.col("end_untrimmed"),
            is_revcomp=pl.col("strand") == '-',
            motif_name=pl.col("motif_name"),
            peak_region_start=pl.col("peak_region_start"),
            peak_id=pl.col("peak_id")
        )
    )

    hits_unique = hits_df.unique(subset=["chr", "start_untrimmed", "motif_name", "is_revcomp"])
    
    region_len = regions.shape[2]
    center = region_len / 2
    hits_filtered = (
        hits_df
        .filter(
            ((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= (center - modisco_half_width)) 
            & ((pl.col("end_untrimmed") - pl.col("peak_region_start")) <= (center + modisco_half_width))
        )
        .unique(subset=["chr", "start_untrimmed", "motif_name", "is_revcomp"])
    )

    # print(seqlets_df.collect()) ####
    # print(hits_df.collect()) ####

    # seqlet_counts_df = seqlets_df.group_by("motif_name").agg(pl.count()).collect()
    # seqlet_counts = {r["motif_name"]: r["count"] for r in seqlet_counts_df.iter_rows(named=True)}
    # print(seqlet_counts_df) ####
    # print(seqlet_counts) ####

    # debug = (
    #     seqlets_df
    #     .filter(
    #         ((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= (center - modisco_half_width)) 
    #         & ((pl.col("end_untrimmed") - pl.col("peak_region_start")) <= (center + modisco_half_width))
    #     )
    # ) ####
    # print(seqlets_df.collect()) ####
    # print(debug.collect()) ####
    
    overlaps_df = (
        hits_filtered.join(
            seqlets_df, 
            on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
            how="inner",
            # validate='1:1'
        )
        # .with_columns(pl.col("seqlet_indicator").fill_null(strategy="zero"))
        .collect()
    )

    # debug = (
    #     overlaps_df
    #     .filter(
    #         ~(((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= (center - modisco_half_width)) 
    #         & ((pl.col("end_untrimmed") - pl.col("peak_region_start")) <= (center + modisco_half_width)))
    #     )
    # ) ####
    # print(overlaps_df) ####
    # print(debug) ####

    seqlets_only_df = (
        seqlets_df.join(
            hits_df, 
            on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
            how="anti",
            # validate='1:1'
        )
        .collect()
    )

    hits_only_filtered_df = (
        hits_filtered.join(
            seqlets_df, 
            on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
            how="anti",
            # validate='1:1'
        )
        .collect()
    )

    # debug_df = (
    #     overlaps_df.join(
    #         hits_only_filtered_df, 
    #         on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
    #         how="inner"
    #     )
    #     .collect()
    # ) ####
    # print(debug_df) ####

    hits_by_motif = hits_unique.collect().partition_by("motif_name", as_dict=True)
    hits_fitered_by_motif = hits_filtered.collect().partition_by("motif_name", as_dict=True)
    seqlets_by_motif = seqlets_df.collect().partition_by("motif_name", as_dict=True)
    overlaps_by_motif = overlaps_df.partition_by("motif_name", as_dict=True)
    seqlets_only_by_motif = seqlets_only_df.partition_by("motif_name", as_dict=True)
    hits_only_filtered_by_motif = hits_only_filtered_df.partition_by("motif_name", as_dict=True)

    recall_data = {}
    cwms = {}
    dummy_df = overlaps_df.clear()
    for m in motif_names:
        hits = hits_by_motif.get(m, dummy_df)
        hits_filtered = hits_fitered_by_motif.get(m, dummy_df)
        seqlets = seqlets_by_motif.get(m, dummy_df)
        overlaps = overlaps_by_motif.get(m, dummy_df)
        seqlets_only = seqlets_only_by_motif.get(m, dummy_df)
        hits_only_filtered = hits_only_filtered_by_motif.get(m, dummy_df)
        # hits_only = hits_only_by_motif[m]

        recall_data[m] = {
            "seqlet_recall": np.float64(overlaps.height) / seqlets.height,
            "num_hits_total": hits.height,
            "num_hits_restricted": hits_filtered.height,
            "num_seqlets": seqlets.height,
            "num_overlaps": overlaps.height,
            "num_seqlets_only": seqlets_only.height,
            "num_hits_restricted_only": hits_only_filtered.height
        }

        cwms[m] = {
            "hits_fc": get_cwms(regions, hits, motif_width),
            "seqlets_fc": get_cwms(regions, seqlets, motif_width),
            "seqlets_only": get_cwms(regions, seqlets_only, motif_width),
            "hits_restricted_only": get_cwms(regions, hits_only_filtered, motif_width),
        }
        cwms[m]["hits_rc"] = cwms[m]["hits_fc"][::-1,::-1]
        # cwms[m]["seqlets_rc"] = cwms[m]["seqlets_fc"][::-1,::-1]

    records = [{"motif_name": k} | v for k, v in recall_data.items()]
    recall_df = pl.from_dicts(records)

    return recall_data, recall_df, cwms


class LogoGlyph(AbstractPathEffect):
    def __init__(self, glyph, ref_glyph='E', font_props=None,
                 offset=(0., 0.), **kwargs):

        super().__init__(offset)

        path_orig = TextPath((0, 0), glyph, size=1, prop=font_props)
        dims = path_orig.get_extents()
        ref_dims = TextPath((0, 0), ref_glyph, size=1, prop=font_props).get_extents()

        h_scale = 1 / dims.height
        ref_width = max(dims.width, ref_dims.width)
        w_scale = 1 / ref_width
        w_shift = (1 - dims.width / ref_width) / 2
        x_shift = -dims.x0
        y_shift = -dims.y0
        stretch = (
            Affine2D()
            .translate(tx=x_shift, ty=y_shift)
            .scale(sx=w_scale, sy=h_scale)
            .translate(tx=w_shift, ty=0)
        )

        self.patch = patches.PathPatch([], **kwargs)
        self.patch._path = stretch.transform_path(path_orig)

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        self.patch.set(color=rgbFace)
        self.patch.set_transform(affine + self._offset_transform(renderer))
        self.patch.set_clip_box(gc.get_clip_rectangle())
        clip_path = gc.get_clip_path()
        if clip_path and self.patch.get_clip_path() is None:
            self.patch.set_clip_path(*clip_path)
        self.patch.draw(renderer)


def plot_logo(ax, heights, glyphs, colors=None, font_props=None):
    if colors is None:
        colors = {g: None for g in glyphs}

    ax.margins(x=0, y=0)
    
    pos_values = np.clip(heights, 0, None)
    neg_values = np.clip(heights, None, 0)
    pos_order = np.argsort(pos_values, axis=0)
    neg_order = np.argsort(neg_values, axis=0)[::-1,:]
    pos_reorder = np.argsort(pos_order, axis=0)
    neg_reorder = np.argsort(neg_order, axis=0)
    pos_offsets = np.take_along_axis(
        np.cumsum(
            np.take_along_axis(pos_values, pos_order, axis=0), axis=0
        ), pos_reorder, axis=0
    )
    neg_offsets = np.take_along_axis(
        np.cumsum(
            np.take_along_axis(neg_values, neg_order, axis=0), axis=0
        ), neg_reorder, axis=0
    )
    bottoms = pos_offsets + neg_offsets - heights

    x = np.arange(heights.shape[1])

    for glyph, height, bottom in zip(glyphs, heights, bottoms):
        # print(colors, glyph) ####
        # print(colors[glyph]) ####
        ax.bar(x, height, 0.95, bottom=bottom, 
               path_effects=[LogoGlyph(glyph, font_props=font_props)], color=colors[glyph])

    ax.axhline(zorder=-1, linewidth=0.5, color='black',)


LOGO_ALPHABET = 'ACGT'
LOGO_COLORS = {"A": '#109648', "C": '#255C99', "G": '#F7B32B', "T": '#D62839'}
LOGO_FONT = FontProperties(weight="bold")

def plot_cwms(cwms, out_dir, alphabet=LOGO_ALPHABET, colors=LOGO_COLORS, font=LOGO_FONT):
    for m, v in cwms.items():
        motif_dir = os.path.join(out_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            output_path = os.path.join(motif_dir, f"{cwm_type}.png")

            fig, ax = plt.subplots(figsize=(10,2))

            # print(cwm) ####
            plot_logo(ax, cwm, alphabet, colors=colors, font_props=font)

            for name, spine in ax.spines.items():
                spine.set_visible(False)

            # fig.tight_layout()
            
            plt.savefig(output_path, dpi=100)
            plt.close(fig)


def plot_hit_vs_seqlet_counts(recall_data, output_path):
    x = []
    y = []
    m = []
    for k, v in recall_data.items():
        x.append(v["num_hits_total"])
        y.append(v["num_seqlets"])
        m.append(k)

    lim = max(np.amax(x), np.amax(y))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.axline((0, 0), (lim, lim), color="0.3", linewidth=0.7, linestyle=(0, (5, 5)))
    ax.scatter(x, y, s=5)
    for i, txt in enumerate(m):
        short = abbreviate_motif_name(txt)
        ax.annotate(short, (x[i], y[i]), fontsize=8, weight="bold")

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel("Hits per motif")
    ax.set_ylabel("Seqlets per motif")

    plt.savefig(output_path, dpi=300)
    plt.close()


def write_report(recall_df, motif_names, out_path):
    template_str = importlib.resources.files(templates).joinpath('report.html').read_text()
    template = Template(template_str)
    report = template.render(seqlet_recall_data=recall_df.iter_rows(named=True), motif_names=motif_names)
    with open(out_path, "w") as f:
        f.write(report)


# def chip_cumlative_importance(importance_df, score_type):
#     score_col = f"hit_score_{score_type}"
#     cumulative_importance = (
#         importance_df.lazy()
#         .sort(score_col, descending=True)
#         .select(cumulative_importance=pl.cumsum("chip_importance"))
#         .collect()
#         .get_column("cumulative_importance")
#     )

#     return cumulative_importance


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

# def order_rows(matrix):
#     linkage = hierarchy.linkage(matrix, method='average', metric='euclidean',optimal_ordering=False)
#     dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
#     order = dendrogram['leaves']
    
#     return order


# def plot_score_distributions(hits_df, plot_dir):
#     os.makedirs(plot_dir, exist_ok=True)
    
#     for name, data in hits_df.groupby("motif_name"):
#         scores = data.get_column("hit_score_scaled").to_numpy()

#         fig, ax = plt.subplots(figsize=(8, 8))

#         ax.hist(scores, bins=15)

#         ax.set_title(f"Distribution of {name} hit scores")
#         ax.set_xlabel("Score")
#         ax.set_ylabel("Count per bin")

#         output_path = os.path.join(plot_dir, f"{name}.png")
#         plt.savefig(output_path, dpi=300)

#         plt.close(fig)


# def plot_homotypic_densities(occ_mat, motif_names, plot_dir):
#     """
#     Plots a CDF of number of motif hits per peak, for each motif.
#     Adapted from: https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/vizutils.py#L161-L174
#     """
#     os.makedirs(plot_dir, exist_ok=True)

#     for i, k in enumerate(motif_names):
#         counts = occ_mat[:, i]
        
#         fig, ax = plt.subplots(figsize=(8, 8))
#         bins = np.concatenate([np.arange(np.max(counts)), [np.inf]])
#         ax.hist(counts, bins=bins, density=True, histtype="step", cumulative=True)

#         ax.set_title(f"Cumulative distribution of {k} hit counts per peak")
#         ax.set_xlabel("Number of hits in peak")
#         ax.set_ylabel("Cumulative fraction of peaks")

#         plt.show()

#         fig.tight_layout()
        
#         output_path = os.path.join(plot_dir, f"{k}.png")
#         plt.savefig(output_path, dpi=300)

#         plt.close(fig)


# def plot_frac_peaks(occ_bin, motif_names, plot_path):
#     """
#     Adapted from https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/run_finemo.py#L267-L280
#     """
#     num_peaks, num_motifs = occ_bin.shape
#     frac_peaks_with_motif = occ_bin.mean(axis=0) 

#     labels = np.array(motif_names)
#     sorted_inds = np.flip(np.argsort(frac_peaks_with_motif))
#     frac_peaks_with_motif = frac_peaks_with_motif[sorted_inds]
#     labels = labels[sorted_inds]
    
#     fig, ax = plt.subplots(figsize=(15, 20))
#     ax.bar(np.arange(num_motifs), frac_peaks_with_motif)

#     ax.set_title("Fraction of peaks with each motif")
#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)
#     plt.xticks(rotation=90)

#     plt.savefig(plot_path)

#     plt.close(fig)

    
# def plot_occurrence(occ_mat, motif_names, peak_order, motif_order, plot_path):
#     matrix = occ_mat[peak_order, motif_order]

#     motif_names_x =  np.array(motif_names)[motif_order]

#     fig, ax = plt.subplots()
    
#     # Plot the heatmap
#     im = ax.imshow(matrix)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax)
#     cbar.ax.set_ylabel("cbarlabel", rotation=-90)

#     # Set axes on heatmap
#     ax.set_xticks(np.arange(len(motif_names)))
#     ax.set_xticklabels(motif_names_x, rotation=90)
#     ax.set_xlabel("Motif")
#     ax.set_ylabel("Peak")

#     fig.tight_layout()
#     plt.savefig(plot_path, dpi=300)

#     plt.close(fig)


# def plot_cooccurrence_counts(coocc, motif_names, motif_order, plot_path):
#     matrix = coocc[np.ix_(motif_order, motif_order)]

#     motif_names = np.array(motif_names)[motif_order]

#     # plt.figure(figsize=(15,15))

#     fig, ax = plt.subplots()
    
#     # Plot the heatmap
#     ax.imshow(matrix)

#     # Set axes on heatmap
#     ax.set_xticks(np.arange(len(motif_names)))
#     ax.set_xticklabels(motif_names, size=4, rotation=90)
#     ax.set_yticks(np.arange(len(motif_names)))
#     ax.set_yticklabels(motif_names, size=4)

#     # Annotate heatmap
#     for i in range(matrix.shape[0]):
#         for j in range(i + 1):
#             text = f"{matrix[i,j]:.1e}"
#             ax.text(j, i, text, ha="center", va="center", size=1.5)

#     fig.tight_layout()
#     plt.savefig(plot_path, dpi=600)


# def plot_cooccurrence_lors(coocc_lor, motif_names, motif_order, plot_path):
#     matrix = coocc_lor[np.ix_(motif_order, motif_order)]

#     motif_names = np.array(motif_names)[motif_order]

#     # plt.figure(figsize=(15,15))

#     fig, ax = plt.subplots()
    
#     # Plot the heatmap
#     ax.imshow(matrix)

#     # Set axes on heatmap
#     ax.set_xticks(np.arange(len(motif_names)))
#     ax.set_xticklabels(motif_names, size=4, rotation=90)
#     ax.set_yticks(np.arange(len(motif_names)))
#     ax.set_yticklabels(motif_names, size=4)

#     # Annotate heatmap
#     for i in range(matrix.shape[0]):
#         for j in range(i):
#             text = f"{matrix[i,j]:.1e}"
#             ax.text(j, i, text, ha="center", va="center", size=1.5)

#     fig.tight_layout()
#     plt.savefig(plot_path, dpi=600)


# def plot_cooccurrence_sigs(coocc_nlp, motif_names, motif_order, plot_path):
#     matrix = coocc_nlp[np.ix_(motif_order, motif_order)]

#     motif_names = np.array(motif_names)[motif_order]

#     # plt.figure(figsize=(15,15))

#     fig, ax = plt.subplots()
    
#     # Plot the heatmap
#     ax.imshow(matrix)

#     # Set axes on heatmap
#     ax.set_xticks(np.arange(len(motif_names)))
#     ax.set_xticklabels(motif_names, size=4, rotation=90)
#     ax.set_yticks(np.arange(len(motif_names)))
#     ax.set_yticklabels(motif_names, size=4)

#     # Annotate heatmap
#     for i in range(matrix.shape[0]):
#         for j in range(i):
#             text = f"{matrix[i,j]:.1e}"
#             ax.text(j, i, text, ha="center", va="center", size=1.5)

#     fig.tight_layout()
#     plt.savefig(plot_path, dpi=600)

#     plt.close(fig)


# def plot_frac_peaks(occ_bin, motif_names, plot_path):
#     """
#     Adapted from https://github.com/kundajelab/FiNeMo/blob/fa7d70974c5ea6a4f83898ce01e9f97ed2273a33/run_finemo.py#L267-L280
#     """
#     num_peaks, num_motifs = occ_bin.shape
#     frac_peaks_with_motif = occ_bin.mean(axis=0) 

#     labels = np.array(motif_names)
#     sorted_inds = np.flip(np.argsort(frac_peaks_with_motif))
#     frac_peaks_with_motif = frac_peaks_with_motif[sorted_inds]
#     labels = labels[sorted_inds]
    
#     fig, ax = plt.subplots(figsize=(15, 20))
#     ax.bar(np.arange(num_motifs), frac_peaks_with_motif)

#     ax.set_title("Fraction of peaks with each motif")
#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)
#     plt.xticks(rotation=90)

#     plt.savefig(plot_path)

#     plt.close(fig)


# def plot_modisco_recall(seqlet_recalls, seqlet_counts, plot_dir):
#     os.makedirs(plot_dir, exist_ok=True)

#     labels = sorted(seqlet_recalls.keys(), 
#                     key=lambda x: (x.split("_")[0] == "neg", int(x.split("_")[-1])), reverse=True)
#     recalls = [seqlet_recalls[l][-1] for l in labels]

#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.barh(labels, recalls)

#     ax.set_title("Total Modisco seqlet recall")

#     plt.savefig(os.path.join(plot_dir, "overall.png"))

#     plt.close(fig)

#     for k, v in seqlet_recalls.items():
#         num_hits = v.shape[0]
#         x = np.arange(num_hits)

#         num_seqlets = seqlet_counts[k]
#         bound = np.full(num_hits, num_seqlets)
#         ramp_max = min(num_seqlets, num_hits)
#         bound[:ramp_max] = np.arange(1, ramp_max + 1)
#         bound = bound.astype(float) / float(num_seqlets)
        
#         plt.plot(x, v)
#         plt.plot(x, bound)
#         plt.title(f"{k} Modisco seqlet recall")
#         plt.xlabel("Hit rank")
#         plt.ylabel("Recall")
        
#         plt.tight_layout()
#         output_path = os.path.join(plot_dir, f"{k}.png")
#         plt.savefig(output_path, dpi=300)

#         plt.close()


# def plot_chip_importance(cumulative_importance, plot_path):
#     num_hits = cumulative_importance.shape[0]
#     x = np.arange(num_hits)

#     plt.plot(x, cumulative_importance)
#     plt.xlabel("Hit rank")
#     plt.ylabel("Cumulative ChIP-seq importance scores")
    
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300)

#     plt.close()


# def plot_xcor_distributions(max_xcors, motif_names, plot_dir):
#     os.makedirs(plot_dir, exist_ok=True)

#     for i, k in enumerate(motif_names):        
#         # plt.hist(max_xcors[:,i], bins="auto", range=(0, 1))
#         plt.hist(max_xcors[:,i], bins="auto", range=(0, 2))

#         plt.title(f"Distribution of maximum {k} cross-correlation per region")
#         plt.xlabel("Arctanh cross-correlation")
#         plt.ylabel("Frequency")

#         plt.show()

#         plt.tight_layout()
        
#         output_path = os.path.join(plot_dir, f"{k}.png")
#         plt.savefig(output_path, dpi=300)

#         plt.close()


# def plot_xcor_quantiles(max_xcor_quantiles, motif_names, plot_dir):
#     os.makedirs(plot_dir, exist_ok=True)
    
#     num_bins = max_xcor_quantiles.shape[0]
#     for i, k in enumerate(motif_names):
#         x = np.arange(num_bins) / num_bins
#         plt.plot(x, max_xcor_quantiles[:,i])

#         plt.xlim(0, 1)
#         plt.ylim(0, 1)

#         plt.title(f"Quantiles of maximum {k} cross-correlation per region")
#         plt.xlabel("Quantile")
#         plt.ylabel("Cross-correlation")

#         plt.show()

#         plt.tight_layout()
        
#         output_path = os.path.join(plot_dir, f"{k}.png")
#         plt.savefig(output_path, dpi=300)

#         plt.close()
