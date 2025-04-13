import os
import warnings
import importlib

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from jinja2 import Template

from . import templates


def abbreviate_motif_name(name):
    try:
        group, motif = name.split(".")

        if group == "pos_patterns":
            group_short = "+"
        elif group == "neg_patterns":
            group_short = "-"
        else:
            raise Exception

        motif_num = motif.split("_")[1]

        return f"{group_short}/{motif_num}"
    
    except:
        return name


def get_motif_occurences(hits_df, motif_names):
    occ_df = (
        hits_df
        .collect()
        .pivot(index="peak_id", columns="motif_name", values="count", aggregate_function="sum")
        .fill_null(0)
    )

    missing_cols = set(motif_names) - set(occ_df.columns)
    occ_df = (
        occ_df
        .with_columns([pl.lit(0).alias(m) for m in missing_cols])
        .with_columns(total=pl.sum_horizontal(*motif_names))
        .sort(["peak_id"])
    )

    num_peaks = occ_df.height
    num_motifs = len(motif_names)

    occ_mat = np.zeros((num_peaks, num_motifs), dtype=np.int16)
    for i, m in enumerate(motif_names):
        occ_mat[:,i] = occ_df.get_column(m).to_numpy()

    occ_bin = (occ_mat > 0).astype(np.int32)
    coocc = occ_bin.T @ occ_bin

    return occ_df, coocc


def plot_hit_stat_distributions(hits_df, motif_names, plot_dir):
    hits_df = hits_df.collect()
    hits_by_motif = hits_df.partition_by("motif_name", as_dict=True)
    dummy_df = hits_df.clear()

    motifs_dir = os.path.join(plot_dir, "motif_stat_distributions")
    os.makedirs(motifs_dir, exist_ok=True)
    for m in motif_names:
        hits = hits_by_motif.get((m,), dummy_df)
        coefficients = hits.get_column("hit_coefficient_global").to_numpy()
        similarities = hits.get_column("hit_similarity").to_numpy()
        importances = hits.get_column("hit_importance").to_numpy()

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(coefficients, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_coefficients.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_coefficients.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(similarities, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_similarities.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_similarities.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(importances, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_importances.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_importances.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)


def plot_hit_peak_distributions(occ_df, motif_names, plot_dir):
    motifs_dir = os.path.join(plot_dir, "motif_hit_distributions")
    os.makedirs(motifs_dir, exist_ok=True)

    for m in motif_names:
        fig, ax = plt.subplots(figsize=(5, 2))

        unique, counts = np.unique(occ_df.get_column(m), return_counts=True)
        freq = counts / counts.sum()
        num_bins = np.amax(unique, initial=0) + 1
        x = np.arange(num_bins)
        y = np.zeros(num_bins)
        y[unique] = freq
        ax.bar(x, y)

        output_path_png = os.path.join(motifs_dir, f"{m}.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}.svg")
        plt.savefig(output_path_svg)

        plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4))

    unique, counts = np.unique(occ_df.get_column("total"), return_counts=True)
    freq = counts / counts.sum()
    num_bins = np.amax(unique, initial=0) + 1
    x = np.arange(num_bins)
    y = np.zeros(num_bins)
    y[unique] = freq
    ax.bar(x, y)

    ax.set_xlabel("Total hits per region")
    ax.set_ylabel("Frequency")

    output_path_png = os.path.join(plot_dir, "total_hit_distribution.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(plot_dir, "total_hit_distribution.svg")
    plt.savefig(output_path_svg, dpi=300)

    plt.close(fig)


def plot_peak_motif_indicator_heatmap(peak_hit_counts, motif_names, output_dir):
    """
    Plots a simple indicator heatmap of the motifs in each peak.
    """
    cov_norm = 1 / np.sqrt(np.diag(peak_hit_counts))
    matrix = peak_hit_counts * cov_norm[:,None] * cov_norm[None,:]
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]

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

    output_path_png = os.path.join(output_dir, "motif_cooocurrence.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "motif_cooocurrence.svg")
    plt.savefig(output_path_svg)

    plt.close()


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

    row_idx = peak_idx[:,None,None]
    pos_idx = start_idx[:,None,None] + np.zeros((1,1,motif_width), dtype=int)
    pos_idx[~is_revcomp,:,:] += np.arange(motif_width)[None,None,:]
    pos_idx[is_revcomp,:,:] += np.arange(motif_width)[None,None,::-1]
    nuc_idx = np.zeros((peak_idx.shape[0],4,1), dtype=int)
    nuc_idx[~is_revcomp,:,:] += np.arange(4)[None,:,None]
    nuc_idx[is_revcomp,:,:] += np.arange(4)[None,::-1,None]

    seqs = regions[row_idx, nuc_idx, pos_idx]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        cwms = seqs.mean(axis=0)

    return cwms


def tfmodisco_comparison(regions, hits_df, peaks_df, seqlets_df, motifs_df, cwms_modisco, 
                         motif_names, modisco_half_width, motif_width, compute_recall):
    hits_df = (
        hits_df
        .with_columns(pl.col('peak_id').cast(pl.UInt32))
        .join(
            peaks_df.lazy(), on="peak_id", how="inner"
        )
        .select(
            chr_id=pl.col("chr_id"),
            start_untrimmed=pl.col("start_untrimmed"),
            end_untrimmed=pl.col("end_untrimmed"),
            is_revcomp=pl.col("strand") == '-',
            motif_name=pl.col("motif_name"),
            peak_region_start=pl.col("peak_region_start"),
            peak_id=pl.col("peak_id")
        )
    )

    hits_unique = hits_df.unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
    
    region_len = regions.shape[2]
    center = region_len / 2
    hits_filtered = (
        hits_df
        .filter(
            ((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= (center - modisco_half_width)) 
            & ((pl.col("end_untrimmed") - pl.col("peak_region_start")) <= (center + modisco_half_width))
        )
        .unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
    )
    
    if compute_recall:
        overlaps_df = (
            hits_filtered.join(
                seqlets_df, 
                on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
                how="inner",
            )
            .collect()
        )

        seqlets_only_df = (
            seqlets_df.join(
                hits_df, 
                on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
                how="anti",
            )
            .collect()
        )

        hits_only_filtered_df = (
            hits_filtered.join(
                seqlets_df, 
                on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
                how="anti",
            )
            .collect()
        )

    hits_by_motif = hits_unique.collect().partition_by("motif_name", as_dict=True)
    hits_fitered_by_motif = hits_filtered.collect().partition_by("motif_name", as_dict=True)

    if seqlets_df is not None:
        seqlets_by_motif = seqlets_df.collect().partition_by("motif_name", as_dict=True)

    if compute_recall:
        overlaps_by_motif = overlaps_df.partition_by("motif_name", as_dict=True)
        seqlets_only_by_motif = seqlets_only_df.partition_by("motif_name", as_dict=True)
        hits_only_filtered_by_motif = hits_only_filtered_df.partition_by("motif_name", as_dict=True)

    report_data = {}
    cwms = {}
    cwm_trim_bounds = {}
    dummy_df = hits_df.clear().collect()
    for m in motif_names:
        hits = hits_by_motif.get((m,), dummy_df)
        hits_filtered = hits_fitered_by_motif.get((m,), dummy_df)

        if seqlets_df is not None:
            seqlets = seqlets_by_motif.get((m,), dummy_df)

        if compute_recall:
            overlaps = overlaps_by_motif.get((m,), dummy_df)
            seqlets_only = seqlets_only_by_motif.get((m,), dummy_df)
            hits_only_filtered = hits_only_filtered_by_motif.get((m,), dummy_df)

        report_data[m] = {
            "num_hits_total": hits.height,
            "num_hits_restricted": hits_filtered.height,
        }

        if seqlets_df is not None:
            report_data[m]["num_seqlets"] = seqlets.height

        if compute_recall:
            report_data[m] |= {
                "num_overlaps": overlaps.height,
                "num_seqlets_only": seqlets_only.height,
                "num_hits_restricted_only": hits_only_filtered.height,
                "seqlet_recall": np.float64(overlaps.height) / seqlets.height
            }

        motif_data_fc = motifs_df.row(by_predicate=(pl.col("motif_name") == m) 
                                      & (pl.col("strand") == "+"), named=True)
        motif_data_rc = motifs_df.row(by_predicate=(pl.col("motif_name") == m) 
                                      & (pl.col("strand") == "-"), named=True)

        cwms[m] = {
            "hits_fc": get_cwms(regions, hits, motif_width),
            "modisco_fc": cwms_modisco[motif_data_fc["motif_id"]],
            "modisco_rc": cwms_modisco[motif_data_rc["motif_id"]],
        }
        cwms[m]["hits_rc"] = cwms[m]["hits_fc"][::-1,::-1]

        if compute_recall:
            cwms[m] |= {
                "seqlets_only": get_cwms(regions, seqlets_only, motif_width),
                "hits_restricted_only": get_cwms(regions, hits_only_filtered, motif_width),
            }

        bounds_fc = (motif_data_fc["motif_start"], motif_data_fc["motif_end"])
        bounds_rc = (motif_data_rc["motif_start"], motif_data_rc["motif_end"])
        
        cwm_trim_bounds[m] = {
            "hits_fc": bounds_fc,
            "modisco_fc": bounds_fc,
            "modisco_rc": bounds_rc,
            "hits_rc": bounds_rc
        }

        if compute_recall:
            cwm_trim_bounds[m] |= {
                "seqlets_only": bounds_fc,
                "hits_restricted_only": bounds_fc,
            }
        
        hits_cwm = cwms[m]["hits_fc"]
        modisco_cwm = cwms[m]["modisco_fc"]
        hnorm = np.sqrt((hits_cwm**2).sum())
        snorm = np.sqrt((modisco_cwm**2).sum())
        cwm_sim = (hits_cwm * modisco_cwm).sum() / (hnorm * snorm)

        report_data[m]["cwm_similarity"] = cwm_sim

    records = [{"motif_name": k} | v for k, v in report_data.items()]
    report_df = pl.from_dicts(records)

    return report_data, report_df, cwms, cwm_trim_bounds


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

        self.path = stretch.transform_path(path_orig)

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        return renderer.draw_path(gc, self.path, affine, rgbFace)


def plot_logo(ax, heights, glyphs, colors=None, font_props=None, shade_bounds=None):
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
        ax.bar(x, height, 0.95, bottom=bottom, 
               path_effects=[LogoGlyph(glyph, font_props=font_props)], color=colors[glyph])

    if shade_bounds is not None:
        start, end = shade_bounds
        ax.axvspan(start - 0.5, end - 0.5, color='0.9', zorder=-1)

    ax.axhline(zorder=-1, linewidth=0.5, color='black')


LOGO_ALPHABET = 'ACGT'
LOGO_COLORS = {"A": '#109648', "C": '#255C99', "G": '#F7B32B', "T": '#D62839'}
LOGO_FONT = FontProperties(weight="bold")

def plot_cwms(cwms, trim_bounds, out_dir, alphabet=LOGO_ALPHABET, colors=LOGO_COLORS, font=LOGO_FONT):
    for m, v in cwms.items():
        motif_dir = os.path.join(out_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            fig, ax = plt.subplots(figsize=(10,2))

            plot_logo(ax, cwm, alphabet, colors=colors, font_props=font, shade_bounds=trim_bounds[m][cwm_type])

            for name, spine in ax.spines.items():
                spine.set_visible(False)
            
            output_path_png = os.path.join(motif_dir, f"{cwm_type}.png")
            plt.savefig(output_path_png, dpi=100)
            output_path_svg = os.path.join(motif_dir, f"{cwm_type}.svg")
            plt.savefig(output_path_svg)

            plt.close(fig)


def plot_hit_vs_seqlet_counts(recall_data, output_dir):
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

    output_path_png = os.path.join(output_dir, "hit_vs_seqlet_counts.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "hit_vs_seqlet_counts.svg")
    plt.savefig(output_path_svg)

    plt.close()


def write_report(report_df, motif_names, out_path, compute_recall, use_seqlets):
    template_str = importlib.resources.files(templates).joinpath('report.html').read_text()
    template = Template(template_str)
    report = template.render(report_data=report_df.iter_rows(named=True), 
                             motif_names=motif_names, compute_recall=compute_recall, 
                             use_seqlets=use_seqlets)
    with open(out_path, "w") as f:
        f.write(report)


