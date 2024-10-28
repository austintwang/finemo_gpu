import os
import warnings
import importlib

import numpy as np
import polars as pl
from scipy.stats import binom, false_discovery_control
import matplotlib.pyplot as plt
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from jinja2 import Template
from tqdm import tqdm

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


def plot_hit_distributions(occ_df, motif_names, plot_dir):
    motifs_dir = os.path.join(plot_dir, "motif_hit_distributions")
    os.makedirs(motifs_dir, exist_ok=True)

    for m in motif_names:
        fig, ax = plt.subplots(figsize=(6, 2))

        unique, counts = np.unique(occ_df.get_column(m), return_counts=True)
        freq = counts / counts.sum()
        num_bins = np.amax(unique, initial=0) + 1
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
    num_bins = np.amax(unique, initial=0) + 1
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

    plt.savefig(output_path, dpi=300)

    plt.close()


def discover_composites(hits_df, motifs_df, motif_names, tol_min, tol_max, fdr_thresh, odds_thresh, counts_thresh, region_len, motif_width, num_regions):
    # hits_df = hits_df.head(10000) ####
    hits_df = hits_df.with_columns(pl.col("start").cast(pl.Int32), pl.col("end").cast(pl.Int32))

    names_to_idx = {n: i for i, n in enumerate(motif_names)}
    motifs_df = (
        motifs_df.lazy()
        .select(
            motif_name=pl.col("motif_name"),
            is_revcomp=pl.col("motif_strand") == '-',
            motif_start=pl.col("motif_start"),
            motif_end=pl.col("motif_end"),
        )
    )

    comp_motifs = (
        motifs_df.join(motifs_df, how="cross")
        .with_columns(overlap=pl.col("motif_start_right") - pl.col("motif_end") + motif_width)
        .collect()
    )
    slots = np.zeros((len(motif_names), 2, len(motif_names), 2), dtype=np.int32)
    for m in comp_motifs.iter_rows(named=True):
        mleft = m["motif_name"]
        mright = m["motif_name_right"]
        sleft = int(m["is_revcomp"])
        sright = int(m["is_revcomp_right"])
        overlap = m["overlap"]
        
        num_slots = region_len - 2 * motif_width + 1 + overlap
        # print(mleft, sleft, mright, sright, num_slots) ####
        slots[names_to_idx[mleft], sleft, names_to_idx[mright], sright] = num_slots
    
    slots = (slots[:,:,:,:,None] - np.arange(tol_min, tol_max)[None,None,None,None,:]) * 2
    
    counts = np.zeros((len(motif_names), 2, len(motif_names), 2, tol_max - tol_min), dtype=np.int32)
    
    hits_by_peak = hits_df.partition_by("peak_id")
    for p in tqdm(hits_by_peak, ncols=120, desc="Evaluating candidate composite motifs"):
        # p.glimpse() ####
        p = p.lazy().with_columns(is_revcomp=pl.col("strand") == '-')
        comps = (
            p.join(p, how="cross")
            .select(
                motif_left=pl.col("motif_name"),
                motif_right=pl.col("motif_name_right"),
                is_revcomp_left=pl.col("is_revcomp"),
                is_revcomp_right=pl.col("is_revcomp_right"),
                tol=pl.col("start_right") - pl.col("end"),
            )
            .filter((pl.col("tol") >= tol_min) & (pl.col("tol") < tol_max))
            .collect()
        )
        for c in comps.iter_rows():
            mleft, mright, sleft, sright, tol = c
            counts[names_to_idx[mleft], int(sleft), names_to_idx[mright], int(sright), tol - tol_min] += 1
            counts[names_to_idx[mright], int(~sright), names_to_idx[mleft], int(~sleft), tol - tol_min] += 1

    counts_single = np.zeros(len(motif_names), dtype=np.int32)
    counts_df = hits_df.group_by("motif_name").count()
    for m, c in counts_df.iter_rows():
        counts_single[names_to_idx[m]] = c

    num_pos = region_len - motif_width + 1
    prob_exp_reg = counts_single / (num_regions * num_pos * 2)
    prob_exp_pair = prob_exp_reg[:,None] * prob_exp_reg[None,:]
    slots_total = slots * num_regions
    counts_exp = prob_exp_pair[:,None,:,None,None] * slots_total

    odds_ratio = counts / counts_exp 
    pval = binom.sf(counts, slots_total, prob_exp_pair[:,None,:,None,None])
    qval = false_discovery_control(pval)
    keeps = (qval <= fdr_thresh) & (odds_ratio >= odds_thresh) 

    comp_key_to_name = {}
    comp_key_to_strand = {}
    for i in range(len(motif_names)):
        for j in range(len(motif_names)):
            mi = motif_names[i]
            mj = motif_names[j]

            comp_key_ff = f"{mi}#0#{mj}#0"
            comp_key_fr = f"{mi}#0#{mj}#1"
            comp_key_rf = f"{mi}#1#{mj}#0"
            comp_key_rr = f"{mi}#1#{mj}#1"

            target_ff = f"{mi}#0#{mj}#0"
            target_rr = f"{mj}#0#{mi}#0"

            strand_ff = "+"
            strand_rr = "-"

            if i <= j:
                target_fr = f"{mi}#0#{mj}#1"
                target_rf = f"{mj}#1#{mi}#0"

                strand_fr = "+"
                strand_rf = "-"

            else:
                target_fr = f"{mj}#0#{mi}#1"
                target_rf = f"{mi}#1#{mj}#0"

                strand_fr = "-"
                strand_rf = "+"
                
            comp_key_to_name[comp_key_ff] = target_ff
            comp_key_to_name[comp_key_fr] = target_fr
            comp_key_to_name[comp_key_rf] = target_rf
            comp_key_to_name[comp_key_rr] = target_rr

            comp_key_to_strand[comp_key_ff] = strand_ff
            comp_key_to_strand[comp_key_fr] = strand_fr
            comp_key_to_strand[comp_key_rf] = strand_rf
            comp_key_to_strand[comp_key_rr] = strand_rr

    keeps_inds = np.argwhere(keeps)
    records = []
    for i, j, k, l, m in keeps_inds:
        comp_key = f"{motif_names[i]}#{j}#{motif_names[k]}#{l}"
        # if comp_key not in comp_key_to_name:
        #     continue

        tol = tol_min + m
        comp_name = f"{comp_key_to_name[comp_key]}#{tol}"
        strand = comp_key_to_strand[comp_key]
        counts_val = counts[i,j,k,l,m]
        counts_exp_val = counts_exp[i,j,k,l,m]
        pval_val = pval[i,j,k,l,m]
        qval_val = qval[i,j,k,l,m]
        odds_ratio_val = odds_ratio[i,j,k,l,m]

        motif_left = motif_names[i]
        strand_left = "+" if j == 0 else "-"
        motif_right = motif_names[k]
        strand_right = "+" if l == 0 else "-"

        records.append({
            "composite_name": comp_name,
            "strand": strand,
            "tol": tol,
            "counts": counts_val,
            "counts_exp": counts_exp_val,
            "odds_ratio": odds_ratio_val,
            "pval": pval_val,
            "qval": qval_val,
            "motif_name_left": motif_left,
            "strand_left": strand_left,
            "motif_name_right": motif_right,
            "strand_right": strand_right,
        })

    comp_df = pl.DataFrame(records).with_columns(pl.col("tol").cast(pl.Int32))
    comp_df_lazy = comp_df.lazy().select(["composite_name", "strand", "motif_name_left", "motif_name_right", "strand_left", "strand_right", "tol"])
    comp_df = comp_df.filter(pl.col("strand") == "+").drop("strand")

    comps_by_peak = []
    for p in tqdm(hits_by_peak, ncols=120, desc="Calling composite motif hits"):
        p = p.lazy().with_columns(is_revcomp=pl.col("strand") == '-')
        p_left = p.select(pl.all().name.suffix("_left"))
        p_right = p.select(pl.all().name.suffix("_right"))
        comps = (
            p_left.join(p_right, how="cross")
            .with_columns(
                tol = (pl.col("start_right") - pl.col("end_left")).cast(pl.Int32),
            )
            # .filter((pl.col("tol") >= tol_min) & (pl.col("tol") < tol_max))
            .join(
                comp_df_lazy,
                on=["motif_name_left", "motif_name_right", "strand_left", "strand_right", "tol"],
                how="inner"
            )
            .select(
                composite_name=pl.col("composite_name"),
                strand=pl.col("strand"),
                chr=pl.col("chr_left"),
                start=pl.col("start_left"),
                end=pl.col("end_right"),
                peak_id=pl.col("peak_id_left"),
                peak_name=pl.col("peak_name_left"),
                start_left=pl.col("start_left"),
                end_left=pl.col("end_left"),
                start_untrimmed_left=pl.col("start_untrimmed_left"),
                end_untrimmed_left=pl.col("end_untrimmed_left"),
                motif_name_left=pl.col("motif_name_left"),
                hit_coefficient_left=pl.col("hit_coefficient_left"),
                hit_coefficient_global_left=pl.col("hit_coefficient_global_left"),
                hit_correlation_left=pl.col("hit_correlation_left"),
                hit_importance_left=pl.col("hit_importance_left"),
                strand_left=pl.col("strand_left"),
                start_right=pl.col("start_right"),
                end_right=pl.col("end_right"),
                start_untrimmed_right=pl.col("start_untrimmed_right"),
                end_untrimmed_right=pl.col("end_untrimmed_right"),
                motif_name_right=pl.col("motif_name_right"),
                hit_coefficient_right=pl.col("hit_coefficient_right"),
                hit_coefficient_global_right=pl.col("hit_coefficient_global_right"),
                hit_correlation_right=pl.col("hit_correlation_right"),
                hit_importance_right=pl.col("hit_importance_right"),
            ).collect()
        )
        comps_by_peak.append(comps)

    comp_hits_df = pl.concat(comps_by_peak)

    return comp_df, comp_hits_df


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
    
    if compute_recall:
        overlaps_df = (
            hits_filtered.join(
                seqlets_df, 
                on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
                how="inner",
            )
            .collect()
        )

        seqlets_only_df = (
            seqlets_df.join(
                hits_df, 
                on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
                how="anti",
            )
            .collect()
        )

        hits_only_filtered_df = (
            hits_filtered.join(
                seqlets_df, 
                on=["chr", "start_untrimmed", "is_revcomp", "motif_name"],
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
        hits = hits_by_motif.get(m, dummy_df)
        hits_filtered = hits_fitered_by_motif.get(m, dummy_df)

        if seqlets_df is not None:
            seqlets = seqlets_by_motif.get(m, dummy_df)

        if compute_recall:
            overlaps = overlaps_by_motif.get(m, dummy_df)
            seqlets_only = seqlets_only_by_motif.get(m, dummy_df)
            hits_only_filtered = hits_only_filtered_by_motif.get(m, dummy_df)

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
                                      & (pl.col("motif_strand") == "+"), named=True)
        motif_data_rc = motifs_df.row(by_predicate=(pl.col("motif_name") == m) 
                                      & (pl.col("motif_strand") == "-"), named=True)

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
        cwm_cor = (hits_cwm * modisco_cwm).sum() / (hnorm * snorm)

        report_data[m]["cwm_correlation"] = cwm_cor

    records = [{"motif_name": k} | v for k, v in report_data.items()]
    report_df = pl.from_dicts(records)

    return report_data, report_df, cwms, cwm_trim_bounds


def get_composite_cwms(regions, comps_df, comp_hits_df, peaks_df):
    region_len = regions.shape[2]
    max_width = (
        comp_hits_df
        .select(motif_width=pl.col("end") - pl.col("start"))
        .select(pl.max("motif_width"))
        .item()
    )

    # print(comp_hits_df) ####

    comp_hits_df = (
        comp_hits_df
        .lazy()
        .with_columns(pl.col('peak_id').cast(pl.UInt32))
        .join(
            peaks_df.lazy(), on="peak_id", how="inner"
        )
        .select(
            chr=pl.col("chr"),
            start=pl.col("start"),
            end=pl.col("end"),
            is_revcomp=pl.col("strand") == '-',
            is_revcomp_int=(pl.col("strand") == '-').cast(pl.Int32),
            composite_name=pl.col("composite_name"),
            peak_region_start=pl.col("peak_region_start"),
            peak_id=pl.col("peak_id")
        )
        .with_columns(
            start_untrimmed=(
                (1 - pl.col("is_revcomp_int")) * (pl.col("start") - (pl.col("end") - pl.col("start") - max_width) // 2)
                + pl.col("is_revcomp_int") * (pl.col("start") + (pl.col("end") - pl.col("start") - max_width) // -2)
            ),
            end_untrimmed=(
                (1 - pl.col("is_revcomp_int")) * (pl.col("end") - (pl.col("end") - pl.col("start") - max_width) // -2)
                + pl.col("is_revcomp_int") * (pl.col("end") + (pl.col("end") - pl.col("start") - max_width) // 2)
            )
        )
        .filter(
            ((pl.col("start_untrimmed") - pl.col("peak_region_start")) >= 0) 
            & ((pl.col("start_untrimmed") - pl.col("peak_region_start") + max_width) < region_len)
        )
        .collect()
    )

    # print(comp_hits_df) ####

    hits_by_composite_motif = comp_hits_df.partition_by("composite_name", as_dict=True)
    cwms = {}

    for m, v in hits_by_composite_motif.items():
        cwms_fc = get_cwms(regions, v, max_width)
        cwms_rc = cwms_fc[::-1,::-1]
        cwms[m] = {
            "fc": cwms_fc,
            "rc": cwms_rc,
        }

    # print(cwms) ####

    return cwms


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
            output_path = os.path.join(motif_dir, f"{cwm_type}.png")

            fig, ax = plt.subplots(figsize=(10,2))

            if trim_bounds is not None:
                shade_bounds = trim_bounds[m][cwm_type]
            else:
                shade_bounds = None

            plot_logo(ax, cwm, alphabet, colors=colors, font_props=font, shade_bounds=shade_bounds)

            for name, spine in ax.spines.items():
                spine.set_visible(False)
            
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


def write_report(report_df, motif_names, out_path, compute_recall, use_seqlets, composites_df):
    template_str = importlib.resources.files(templates).joinpath('report.html').read_text()
    template = Template(template_str)
    if composites_df is not None:
        composites_data = composites_df.iter_rows(named=True)
    else:
        composites_data = None
    report = template.render(report_data=report_df.iter_rows(named=True), 
                             motif_names=motif_names, compute_recall=compute_recall, use_seqlets=use_seqlets, composites_data=composites_data)
    with open(out_path, "w") as f:
        f.write(report)


