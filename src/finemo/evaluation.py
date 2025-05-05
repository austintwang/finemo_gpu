import warnings

import numpy as np
import polars as pl


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

    # Ignore hits outside of region
    valid_mask = (start_idx >= 0) & (start_idx + motif_width <= regions.shape[2])
    peak_idx = peak_idx[valid_mask]
    start_idx = start_idx[valid_mask]
    is_revcomp = is_revcomp[valid_mask]

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


def seqlet_confusion(hits_df, seqlets_df, peaks_df, motif_names, motif_width):
    bin_size = motif_width

    hits_binned = (
        hits_df
        .with_columns(
            peak_id=pl.col('peak_id').cast(pl.UInt32),
            is_revcomp=pl.col("strand") == '-'
        )
        .join(
            peaks_df.lazy(), on="peak_id", how="inner"
        )
        .unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
        .select(
            chr_id=pl.col("chr_id"),
            start_bin=pl.col("start_untrimmed") // bin_size,
            end_bin=pl.col("end_untrimmed") // bin_size,
            motif_name=pl.col("motif_name")
        )
    )

    seqlets_binned = (
        seqlets_df
        .select(
            chr_id=pl.col("chr_id"),
            start_bin=pl.col("start_untrimmed") // bin_size,
            end_bin=pl.col("end_untrimmed") // bin_size,
            motif_name=pl.col("motif_name")
        )
    )

    overlaps_df = (
        seqlets_binned.join(
            hits_binned, 
            on=["chr_id", "start_bin", "end_bin"],
            how="inner",
            suffix="_hits"
        )
    )

    seqlet_counts = seqlets_binned.group_by("motif_name").len(name="num_seqlets")
    overlap_counts = overlaps_df.group_by(["motif_name", "motif_name_hits"]).len(name="num_overlaps")

    num_motifs = len(motif_names)
    confusion_mat = np.zeros((num_motifs, num_motifs), dtype=np.float32)
    name_to_idx = {m: i for i, m in enumerate(motif_names)}

    confusion_df = (
        overlap_counts
        .join(
            seqlet_counts,
            on="motif_name",
            how="inner"
        )
        .select(
            motif_name_seqlets=pl.col("motif_name"),
            motif_name_hits=pl.col("motif_name_hits"),
            frac_overlap=pl.col("num_overlaps") / pl.col("num_seqlets"),
        )
        .collect()
    )

    confusion_idx_df = (
        confusion_df
        .select(
            row_idx=pl.col("motif_name_seqlets").replace_strict(name_to_idx),
            col_idx=pl.col("motif_name_hits").replace_strict(name_to_idx),
            frac_overlap=pl.col("frac_overlap")
        )
    )

    row_idx = confusion_idx_df["row_idx"].to_numpy()
    col_idx = confusion_idx_df["col_idx"].to_numpy()
    frac_overlap = confusion_idx_df["frac_overlap"].to_numpy()

    confusion_mat[row_idx, col_idx] = frac_overlap
    
    return confusion_df, confusion_mat


