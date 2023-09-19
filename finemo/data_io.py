import json
import os

import numpy as np
import h5py
import hdf5plugin
import polars as pl
import pyBigWig
import pyfaidx

from tqdm import tqdm

NARROWPEAK_SCHEMA = ["chr", "peak_start", "peak_end", "peak_name", "peak_score", 
                     "peak_strand", "peak_signal", "peak_pval", "peak_qval", "peak_summit"]
NARROWPEAK_DTYPES = [pl.Utf8, pl.UInt32, pl.UInt32, pl.Utf8, pl.UInt32, 
                     pl.Utf8, pl.Float32, pl.Float32, pl.Float32, pl.UInt32] 

def load_peaks(peaks_path, half_width):
    peaks = (
        pl.scan_csv(peaks_path, has_header=False, new_columns=NARROWPEAK_SCHEMA, 
                    separator='\t', quote_char=None, dtypes=NARROWPEAK_DTYPES)
        .select(
            chr=pl.col("chr"),
            peak_region_start=pl.col("peak_start") + pl.col("peak_summit") - half_width,
            peak_name=pl.col("peak_name")
        )
        .with_row_count(name="peak_id")
        .collect()
    )
    
    chrom_order = peaks.get_column("chr").unique(maintain_order=True)
    chrom_ind_map = {val: ind for ind, val in enumerate(chrom_order)}

    peaks = peaks.with_columns(
        pl.col("chr").map_dict(chrom_ind_map).alias("chr_id")
    )
    
    return peaks


SEQ_ALPHABET = np.array(["A","C","G","T"], dtype="S1")

def one_hot_encode(sequence, dtype=np.int8):
    sequence = sequence.upper()

    seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
    one_hot = (seq_chararray[None,:] == SEQ_ALPHABET[:,None]).astype(dtype)

    return one_hot


def load_regions_from_peaks(peaks, fa_path, bw_paths, half_width):
    num_peaks = peaks.height

    sequences = np.zeros((num_peaks, 4, half_width * 2), dtype=np.int8)
    contribs = np.zeros((num_peaks, half_width * 2), dtype=np.float16)

    genome = pyfaidx.Fasta(fa_path, one_based_attributes=False)
    
    bws = [pyBigWig.open(i) for i in bw_paths]
    contrib_buffer = np.zeros((len(bw_paths), half_width * 2), dtype=np.float16)

    try:
        for ind, row in tqdm(enumerate(peaks.iter_rows(named=True)), disable=None, unit="regions", total=num_peaks):
            chrom = row["chr"]
            start = row["peak_region_start"]
            end = start + 2 * half_width
            
            sequence_data = genome[chrom][start:end]
            sequence = sequence_data.seq
            start_adj = sequence_data.start
            end_adj = sequence_data.end
            a = start_adj - start
            b = end_adj - start

            sequences[ind,:,a:b] = one_hot_encode(sequence)

            for j, bw in enumerate(bws):
                contrib_buffer[j,:] = bw.values(chrom, start_adj, end_adj, numpy=True)

            contribs[ind,a:b] = np.nanmean(contrib_buffer, axis=0)
    
    finally:
        for bw in bws:
            bw.close()
    
    return sequences, contribs


def load_regions_npz(npz_path):
    data = np.load(npz_path)

    return data["sequences"], data["contributions"]


def write_regions_npz(sequences, contributions, out_path):
    np.savez(out_path, sequences=sequences, contributions=contributions)


def trim_motif(cwm, trim_threshold):
    """
    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L213-L236
    """
    score = np.sum(np.abs(cwm), axis=0)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.nonzero(score >= trim_thresh)
    start = max(np.min(pass_inds) - 4, 0)
    end = min(np.max(pass_inds) + 4 + 1, len(score))

    return start, end


MODISCO_PATTERN_GROUPS = ['pos_patterns', 'neg_patterns']

def load_modisco_motifs(modisco_h5_path, trim_threshold):
    """
    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L252-L272
    """
    motif_data_lsts = {"motif_id": [], "motif_name": [], "motif_strand": [], 
                       "motif_start": [], "motif_end": [], "motif_scale": []}
    cwm_lst = [] 

    with h5py.File(modisco_h5_path, 'r') as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            key = lambda x: int(x[0].split("_")[-1])
            for ind, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{name}.{pattern_name}'

                cwm_raw = pattern['contrib_scores'][:].T
                cwm_norm = np.sqrt((cwm_raw**2).sum())

                cwm_fwd = cwm_raw / cwm_norm
                cwm_rev = cwm_fwd[::-1,::-1]
                start_fwd, end_fwd = trim_motif(cwm_fwd, trim_threshold)
                start_rev, end_rev = trim_motif(cwm_rev, trim_threshold)

                motif_data_lsts["motif_id"].append(2 * ind)
                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_strand"].append('+')
                motif_data_lsts["motif_start"].append(start_fwd)
                motif_data_lsts["motif_end"].append(end_fwd)
                motif_data_lsts["motif_scale"].append(cwm_norm)

                motif_data_lsts["motif_id"].append(2 * ind + 1)
                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_strand"].append('-')
                motif_data_lsts["motif_start"].append(start_rev)
                motif_data_lsts["motif_end"].append(end_rev)
                motif_data_lsts["motif_scale"].append(cwm_norm)

                cwm_lst.extend([cwm_fwd, cwm_rev])

    motifs_df = pl.DataFrame(motif_data_lsts)
    cwms = np.stack(cwm_lst, dtype=np.float16, axis=1)

    return motifs_df, cwms


def write_hits(hits_df, peaks_df, motifs_df, qc_df, out_path_tsv, out_path_bed, half_width):
    data_all = (
        hits_df
        .lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .join(qc_df.lazy(), on="peak_id", how="inner")
        .join(motifs_df.lazy(), on="motif_id", how="inner")
        .select(
            chr_id=pl.col("chr_id"),
            chr=pl.col("chr"),
            start=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_start"),
            end=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_end"),
            motif_name=pl.col("motif_name"),
            hit_score_scaled=pl.col("hit_score"),
            hit_score_unscaled=pl.col("hit_score") * pl.col("contrib_scale") * pl.col("motif_scale"),
            strand=pl.col("motif_strand"),
            peak_name=pl.col("peak_name"),
            peak_id=pl.col("peak_id"),
            peak_summit_distance=pl.col("hit_start") - half_width,
        )
        .sort(["chr_id", "start"])
        .drop("chr_id")
    )

    data_tsv = data_all.collect()
    data_tsv.write_csv(out_path_tsv, separator="\t")

    data_bed = (
        data_all
        .select(["chr", "start", "end", "motif_name", "strand"])        
        .unique(maintain_order=True)
        .collect()
    )
    data_bed.write_csv(out_path_bed, has_header=False, separator="\t")


def load_hits(hits_path):
    hits_df = (
        pl.scan_csv(hits_path, separator='\t', quote_char=None)
        .select(["motif_name", "hit_score", "peak_id"])
        .with_column(pl.lit(1).alias("count"))
        .collect()
    )

    return hits_df


def write_qc(qc_df, peaks_df, out_path):
    df = (
        qc_df
        .lazy()
        .with_row_count(name="peak_id")
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .sort(["chr_id", "start"])
        .drop("chr_id")
        .collect()
    )
    df.write_csv(out_path, separator="\t")


def write_params(out_path):
    with open(out_path, "w") as f:
        json.dump(out_path, f, indent=4)


def write_occ_df(occ_df, out_path):
    occ_df.write_csv(out_path, separator="\t")


def write_coocc_mats(coocc_counts, coocc_sigs, motif_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    counts_path = os.path.join(out_dir, "cooccurrence_counts.txt")
    sigs_path = os.path.join(out_dir, "cooccurrence_neg_log10p.txt")
    names_path = os.path.join(out_dir, "motif_name.txt")

    np.savetxt(counts_path, coocc_counts, delimiter="\t")
    np.savetxt(sigs_path, coocc_sigs, delimiter="\t")
    
    with open(names_path, "w") as f:
        for n in motif_names:
            f.write(f"{n}\n")



