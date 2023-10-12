import json
import os
from contextlib import ExitStack

import numpy as np
import h5py
import hdf5plugin
import polars as pl
import pyBigWig
import pyfaidx

from tqdm import tqdm, trange

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


def load_regions_from_bw(peaks, fa_path, bw_paths, half_width):
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


def load_regions_from_h5(peaks, h5_paths, half_width):
    num_peaks = peaks.height

    sequences = np.zeros((num_peaks, 4, half_width * 2), dtype=np.int8)
    contribs = np.zeros((num_peaks, 4, half_width * 2), dtype=np.float16)

    with ExitStack() as stack:
        h5s = [stack.enter_context(h5py.File(i)) for i in h5_paths]

        start = h5s[0]['raw/seq'].shape[-1] // 2 - half_width
        end = start + 2 * half_width
        
        sequences = h5s[0]['raw/seq'][:,:,start:end] 
        contribs = np.nanmean([f['shap/seq'][:,:,start:end] for f in h5s], axis=0)

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
    start = max(np.min(pass_inds), 0)
    end = min(np.max(pass_inds) + 1, len(score))

    return start, end


MODISCO_PATTERN_GROUPS = ['pos_patterns', 'neg_patterns']

def load_modisco_motifs(modisco_h5_path, trim_threshold, use_hypothetical):
    """
    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L252-L272
    """
    motif_data_lsts = {"motif_name": [], "motif_strand": [], 
                       "motif_start": [], "motif_end": [], "motif_scale": []}
    cwm_lst = [] 

    if use_hypothetical:
        cwm_key = 'hypothetical_contribs'
    else:
        cwm_key = 'contrib_scores'

    with h5py.File(modisco_h5_path, 'r') as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            key = lambda x: int(x[0].split("_")[-1])
            for ind, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{name}.{pattern_name}'

                cwm_raw = pattern[cwm_key][:].T
                cwm_norm = np.sqrt((cwm_raw**2).sum())

                cwm_fwd = cwm_raw / cwm_norm
                cwm_rev = cwm_fwd[::-1,::-1]
                start_fwd, end_fwd = trim_motif(cwm_fwd, trim_threshold)
                start_rev, end_rev = trim_motif(cwm_rev, trim_threshold)

                # motif_data_lsts["motif_id"].append(2 * ind)
                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_strand"].append('+')
                motif_data_lsts["motif_start"].append(start_fwd)
                motif_data_lsts["motif_end"].append(end_fwd)
                motif_data_lsts["motif_scale"].append(cwm_norm)

                # motif_data_lsts["motif_id"].append(2 * ind + 1)
                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_strand"].append('-')
                motif_data_lsts["motif_start"].append(start_rev)
                motif_data_lsts["motif_end"].append(end_rev)
                motif_data_lsts["motif_scale"].append(cwm_norm)

                cwm_lst.extend([cwm_fwd, cwm_rev])

    motifs_df = pl.DataFrame(motif_data_lsts).with_row_count(name="motif_id")
    cwms = np.stack(cwm_lst, dtype=np.float16, axis=1)

    return motifs_df, cwms


def load_hits(hits_path, lazy=False):
    hits_df = (
        pl.scan_csv(hits_path, separator='\t', quote_char=None)
        .with_columns(pl.lit(1).alias("count"))
    )

    # if deduplicate:
    #     hits_df = hits_df.unique(subset=["chr", "start", "motif_name", "strand"])

    # print(hits_df.head().collect()) ####

    return hits_df if lazy else hits_df.collect()


def load_modisco_seqlets(modisco_h5_path, peaks_df, lazy=False):
    
    start_lst = []
    end_lst = []
    is_revcomp_lst = []
    peak_id_lst = []
    pattern_tags = []

    # seqlet_counts = {}
    with h5py.File(modisco_h5_path, 'r') as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            key = lambda x: int(x[0].split("_")[-1])
            for ind, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{name}.{pattern_name}'

                starts = pattern['seqlets/start'][:]
                ends = pattern['seqlets/end'][:]
                is_revcomps = pattern['seqlets/is_revcomp'][:]
                peak_ids = pattern['seqlets/example_idx'][:].astype(np.uint32)

                n_seqlets = int(pattern['seqlets/n_seqlets'][0])

                start_lst.append(starts)
                end_lst.append(ends)
                is_revcomp_lst.append(is_revcomps)
                peak_id_lst.append(peak_ids)
                pattern_tags.extend([pattern_tag for _ in range(n_seqlets)])

                # seqlet_counts[pattern_tag] = n_seqlets

    df_data = {
        "seqlet_start": np.concatenate(start_lst),
        "seqlet_end": np.concatenate(end_lst),
        "is_revcomp": np.concatenate(is_revcomp_lst),
        "peak_id": np.concatenate(peak_id_lst),
        "motif_name": pattern_tags,
    }
    
    seqlets_df = (
        pl.LazyFrame(df_data)
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .select(
            chr=pl.col("chr"),
            start_untrimmed=pl.col("peak_region_start") + pl.col("seqlet_start"),
            end_untrimmed=pl.col("peak_region_start") + pl.col("seqlet_end"),
            is_revcomp=pl.col("is_revcomp"),
            motif_name= pl.col("motif_name")
        )
        .unique()
        .with_columns(pl.lit(1).alias('seqlet_indicator'))
    )

    # print(seqlets_df.head().collect()) ####

    seqlets_df = seqlets_df if lazy else seqlets_df.collect()

    return seqlets_df


def load_chip_importances(fa_path, bw_path, hits_df, cwm_fwd, cwm_rev, motif_name):
    hits_motif = (
        hits_df
        .filter(pl.col("motif_name") == motif_name)
        .select(
            ["chr", "start", "end", "strand", "hit_score_raw", 
             "hit_score_unscaled", "hit_score_scaled"]
        )
        .unique()
        .collect()
    )

    chip_importance = np.zeros(hits_motif.length)
    genome = pyfaidx.Fasta(fa_path, one_based_attributes=False)
    try:
        bw = pyBigWig.open(bw_path)
        for i, r in tqdm(enumerate(hits_motif.iter_rows(named=True)), disable=None, unit="hits"):
            chrom = r["chrom"]
            start = r["start"]
            end = r["end"]
            strand = r["strand"]

            sequence = str(genome[chrom][start:end])
            one_hot = one_hot_encode([sequence])
            contribs = np.nan_to_num(bw.values(chrom, start, end))
            if strand == "+":
                val_bp = np.mean((contribs * np.sum(cwm_fwd * one_hot, axis=0)))
            else:
                val_bp = np.mean((contribs * np.sum(cwm_rev * one_hot, axis=0)))

            chip_importance[i] = val_bp

    finally:
        bw.close()

    df = hits_motif.with_column(pl.Series(name="chip_importance", values=chip_importance)) 




def write_hits(hits_df, peaks_df, motifs_df, out_path_tsv, out_path_bed, motif_width):

    data_all = (
        hits_df
        .lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        # .join(qc_df.lazy(), on="peak_id", how="inner")
        .join(motifs_df.lazy(), on="motif_id", how="inner")
        .select(
            chr_id=pl.col("chr_id"),
            chr=pl.col("chr"),
            start=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_start"),
            end=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_end"),
            start_untrimmed=pl.col("peak_region_start") + pl.col("hit_start"),
            end_untrimmed=pl.col("peak_region_start") + pl.col("hit_start") + motif_width,
            motif_name=pl.col("motif_name"),
            hit_score_raw=pl.col("hit_score_raw"),
            hit_score_unscaled=pl.col("hit_score_unscaled"),
            hit_score_scaled=pl.col("hit_score_unscaled")**2 / pl.col("hit_score_raw"),
            strand=pl.col("motif_strand"),
            peak_name=pl.col("peak_name"),
            peak_id=pl.col("peak_id"),
            # peak_summit_distance=pl.col("hit_start") + pl.col("motif_start") - half_width,
        )
        .sort(["chr_id", "start"])
        .drop("chr_id")
    )

    data_tsv = data_all.collect()
    data_tsv.write_csv(out_path_tsv, separator="\t")

    data_bed = (
        data_all
        .with_columns(pl.lit(0).alias('dummy_score'))
        .select(["chr", "start", "end", "motif_name", "dummy_score", "strand"])        
        .unique(maintain_order=True)
        .collect()
    )
    data_bed.write_csv(out_path_bed, has_header=False, separator="\t")


def write_qc(qc_df, peaks_df, out_path):
    df = (
        qc_df
        .lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .sort(["chr_id", "peak_region_start"])
        .drop("chr_id")
        .collect()
    )
    df.write_csv(out_path, separator="\t")


def write_params(params, out_path):
    with open(out_path, "w") as f:
        json.dump(params, f, indent=4)


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


def write_modisco_recall(seqlet_recalls, overlaps_df, nonoverlaps_df, seqlet_counts, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    overlaps_df.write_csv(os.path.join(out_dir, "overlaps.tsv"), separator="\t")
    nonoverlaps_df.write_csv(os.path.join(out_dir, "non_overlaps.tsv"), separator="\t")

    with open(os.path.join(out_dir, "seqlet_counts.json"), "w") as f:
        json.dump(seqlet_counts, f, indent=4)

    for k, v in seqlet_recalls.items():
        np.savetxt(os.path.join(out_dir, f'{k}.txt.gz'), v)
