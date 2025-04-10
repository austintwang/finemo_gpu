import json
import os
import warnings
from contextlib import ExitStack

import numpy as np
import h5py
import hdf5plugin
import polars as pl
import pyBigWig
import pyfaidx

from tqdm import tqdm


def load_txt(path):
    entries = []
    with open(path) as f:
        for line in f:
            item = line.rstrip("\n").split("\t")[0]
            entries.append(item)
    
    return entries


def load_mapping(path, type):
    mapping = {}
    with open(path) as f:
        for line in f:
            key, val = line.rstrip("\n").split("\t")
            mapping[key] = type(val)

    return mapping


NARROWPEAK_SCHEMA = ["chr", "peak_start", "peak_end", "peak_name", "peak_score", 
                     "peak_strand", "peak_signal", "peak_pval", "peak_qval", "peak_summit"]
NARROWPEAK_DTYPES = [pl.String, pl.Int32, pl.Int32, pl.String, pl.UInt32, 
                     pl.String, pl.Float32, pl.Float32, pl.Float32, pl.Int32] 

def load_peaks(peaks_path, chrom_order_path, half_width):
    peaks = (
        pl.scan_csv(peaks_path, has_header=False, new_columns=NARROWPEAK_SCHEMA, separator='\t', 
                    quote_char=None, schema_overrides=NARROWPEAK_DTYPES, null_values=['.', 'NA', 'null', 'NaN'])
        .select(
            chr=pl.col("chr"),
            peak_region_start=pl.col("peak_start") + pl.col("peak_summit") - half_width,
            peak_name=pl.col("peak_name")
        )
        .with_row_index(name="peak_id")
        .collect()
    )
    
    if chrom_order_path is not None:
        chrom_order = load_txt(chrom_order_path)
    else:
        chrom_order = []

    chrom_order_set = set(chrom_order)
    chrom_order_peaks = [i for i in peaks.get_column("chr").unique(maintain_order=True) if i not in chrom_order_set]
    chrom_order.extend(chrom_order_peaks)
    chrom_ind_map = {val: ind for ind, val in enumerate(chrom_order)}

    peaks = peaks.with_columns(
        pl.col("chr").replace_strict(chrom_ind_map).alias("chr_id")
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

            if b > a:
                sequences[ind,:,a:b] = one_hot_encode(sequence)

                for j, bw in enumerate(bws):
                    contrib_buffer[j,:] = np.nan_to_num(bw.values(chrom, start_adj, end_adj, numpy=True))

                contribs[ind,a:b] = np.mean(contrib_buffer, axis=0)
    
    finally:
        for bw in bws:
            bw.close()
    
    return sequences, contribs


def load_regions_from_chrombpnet_h5(h5_paths, half_width):
    with ExitStack() as stack:
        h5s = [stack.enter_context(h5py.File(i)) for i in h5_paths]

        start = h5s[0]['raw/seq'].shape[-1] // 2 - half_width
        end = start + 2 * half_width
        
        sequences = h5s[0]['raw/seq'][:,:,start:end].astype(np.int8) 
        contribs = np.mean([np.nan_to_num(f['shap/seq'][:,:,start:end]) for f in h5s], axis=0, dtype=np.float16)

    return sequences, contribs


def load_regions_from_bpnet_h5(h5_paths, half_width):
    with ExitStack() as stack:
        h5s = [stack.enter_context(h5py.File(i)) for i in h5_paths]

        start = h5s[0]['input_seqs'].shape[-2] // 2 - half_width
        end = start + 2 * half_width

        sequences = h5s[0]['input_seqs'][:,start:end,:].swapaxes(1,2).astype(np.int8) 
        contribs = np.mean([np.nan_to_num(f['hyp_scores'][:,start:end,:].swapaxes(1,2)) for f in h5s], axis=0, dtype=np.float16)

    return sequences, contribs


def load_npy_or_npz(path):
    f = np.load(path)
    if isinstance(f, np.ndarray):
        arr = f
    else:
        arr = f['arr_0']

    return arr

def load_regions_from_modisco_fmt(shaps_paths, ohe_path, half_width):
    sequences_raw = load_npy_or_npz(ohe_path)

    start = sequences_raw.shape[-1] // 2 - half_width
    end = start + 2 * half_width

    sequences = sequences_raw[:,:,start:end].astype(np.int8)

    shaps = [np.nan_to_num(load_npy_or_npz(p)[:,:,start:end]) for p in shaps_paths]
    contribs = np.mean(shaps, axis=0, dtype=np.float16)

    return sequences, contribs


def load_regions_npz(npz_path):
    data = np.load(npz_path)
    
    if "chr" not in data.keys():
        warnings.warn("No genome coordinates present in the input .npz file. Returning sequences and contributions only.")
        has_peaks = False
        num_regions = data["sequences"].shape[0]
        peak_data = {"chr": np.array(["NA"] * num_regions, dtype='U'), "chr_id": np.arange(num_regions, dtype=np.uint32), 
                       "peak_region_start": np.zeros(num_regions, dtype=np.int32), "peak_id": np.arange(num_regions, dtype=np.uint32), 
                       "peak_name": np.array(["NA"] * num_regions, dtype='U')}

    else:
        has_peaks = True
        peak_data = {"chr": data["chr"], "chr_id": data["chr_id"], "peak_region_start": data["start"],
                    "peak_id": data["peak_id"], "peak_name": data["peak_name"]}
        
    peaks_df = pl.DataFrame(peak_data)

    return data["sequences"], data["contributions"], peaks_df, has_peaks


def write_regions_npz(sequences, contributions, out_path, peaks_df=None):
    if peaks_df is None:
        warnings.warn("No genome coordinates provided. Writing sequences and contributions only.")
        np.savez_compressed(out_path, sequences=sequences, contributions=contributions)

    else:
        num_regions = peaks_df.height
        if (num_regions != sequences.shape[0]) or (num_regions != contributions.shape[0]):
            raise ValueError(f"Input sequences of shape {sequences.shape} and/or " 
                            f"input contributions of shape {contributions.shape} "
                            f"are not compatible with peak region count of {num_regions}" )

        chr_arr = peaks_df.get_column("chr").to_numpy().astype('U')
        chr_id_arr = peaks_df.get_column("chr_id").to_numpy()
        start_arr = peaks_df.get_column("peak_region_start").to_numpy()
        peak_id_arr = peaks_df.get_column("peak_id").to_numpy()
        peak_name_arr = peaks_df.get_column("peak_name").to_numpy().astype('U')
        np.savez_compressed(out_path, sequences=sequences, contributions=contributions,
                            chr=chr_arr, chr_id=chr_id_arr, start=start_arr, peak_id=peak_id_arr, peak_name=peak_name_arr)



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


def softmax(x, temp=100):
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    exp = np.exp(temp * norm_x)
    return exp / np.sum(exp, axis=0, keepdims=True)


def _motif_name_sort_key(data):
    name = data[0]
    if name.startswith("pattern_"):
        pattern_num = int(name.split("_")[-1])
        return (pattern_num,)
    else:
        return (-1, name)

MODISCO_PATTERN_GROUPS = ['pos_patterns', 'neg_patterns']

def load_modisco_motifs(modisco_h5_path, trim_threshold, motif_type, motifs_include, 
                        motif_name_map, motif_lambdas, motif_lambda_default, include_rc):
    """
    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L252-L272
    """
    motif_data_lsts = {"motif_name": [], "motif_name_orig": [], "strand": [], "motif_start": [], 
                       "motif_end": [], "motif_scale": [], "lambda": []}
    motif_lst = [] 
    trim_mask_lst = []

    if motifs_include is not None:
        motifs_include = set(motifs_include)

    if motif_name_map is None:
        motif_name_map = {}

    if motif_lambdas is None:
        motif_lambdas = {}

    if len(motif_name_map.values()) != len(set(motif_name_map.values())):
        raise ValueError("Specified motif names are not unique")

    with h5py.File(modisco_h5_path, 'r') as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            for ind, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=_motif_name_sort_key)):
                pattern_tag = f'{name}.{pattern_name}'

                if motifs_include is not None and pattern_tag not in motifs_include:
                    continue

                motif_lambda = motif_lambdas.get(pattern_tag, motif_lambda_default)
                pattern_tag_orig = pattern_tag
                pattern_tag = motif_name_map.get(pattern_tag, pattern_tag)

                cwm_raw = pattern['contrib_scores'][:].T
                cwm_norm = np.sqrt((cwm_raw**2).sum())

                cwm_fwd = cwm_raw / cwm_norm
                cwm_rev = cwm_fwd[::-1,::-1]
                start_fwd, end_fwd = trim_motif(cwm_fwd, trim_threshold)
                start_rev, end_rev = trim_motif(cwm_rev, trim_threshold)
                
                trim_mask_fwd = np.zeros(cwm_fwd.shape[1], dtype=np.int8)
                trim_mask_fwd[start_fwd:end_fwd] = 1
                trim_mask_rev = np.zeros(cwm_rev.shape[1], dtype=np.int8)
                trim_mask_rev[start_rev:end_rev] = 1

                if motif_type == "cwm":
                    motif_fwd = cwm_fwd
                    motif_rev = cwm_rev
                    motif_norm = cwm_norm

                elif motif_type == "hcwm":
                    motif_raw = pattern['hypothetical_contribs'][:].T
                    motif_norm = np.sqrt((motif_raw**2).sum())

                    motif_fwd = motif_raw / motif_norm
                    motif_rev = motif_fwd[::-1,::-1]

                elif motif_type == "pfm":
                    motif_raw = pattern['sequence'][:].T
                    motif_norm = 1

                    motif_fwd = motif_raw / np.sum(motif_raw, axis=0, keepdims=True)
                    motif_rev = motif_fwd[::-1,::-1]

                elif motif_type == "pfm_softmax":
                    motif_raw = pattern['sequence'][:].T
                    motif_norm = 1

                    motif_fwd = softmax(motif_raw)
                    motif_rev = motif_fwd[::-1,::-1]

                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_name_orig"].append(pattern_tag_orig)
                motif_data_lsts["strand"].append('+')
                motif_data_lsts["motif_start"].append(start_fwd)
                motif_data_lsts["motif_end"].append(end_fwd)
                motif_data_lsts["motif_scale"].append(motif_norm)
                motif_data_lsts["lambda"].append(motif_lambda)

                if include_rc:
                    motif_data_lsts["motif_name"].append(pattern_tag)
                    motif_data_lsts["motif_name_orig"].append(pattern_tag_orig)
                    motif_data_lsts["strand"].append('-')
                    motif_data_lsts["motif_start"].append(start_rev)
                    motif_data_lsts["motif_end"].append(end_rev)
                    motif_data_lsts["motif_scale"].append(motif_norm)
                    motif_data_lsts["lambda"].append(motif_lambda)

                    motif_lst.extend([motif_fwd, motif_rev])
                    trim_mask_lst.extend([trim_mask_fwd, trim_mask_rev])

                else:
                    motif_lst.append(motif_fwd)
                    trim_mask_lst.append(trim_mask_fwd)
                
    motifs_df = pl.DataFrame(motif_data_lsts).with_row_index(name="motif_id")
    cwms = np.stack(motif_lst, dtype=np.float16, axis=0)
    trim_masks = np.stack(trim_mask_lst, dtype=np.int8, axis=0)
    names = motifs_df.filter(pl.col("strand") == "+").get_column("motif_name").to_numpy()

    return motifs_df, cwms, trim_masks, names


def load_modisco_seqlets(modisco_h5_path, peaks_df, motifs_df, half_width, modisco_half_width, lazy=False):
    
    start_lst = []
    end_lst = []
    is_revcomp_lst = []
    strand_lst = []
    peak_id_lst = []
    pattern_tags = []

    with h5py.File(modisco_h5_path, 'r') as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            key = lambda x: int(x[0].split("_")[-1])
            for ind, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{name}.{pattern_name}'

                starts = pattern['seqlets/start'][:].astype(np.int32)
                ends = pattern['seqlets/end'][:].astype(np.int32)
                is_revcomps = pattern['seqlets/is_revcomp'][:].astype(bool)
                strands = ['+' if not i else '-' for i in is_revcomps]
                peak_ids = pattern['seqlets/example_idx'][:].astype(np.uint32)

                n_seqlets = int(pattern['seqlets/n_seqlets'][0])

                start_lst.append(starts)
                end_lst.append(ends)
                is_revcomp_lst.append(is_revcomps)
                strand_lst.extend(strands)
                peak_id_lst.append(peak_ids)
                pattern_tags.extend([pattern_tag for _ in range(n_seqlets)])

    df_data = {
        "seqlet_start": np.concatenate(start_lst),
        "seqlet_end": np.concatenate(end_lst),
        "is_revcomp": np.concatenate(is_revcomp_lst),
        "strand": strand_lst,
        "peak_id": np.concatenate(peak_id_lst),
        "motif_name_orig": pattern_tags,
    }
    
    offset = half_width - modisco_half_width

    seqlets_df = (
        pl.LazyFrame(df_data)
        .join(motifs_df.lazy(), on=("motif_name_orig", "strand"), how="inner")
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .select(
            chr=pl.col("chr"),
            chr_id=pl.col("chr_id"),
            start=pl.col("peak_region_start") + pl.col("seqlet_start") + pl.col("motif_start") + offset,
            end=pl.col("peak_region_start") + pl.col("seqlet_start") + pl.col("motif_end") + offset,
            start_untrimmed=pl.col("peak_region_start") + pl.col("seqlet_start") + offset,
            end_untrimmed=pl.col("peak_region_start") + pl.col("seqlet_end") + offset,
            is_revcomp=pl.col("is_revcomp"),
            strand=pl.col("strand"),
            motif_name=pl.col("motif_name"),
            peak_id=pl.col("peak_id"),
            peak_region_start=pl.col("peak_region_start")
        )
        .unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
    )

    seqlets_df = seqlets_df if lazy else seqlets_df.collect()

    return seqlets_df


def write_modisco_seqlets(seqlets_df, out_path):
    seqlets_df = seqlets_df.drop(["chr_id", "is_revcomp"])
    seqlets_df.write_csv(out_path, separator="\t")


HITS_DTYPES = {
    "chr": pl.String,
    "start": pl.Int32,
    "end": pl.Int32,
    "start_untrimmed": pl.Int32,
    "end_untrimmed": pl.Int32,
    "motif_name": pl.String,
    "hit_coefficient": pl.Float32,
    "hit_coefficient_global": pl.Float32,
    "hit_similarity": pl.Float32,
    "hit_correlation": pl.Float32,
    "hit_importance": pl.Float32,
    "hit_importance_sq": pl.Float32,
    "strand": pl.String,
    "peak_name": pl.String,
    "peak_id": pl.UInt32,
    
}

def load_hits(hits_path, lazy=False):
    hits_df = (
        pl.scan_csv(hits_path, separator='\t', quote_char=None, schema=HITS_DTYPES)
        .with_columns(pl.lit(1).alias("count"))
    )

    return hits_df if lazy else hits_df.collect()


def write_hits(hits_df, peaks_df, motifs_df, qc_df, out_dir, motif_width):
    os.makedirs(out_dir, exist_ok=True)
    out_path_tsv = os.path.join(out_dir, "hits.tsv")
    out_path_tsv_unique = os.path.join(out_dir, "hits_unique.tsv")
    out_path_bed = os.path.join(out_dir, "hits.bed")

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
            start_untrimmed=pl.col("peak_region_start") + pl.col("hit_start"),
            end_untrimmed=pl.col("peak_region_start") + pl.col("hit_start") + motif_width,
            motif_name=pl.col("motif_name"),
            hit_coefficient=pl.col("hit_coefficient"),
            hit_coefficient_global=pl.col("hit_coefficient") * (pl.col("global_scale")**2),
            hit_similarity=pl.col("hit_similarity"),
            hit_correlation=pl.col("hit_similarity"),
            hit_importance=pl.col("hit_importance") * pl.col("global_scale"),
            hit_importance_sq=pl.col("hit_importance_sq") * (pl.col("global_scale")**2),
            strand=pl.col("strand"),
            peak_name=pl.col("peak_name"),
            peak_id=pl.col("peak_id"),
            motif_lambda = pl.col("lambda"),
        )
        .sort(["chr_id", "start"])
        .select(HITS_DTYPES.keys())
    )

    data_unique = (
        data_all
        .unique(subset=["chr", "start", "motif_name", "strand"], maintain_order=True)
    )

    data_bed = (
        data_unique
        .select(
            chr=pl.col("chr"),
            start=pl.col("start"),
            end=pl.col("end"),
            motif_name=pl.col("motif_name"),
            score=pl.lit(0),
            strand=pl.col("strand")
        )
    )

    data_all.collect().write_csv(out_path_tsv, separator="\t")
    data_unique.collect().write_csv(out_path_tsv_unique, separator="\t")
    data_bed.collect().write_csv(out_path_bed, include_header=False, separator="\t")


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


def write_motifs_df(motifs_df, out_path):
    motifs_df.write_csv(out_path, separator="\t")


MOTIF_DTYPES = {
    "motif_id": pl.UInt32,
    "motif_name": pl.String,
    "motif_name_orig": pl.String,
    "strand": pl.String,
    "motif_start": pl.UInt32,
    "motif_end": pl.UInt32,
    "motif_scale": pl.Float32,
    "lambda": pl.Float32,
}

def load_motifs_df(motifs_path):
    motifs_df = pl.read_csv(motifs_path, separator="\t", schema=MOTIF_DTYPES)
    motif_names = motifs_df.filter(pl.col("strand") == "+").get_column("motif_name").to_numpy()

    return motifs_df, motif_names


def write_motif_cwms(cwms, out_path):
    np.save(out_path, cwms)


def load_motif_cwms(cwms_path):
    return np.load(cwms_path)


def write_params(params, out_path):
    with open(out_path, "w") as f:
        json.dump(params, f, indent=4)


def load_params(params_path):
    with open(params_path) as f:
        params = json.load(f)

    return params


def write_occ_df(occ_df, out_path):
    occ_df.write_csv(out_path, separator="\t")


def write_report_data(report_df, cwms, out_dir):
    cwms_dir = os.path.join(out_dir, "CWMs")
    os.makedirs(cwms_dir, exist_ok=True)

    for m, v in cwms.items():
        motif_dir = os.path.join(cwms_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            np.savetxt(os.path.join(motif_dir, f"{cwm_type}.txt"), cwm)

    report_df.write_csv(os.path.join(out_dir, "motif_report.tsv"), separator="\t")

