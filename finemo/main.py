from . import data_io, hitcaller, visualization

import os
import argparse

import polars as pl


def extract_regions(peaks_path, fa_path, bw_paths, out_path, region_width):
    half_width = region_width // 2

    peaks_df = data_io.load_peaks(peaks_path, half_width)
    sequences, contribs = data_io.load_regions_from_peaks(peaks_df, fa_path, bw_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path)


def call_hits(regions_path, peaks_path, modisco_h5_path, out_dir, cwm_trim_threshold, 
              alpha, l1_ratio, step_size, convergence_tol, max_steps, batch_size, device):
    
    sequences, contribs = data_io.load_regions_npz(regions_path)

    if (sequences.shape[0] != contribs.shape[0]) or (sequences.shape[2] != contribs.shape[1]):
        raise ValueError(f"Input sequences of shape {sequences.shape} is not compatible with " 
                         f"input contributions of shape {contribs.shape}.")

    region_width = sequences.shape[2]
    if region_width % 2 != 0:
        raise ValueError(f"Region width of {region_width} is not divisible by 2.")
    
    half_width = region_width // 2

    peaks_df = data_io.load_peaks(peaks_path, half_width)
    num_regions = peaks_df.height
    if (num_regions != sequences.shape[0]) or (num_regions != contribs.shape[0]):
        raise ValueError(f"Input sequences of shape {sequences.shape} and/or " 
                         f"input contributions of shape {contribs.shape} "
                         f"are not compatible with region count of {num_regions}" )

    motifs_df, cwms, motif_norm = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold)
    num_motifs = cwms.shape[1]
    motif_width = cwms.shape[2]

    hits, qc, contrib_norm = hitcaller.fit_contribs(cwms, contribs, sequences, alpha, l1_ratio, step_size, 
                                                    convergence_tol, max_steps, batch_size, device)
    hits_df = pl.DataFrame(hits)
    qc_df = pl.DataFrame(qc)

    os.makedirs(out_dir, exist_ok=True)
    out_dir_tsv = os.path.join(out_dir, "hits.tsv")
    out_dir_bed = os.path.join(out_dir, "hits.bed")
    out_dir_qc = os.path.join(out_dir, "peaks_qc.tsv")
    data_io.write_hits(hits_df, peaks_df, motifs_df, 
                       out_dir_tsv, out_dir_bed, half_width, motif_norm, contrib_norm)
    data_io.write_qc(qc_df, peaks_df, out_dir_qc)

    params = {
        "inputs": {
            "regions_path": regions_path,
            "peaks_path": peaks_path,
            "modisco_h5_path": modisco_h5_path,
        },
        "outputs": {
            "out_dir": out_dir,
        },
        "provided_params": {
            "cwm_trim_threshold": cwm_trim_threshold,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "step_size": step_size,
            "convergence_tol": convergence_tol,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "device": device,
        },
        "inferred_params": {
            "region_width": region_width,
            "num_regions": num_regions,
            "untrimmed_motif_width": motif_width,
            "num_motifs": num_motifs,
        }
    }
    out_dir_params = os.path.join(out_dir, "parameters.json")
    data_io.write_params(params, out_dir_params)


def visualize(hits_path, out_dir):
    hits_df = data_io.load_hits(hits_path)
    num_peaks = hits_df.height

    occ_df, occ_mat, occ_bin, coocc, motif_names = visualization.get_motif_occurences(hits_df)
    peak_order = visualization.cluster_matrix_indices(occ_mat)
    motif_order = visualization.cluster_matrix_indices(occ_mat.T)
    coocc_nlp = visualization.cooccurrence_sigs(coocc, num_peaks)

    occ_path = os.path.join(out_dir, "motif_occurrences.tsv")
    data_io.write_occ_df(occ_df, occ_path)
    
    coocc_dir = os.path.join(out_dir, "motif_cooccurrence_matrices")
    data_io.write_coocc_mats(coocc, coocc_nlp, motif_names, coocc_dir)

    score_dist_dir = os.path.join(out_dir, "hit_score_distributions")
    visualization.plot_score_distributions(hits_df, score_dist_dir)

    hits_cdf_dir = os.path.join(out_dir, "motif_peak_hit_cdfs")
    visualization.plot_homotypic_densities(occ_mat, motif_names, hits_cdf_dir)

    frac_peaks_path = os.path.join(out_dir, "frac_peaks_with_motif.png")
    visualization.plot_frac_peaks(occ_bin, motif_names, frac_peaks_path)

    occ_path = os.path.join(out_dir, "peak_motif_occurrences.png")
    visualization.plot_occurrence(occ_mat, motif_names, peak_order, motif_order, occ_path)

    coocc_counts_path = os.path.join(out_dir, "motif_cooccurrence_counts.png")
    visualization.plot_cooccurrence_counts(coocc, motif_names, motif_order, coocc_counts_path)

    coocc_sigs_path = os.path.join(out_dir, "motif_cooccurrence_neg_log10p.png")
    visualization.plot_cooccurrence_sigs(coocc_nlp, motif_names, motif_order, coocc_sigs_path)


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Must be either `call-hits`, `extract-regions`, or `visualize`.", required=True, dest='cmd')

    call_hits_parser = subparsers.add_parser("call-hits", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Call hits on provided sequences, contributions, and motifs.")

    call_hits_parser.add_argument("-r", "--regions", type=str, required=True,
        help="A .npz file of input sequences and contributions. Can be generated using `finemo extract_regions`.")
    call_hits_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A sorted peak regions file in ENCODE NarrowPeak format. These should exactly match the regions in `--regions`.")
    call_hits_parser.add_argument("-m", "--modisco-h5", type=str, required=True,
        help="A tfmodisco-lite output H5 file of motif patterns.")
    
    call_hits_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    
    call_hits_parser.add_argument("-t", "--cwm-trim-threshold", type=float, default=0.3,
        help="Trim treshold for determining motif start and end positions within the full input motif CWM's.")
    call_hits_parser.add_argument("-a", "--alpha", type=float, default=10.,
        help="Total regularization weight.")
    call_hits_parser.add_argument("-l", "--l1-ratio", type=float, default=0.99,
        help="Elastic net mixing parameter. This specifies the fraction of `alpha` used for L1 regularization.")
    call_hits_parser.add_argument("-s", "--step-size", type=float, default=0.75,
        help="Optimizer step size.")
    call_hits_parser.add_argument("-c", "--convergence-tol", type=float, default=0.001,
        help="Tolerance for assessing convergence. The optimizer exits when the dual gap is less than the tolerance.")
    call_hits_parser.add_argument("-S", "--max-steps", type=int, default=1000,
        help="Maximum optimizer steps.")
    call_hits_parser.add_argument("-b", "--batch-size", type=int, default=1000,
        help="Batch size for optimization.")
    call_hits_parser.add_argument("-d", "--device", type=str, default="cuda",
        help="Pytorch device name. Set to `cpu` to run without a GPU.")
    
    extract_regions_parser = subparsers.add_parser("extract-regions", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from FASTA and bigwig files.")
    
    extract_regions_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A sorted peak regions file in ENCODE NarrowPeak format.")
    extract_regions_parser.add_argument("-f", "--fasta", type=str, required=True,
        help="A genome FASTA file. An .fai index file will be built in the same directory as the fasta file if one does not already exist.")
    extract_regions_parser.add_argument("-b", "--bigwigs", type=str, required=True, nargs='+',
        help="One or more bigwig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the region extracted around each peak summit.")
    
    visualize_parser = subparsers.add_parser("visualize", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from FASTA and bigwig files.")
    
    visualize_parser.add_argument("-H", "--hits", type=str, required=True,
        help="The `hits.tsv` output file from `call-hits`.")
    
    visualize_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    
    args = parser.parse_args()

    if args.cmd == "call-hits":
        call_hits(args.regions, args.peaks, args.modisco_h5, args.out_dir, args.cwm_trim_threshold, 
                  args.alpha, args.l1_ratio, args.step_size, args.convergence_tol, args.max_steps, 
                  args.batch_size, args.device)
    
    elif args.cmd == "extract-regions":
        extract_regions(args.peaks, args.fasta, args.bigwigs, args.out_path, args.region_width)

    elif args.cmd == "visualize":
        visualize(args.hits, args.out_dir)
