from . import data_io

import os
import argparse

import polars as pl


def extract_regions_bw(peaks_path, fa_path, bw_paths, out_path, region_width):
    half_width = region_width // 2

    peaks_df = data_io.load_peaks(peaks_path, half_width)
    sequences, contribs = data_io.load_regions_from_bw(peaks_df, fa_path, bw_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path)


def extract_regions_h5(peaks_path, h5_paths, out_path, region_width):
    half_width = region_width // 2

    peaks_df = data_io.load_peaks(peaks_path, half_width)
    sequences, contribs = data_io.load_regions_from_h5(peaks_df, h5_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path)


def call_hits(regions_path, peaks_path, modisco_h5_path, out_dir, cwm_trim_threshold, alpha, l1_ratio, 
              step_size, convergence_tol, max_steps, buffer_size, step_adjust, device, use_hypothetical):
    
    params = locals()
    from . import hitcaller
    
    sequences, contribs = data_io.load_regions_npz(regions_path)

    # if (sequences.shape[0] != contribs.shape[0]) or (sequences.shape[2] != contribs.shape[1]):
    #     raise ValueError(f"Input sequences of shape {sequences.shape} is not compatible with " 
    #                      f"input contributions of shape {contribs.shape}.")

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

    motifs_df, cwms = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold, use_hypothetical)
    num_motifs = cwms.shape[1]
    motif_width = cwms.shape[2]

    hits, qc = hitcaller.fit_contribs(cwms, contribs, sequences, use_hypothetical, alpha, l1_ratio, step_size, 
                                      convergence_tol, max_steps, buffer_size, step_adjust, device)
    hits_df = pl.DataFrame(hits)
    qc_df = pl.DataFrame(qc).with_row_count(name="peak_id")

    os.makedirs(out_dir, exist_ok=True)
    out_path_tsv = os.path.join(out_dir, "hits.tsv")
    out_path_bed = os.path.join(out_dir, "hits.bed")
    out_path_qc = os.path.join(out_dir, "peaks_qc.tsv")
    data_io.write_hits(hits_df, peaks_df, motifs_df, qc_df, out_path_tsv, 
                       out_path_bed, half_width, motif_width)
    data_io.write_qc(qc_df, peaks_df, out_path_qc)

    params |= {
        "region_width": region_width,
        "num_regions": num_regions,
        "untrimmed_motif_width": motif_width,
        "num_motifs": num_motifs,
    }
    # params = {
    #     "inputs": {
    #         "regions_path": regions_path,
    #         "peaks_path": peaks_path,
    #         "modisco_h5_path": modisco_h5_path,
    #     },
    #     "outputs": {
    #         "out_dir": out_dir,
    #     },
    #     "provided_params": {
    #         "cwm_trim_threshold": cwm_trim_threshold,
    #         "alpha": alpha,
    #         "l1_ratio": l1_ratio,
    #         "step_size": step_size,
    #         "convergence_tol": convergence_tol,
    #         "max_steps": max_steps,
    #         "batch_size": batch_size,
    #         "device": device,
    #     },
    #     "inferred_params": {
    #         "region_width": region_width,
    #         "num_regions": num_regions,
    #         "untrimmed_motif_width": motif_width,
    #         "num_motifs": num_motifs,
    #     }
    # }
    out_path_params = os.path.join(out_dir, "parameters.json")
    data_io.write_params(params, out_path_params)


def visualize(hits_path, out_dir):
    from . import evaluation

    hits_df = data_io.load_hits(hits_path)
    num_peaks = hits_df.height

    occ_df, occ_mat, occ_bin, coocc, motif_names = evaluation.get_motif_occurences(hits_df)
    # peak_order = visualization.order_rows(occ_mat)
    motif_order = evaluation.order_rows(occ_mat.T)
    coocc_nlp, coocc_lor = evaluation.cooccurrence_sigs(coocc, num_peaks)

    os.makedirs(out_dir, exist_ok=True)

    occ_path = os.path.join(out_dir, "motif_occurrences.tsv")
    data_io.write_occ_df(occ_df, occ_path)
    
    coocc_dir = os.path.join(out_dir, "motif_cooccurrence_matrices")
    data_io.write_coocc_mats(coocc, coocc_nlp, motif_names, coocc_dir)

    score_dist_dir = os.path.join(out_dir, "hit_score_distributions")
    evaluation.plot_score_distributions(hits_df, score_dist_dir)

    hits_cdf_dir = os.path.join(out_dir, "motif_peak_hit_cdfs")
    evaluation.plot_homotypic_densities(occ_mat, motif_names, hits_cdf_dir)

    frac_peaks_path = os.path.join(out_dir, "frac_peaks_with_motif.png")
    evaluation.plot_frac_peaks(occ_bin, motif_names, frac_peaks_path)

    # occ_path = os.path.join(out_dir, "peak_motif_occurrences.png")
    # visualization.plot_occurrence(occ_mat, motif_names, peak_order, motif_order, occ_path)

    coocc_counts_path = os.path.join(out_dir, "motif_cooccurrence_counts.png")
    evaluation.plot_cooccurrence_counts(coocc, motif_names, motif_order, coocc_counts_path)

    coocc_sigs_path = os.path.join(out_dir, "motif_cooccurrence_neg_log10p.png")
    evaluation.plot_cooccurrence_sigs(coocc_nlp, motif_names, motif_order, coocc_sigs_path)

    coocc_ors_path = os.path.join(out_dir, "motif_cooccurrence_odds_ratios.png")
    evaluation.plot_cooccurrence_lors(coocc_lor, motif_names, motif_order, coocc_ors_path)


def modisco_recall(hits_path, modisco_h5_path, peaks_path, out_dir, modisco_region_width, scale_hits):
    from . import evaluation

    half_width = modisco_region_width // 2
    peaks_df = data_io.load_peaks(peaks_path, half_width)
    hits_df = data_io.load_hits(hits_path, lazy=True)
    seqlets_df, seqlet_counts = data_io.load_modisco_seqlets(modisco_h5_path, peaks_df, lazy=True)

    seqlet_recalls = evaluation.seqlet_recall(hits_df, seqlets_df, seqlet_counts, scale_hits)
    
    recall_dir= os.path.join(out_dir, "modisco_recall_data")
    data_io.write_modisco_recall(seqlet_recalls, seqlet_counts, recall_dir)

    plot_dir = os.path.join(out_dir, "modisco_recall_plots")
    evaluation.plot_modisco_recall(seqlet_recalls, plot_dir)


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='cmd')

    call_hits_parser = subparsers.add_parser("call-hits", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Call hits on provided sequences, contributions, and motif CWM's.")
    
    call_hits_parser.add_argument("-y", "--hypothetical", action="store_true", default=False,
        help="Use hypothetical contribution scores and CWM's.")

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
    call_hits_parser.add_argument("-a", "--alpha", type=float, default=3.,
        help="Total regularization weight.")
    call_hits_parser.add_argument("-l", "--l1-ratio", type=float, default=0.99,
        help="Elastic net mixing parameter. This specifies the fraction of `alpha` used for L1 regularization.")
    call_hits_parser.add_argument("-s", "--step-size", type=float, default=0.02,
        help="Maximum optimizer step size.")
    call_hits_parser.add_argument("-A", "--step-adjust", type=float, default=0.7,
        help="Optimizer step size adjustment factor. If the optimizer diverges, the step size is multiplicatively adjusted by this factor")
    call_hits_parser.add_argument("-c", "--convergence-tol", type=float, default=0.001,
        help="Tolerance for assessing convergence. The optimizer exits when the dual gap is less than the tolerance.")
    call_hits_parser.add_argument("-S", "--max-steps", type=int, default=10000,
        help="Maximum optimizer steps.")
    call_hits_parser.add_argument("-b", "--buffer-size", type=int, default=2000,
        help="Size of buffer used for optimization.")
    call_hits_parser.add_argument("-d", "--device", type=str, default="cuda",
        help="Pytorch device name. Set to `cpu` to run without a GPU.")
    
    
    extract_regions_bw_parser = subparsers.add_parser("extract-regions-bw", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from FASTA and bigwig files.")
    
    extract_regions_bw_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A sorted peak regions file in ENCODE NarrowPeak format.")
    extract_regions_bw_parser.add_argument("-f", "--fasta", type=str, required=True,
        help="A genome FASTA file. An .fai index file will be built in the same directory as the fasta file if one does not already exist.")
    extract_regions_bw_parser.add_argument("-b", "--bigwigs", type=str, required=True, nargs='+',
        help="One or more bigwig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_bw_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_bw_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the region extracted around each peak summit.")
    

    extract_regions_h5_parser = subparsers.add_parser("extract-regions-h5", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from chromBPNet contributions h5 files.")
    
    extract_regions_h5_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A sorted peak regions file in ENCODE NarrowPeak format.")
    extract_regions_h5_parser.add_argument("-c", "--h5s", type=str, required=True, nargs='+',
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_h5_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_h5_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the region extracted around each peak summit.")
    

    visualize_parser = subparsers.add_parser("visualize", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from FASTA and bigwig files.")
    
    visualize_parser.add_argument("-H", "--hits", type=str, required=True,
        help="The `hits.tsv` output file from `call-hits`.")
    
    visualize_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    

    modisco_recall_parser = subparsers.add_parser("modisco-recall", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Assess recall of TFMoDisCo seqlets from called hits.")
    
    modisco_recall_parser.add_argument("-H", "--hits", type=str, required=True,
        help="The `hits.tsv` output file from `call-hits`.")
    modisco_recall_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A sorted peak regions file in ENCODE NarrowPeak format. These should exactly match the regions in `--regions`.")
    modisco_recall_parser.add_argument("-m", "--modisco-h5", type=str, required=True,
        help="A tfmodisco-lite output H5 file of motif patterns.")
    
    modisco_recall_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    
    modisco_recall_parser.add_argument("-W", "--modisco-region-width", type=int, default=400,
        help="The width of the region extracted around each peak summit used by TFMoDisCo.")
    
    call_hits_parser.add_argument("-n", "--scale-hits", action="store_true", default=False,
        help="Use scaled hit scores (normalized for overall contribution strength of region).")

    
    args = parser.parse_args()

    if args.cmd == "call-hits":
        call_hits(args.regions, args.peaks, args.modisco_h5, args.out_dir, args.cwm_trim_threshold, 
                  args.alpha, args.l1_ratio, args.step_size, args.convergence_tol, args.max_steps, 
                  args.buffer_size, args.step_adjust, args.device, args.hypothetical)
    
    elif args.cmd == "extract-regions-bw":
        extract_regions_bw(args.peaks, args.fasta, args.bigwigs, args.out_path, args.region_width)

    elif args.cmd == "extract-regions-h5":
        extract_regions_h5(args.peaks, args.h5s, args.out_path, args.region_width)

    elif args.cmd == "visualize":
        visualize(args.hits, args.out_dir)

    elif args.cmd == "modisco-recall":
        modisco_recall(args.hits, args.modisco_h5, args.peaks, args.out_dir, args.modisco_region_width, args.scale_hits)
