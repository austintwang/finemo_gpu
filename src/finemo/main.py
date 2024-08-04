from . import data_io

import os
import argparse
import warnings


def extract_regions_bw(peaks_path, chrom_order_path, fa_path, bw_paths, out_path, region_width):
    half_width = region_width // 2

    peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    sequences, contribs = data_io.load_regions_from_bw(peaks_df, fa_path, bw_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_chrombpnet_h5(peaks_path, chrom_order_path, h5_paths, out_path, region_width):
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_chrombpnet_h5(h5_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_bpnet_h5(peaks_path, chrom_order_path, h5_paths, out_path, region_width):
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_bpnet_h5(h5_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_modisco_fmt(peaks_path, chrom_order_path, shaps_paths, ohe_path, out_path, region_width):
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_modisco_fmt(shaps_paths, ohe_path, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def tune_alphas(regions_path, modisco_h5_path, motifs_include_path, p_vals, out_dir, cwm_trim_threshold, batch_size, device, mode):
    from . import hitcaller

    sequences, contribs = data_io.load_regions_npz(regions_path)
    
    if mode == "pp":
        motif_type = "cwm"
        use_hypothetical_contribs = False
    elif mode == "ph":
        motif_type = "cwm"
        use_hypothetical_contribs = True
    elif mode == "hp":
        motif_type = "hcwm"
        use_hypothetical_contribs = False
    elif mode == "hh":
        motif_type = "hcwm"
        use_hypothetical_contribs = True

    if motifs_include_path is not None:
        motifs_include = data_io.load_txt(motifs_include_path)
    else:
        motifs_include = None

    motifs_df, cwms, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold, motif_type, 
                                                                           motifs_include, None, None, None, False)

    alphas = hitcaller.calibrate_alphas(cwms, contribs, sequences, trim_masks, use_hypothetical_contribs, p_vals, batch_size, device)

    data_io.write_alphas(alphas, p_vals, motif_names, out_dir)


def tune_alphas(regions_path, modisco_h5_path, motifs_include_path, p_vals, out_dir, cwm_trim_threshold, batch_size, device, mode):
    from . import hitcaller

    sequences, contribs = data_io.load_regions_npz(regions_path)
    
    if mode == "pp":
        motif_type = "cwm"
        use_hypothetical_contribs = False
    elif mode == "ph":
        motif_type = "cwm"
        use_hypothetical_contribs = True
    elif mode == "hp":
        motif_type = "hcwm"
        use_hypothetical_contribs = False
    elif mode == "hh":
        motif_type = "hcwm"
        use_hypothetical_contribs = True

    if motifs_include_path is not None:
        motifs_include = data_io.load_txt(motifs_include_path)
    else:
        motifs_include = None

    motifs_df, cwms, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold, motif_type, 
                                                                           motifs_include, None, None, None, False)

    alphas = hitcaller.calibrate_alphas(cwms, contribs, sequences, trim_masks, use_hypothetical_contribs, p_vals, batch_size, device)

    data_io.write_alphas(alphas, p_vals, motif_names, out_dir)


def call_hits(regions_path, peaks_path, modisco_h5_path, chrom_order_path, motifs_include_path, motif_names_path, 
              motif_lambdas_path, out_dir, cwm_trim_coords_path, cwm_trim_thresholds_path, cwm_trim_threshold_default, 
              lambda_default, step_size_max, step_size_min, sqrt_transform, convergence_tol, max_steps, batch_size, 
              step_adjust, device, mode, no_post_filter, compile_optimizer):
    
    params = locals()
    from . import hitcaller

    if device is not None:
        warnings.warn("The `--device` flag is deprecated and will be removed in a future version. Please use the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU device.")
    
    sequences, contribs, peaks_df, has_peaks = data_io.load_regions_npz(regions_path)

    region_width = sequences.shape[2]
    if region_width % 2 != 0:
        raise ValueError(f"Region width of {region_width} is not divisible by 2.")
    
    half_width = region_width // 2
    num_regions = contribs.shape[0]

    if peaks_path is not None:
        warnings.warn("Providing a peaks file to `call-hits` is deprecated, and this option will be removed in a future version. Peaks should instead be provided in the preprocessing step to be included in `regions.npz`.")
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
        has_peaks = True

    if not has_peaks:
        warnings.warn("No peak region data provided. Output hits will lack absolute genomic coordinates.")

    if mode == "pp":
        motif_type = "cwm"
        use_hypothetical_contribs = False
    elif mode == "ph":
        motif_type = "cwm"
        use_hypothetical_contribs = True
    elif mode == "hp":
        motif_type = "hcwm"
        use_hypothetical_contribs = False
    elif mode == "hh":
        motif_type = "hcwm"
        use_hypothetical_contribs = True

    if motifs_include_path is not None:
        motifs_include = data_io.load_txt(motifs_include_path)
    else:
        motifs_include = None

    if motif_names_path is not None:
        motif_name_map = data_io.load_mapping(motif_names_path, str)
    else:
        motif_name_map = None

    if motif_lambdas_path is not None:
        motif_lambdas = data_io.load_mapping(motif_lambdas_path, float)
    else:
        motif_lambdas = None

    if cwm_trim_coords_path is not None:
        trim_coords = data_io.load_mapping_tuple(cwm_trim_coords_path, int)
    else:
        trim_coords = None

    if cwm_trim_thresholds_path is not None:
        trim_thresholds = data_io.load_mapping(cwm_trim_thresholds_path, float)
    else:
        trim_thresholds = None
    
    motifs_df, cwms, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, trim_coords, trim_thresholds, cwm_trim_threshold_default, 
                                                                           motif_type, motifs_include, motif_name_map, motif_lambdas, lambda_default, True)
    num_motifs = cwms.shape[0]
    motif_width = cwms.shape[2]
    lambdas = motifs_df.get_column("lambda").to_numpy(writable=True)

    hits_df, qc_df = hitcaller.fit_contribs(cwms, contribs, sequences, trim_masks, use_hypothetical_contribs, lambdas, step_size_max, step_size_min, 
                                            sqrt_transform, convergence_tol, max_steps, batch_size, step_adjust, not no_post_filter, device, compile_optimizer)

    os.makedirs(out_dir, exist_ok=True)
    out_path_qc = os.path.join(out_dir, "peaks_qc.tsv")
    data_io.write_hits(hits_df, peaks_df, motifs_df, qc_df, out_dir, motif_width)
    data_io.write_qc(qc_df, peaks_df, out_path_qc)

    out_path_motif_df = os.path.join(out_dir, "motif_data.tsv")
    data_io.write_motifs_df(motifs_df, out_path_motif_df)

    out_path_motif_cwms = os.path.join(out_dir, "motif_cwms.npy")
    data_io.write_motif_cwms(cwms, out_path_motif_cwms)

    params |= {
        "region_width": region_width,
        "num_regions": num_regions,
        "untrimmed_motif_width": motif_width,
        "num_motifs": num_motifs,
    }

    out_path_params = os.path.join(out_dir, "parameters.json")
    data_io.write_params(params, out_path_params)


def report(regions_path, hits_dir, modisco_h5_path, peaks_path, motifs_include_path, motif_names_path, 
           out_dir, modisco_region_width, cwm_trim_threshold, compute_recall, use_seqlets):
    from . import evaluation, visualization     

    sequences, contribs, peaks_df, _ = data_io.load_regions_npz(regions_path)
    if len(contribs.shape) == 3:
        regions = contribs * sequences
    elif len(contribs.shape) == 2:
        regions = contribs[:,None,:] * sequences

    half_width = regions.shape[2] // 2
    modisco_half_width = modisco_region_width // 2

    if peaks_path is not None:
        warnings.warn("Providing a peaks file to `report` is deprecated, and this option will be removed in a future version. Peaks should instead be provided in the preprocessing step to be included in `regions.npz`.")
        peaks_df = data_io.load_peaks(peaks_path, None, half_width)    

    if hits_dir.endswith(".tsv"):
        warnings.warn("Passing a hits.tsv file to `finemo report` is deprecated. Please provide the directory containing the hits.tsv file instead.")

        hits_path = hits_dir
    
        hits_df = data_io.load_hits(hits_path, lazy=True)

        if motifs_include_path is not None:
            motifs_include = data_io.load_txt(motifs_include_path)
        else:
            motifs_include = None

        if motif_names_path is not None:
            motif_name_map = data_io.load_txt(motif_names_path)
        else:
            motif_name_map = None

        motifs_df, cwms_modisco, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, None, None, cwm_trim_threshold, "cwm", 
                                                                                       motifs_include, motif_name_map, None, None, True)

    else:
        hits_df_path = os.path.join(hits_dir, "hits.tsv")
        hits_df = data_io.load_hits(hits_df_path, lazy=True)

        motifs_df_path = os.path.join(hits_dir, "motif_data.tsv")
        motifs_df, motif_names = data_io.load_motifs_df(motifs_df_path)

        cwms_modisco_path = os.path.join(hits_dir, "motif_cwms.npy")
        cwms_modisco = data_io.load_motif_cwms(cwms_modisco_path)

        params_path = os.path.join(hits_dir, "parameters.json")
        params = data_io.load_params(params_path)
        cwm_trim_threshold = params["cwm_trim_threshold_default"]

    if not use_seqlets:
        warnings.warn("Usage of the `--no-seqlets` flag is deprecated and will be removed in a future version. Please omit the `--modisco-h5` argument instead.")
        seqlets_df = None
    elif modisco_h5_path is None:
        compute_recall = False
        seqlets_df = None
    else:
        seqlets_df = data_io.load_modisco_seqlets(modisco_h5_path, peaks_df, motifs_df, half_width, modisco_half_width, lazy=True)

    motif_width = cwms_modisco.shape[2]

    occ_df, coooc = evaluation.get_motif_occurences(hits_df, motif_names)

    report_data, report_df, cwms, trim_bounds = evaluation.tfmodisco_comparison(regions, hits_df, peaks_df, seqlets_df, motifs_df,
                                                                                cwms_modisco, motif_names, modisco_half_width, 
                                                                                motif_width, compute_recall)
    
    if seqlets_df is not None:
        confusion_df, confusion_mat = evaluation.seqlet_confusion(hits_df, seqlets_df, peaks_df, motif_names, motif_width)
    
    os.makedirs(out_dir, exist_ok=True)
    
    occ_path = os.path.join(out_dir, "motif_occurrences.tsv")
    data_io.write_occ_df(occ_df, occ_path)

    data_io.write_report_data(report_df, cwms, out_dir)

    visualization.plot_hit_stat_distributions(hits_df, motif_names, out_dir)
    visualization.plot_hit_peak_distributions(occ_df, motif_names, out_dir)
    visualization.plot_peak_motif_indicator_heatmap(coooc, motif_names, out_dir)

    plot_dir = os.path.join(out_dir, "CWMs")
    visualization.plot_cwms(cwms, trim_bounds, plot_dir)

    if seqlets_df is not None:
        seqlets_df = seqlets_df.collect()
        seqlets_path = os.path.join(out_dir, "seqlets.tsv")
        data_io.write_modisco_seqlets(seqlets_df, seqlets_path)

        seqlet_confusion_path = os.path.join(out_dir, "seqlet_confusion.tsv")
        data_io.write_seqlet_confusion_df(confusion_df, seqlet_confusion_path)

        visualization.plot_hit_vs_seqlet_counts(report_data, out_dir)
        visualization.plot_seqlet_confusion_heatmap(confusion_mat, motif_names, out_dir)

    report_path = os.path.join(out_dir, "report.html")
    visualization.write_report(report_df, motif_names, report_path, compute_recall, seqlets_df is not None)


def collapse_hits(hits_path, out_path, overlap_frac):
    from . import postprocessing

    hits_df = data_io.load_hits(hits_path, lazy=False)
    hits_collapsed_df = postprocessing.collapse_hits(hits_df, overlap_frac)

    data_io.write_hits_processed(hits_collapsed_df, out_path, schema=data_io.HITS_COLLAPSED_DTYPES)


def intersect_hits(hits_paths, out_path, relaxed):
    from . import postprocessing

    hits_dfs = [data_io.load_hits(hits_path, lazy=False) for hits_path in hits_paths]
    hits_df = postprocessing.intersect_hits(hits_dfs, relaxed)

    data_io.write_hits_processed(hits_df, out_path, schema=None)


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='cmd')
    

    extract_regions_bw_parser = subparsers.add_parser("extract-regions-bw", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from FASTA and bigwig files.")
    
    extract_regions_bw_parser.add_argument("-p", "--peaks", type=str, required=True,
        help="A peak regions file in ENCODE NarrowPeak format.")
    extract_regions_bw_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.")
    extract_regions_bw_parser.add_argument("-f", "--fasta", type=str, required=True,
        help="A genome FASTA file. If an .fai index file doesn't exist in the same directory, it will be created.")
    extract_regions_bw_parser.add_argument("-b", "--bigwigs", type=str, required=True, nargs='+',
        help="One or more bigwig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_bw_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_bw_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the input region centered around each peak summit.")
    

    extract_chrombpnet_regions_h5_parser = subparsers.add_parser("extract-regions-chrombpnet-h5", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from ChromBPNet contributions H5 files.")

    extract_chrombpnet_regions_h5_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.")
    extract_chrombpnet_regions_h5_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.")

    extract_chrombpnet_regions_h5_parser.add_argument("-c", "--h5s", type=str, required=True, nargs='+',
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_chrombpnet_regions_h5_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_chrombpnet_regions_h5_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the input region centered around each peak summit.")
    
    
    extract_regions_h5_parser = subparsers.add_parser("extract-regions-h5", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from ChromBPNet contributions H5 files. DEPRECATED: Use `extract-regions-chrombpnet-h5` instead.")

    extract_regions_h5_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.")
    extract_regions_h5_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.")

    extract_regions_h5_parser.add_argument("-c", "--h5s", type=str, required=True, nargs='+',
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_h5_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_h5_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the input region centered around each peak summit.")
    

    extract_bpnet_regions_h5_parser = subparsers.add_parser("extract-regions-bpnet-h5", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from BPNet contributions H5 files.")
    
    extract_bpnet_regions_h5_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.")
    extract_bpnet_regions_h5_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.")
    
    extract_bpnet_regions_h5_parser.add_argument("-c", "--h5s", type=str, required=True, nargs='+',
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_bpnet_regions_h5_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_bpnet_regions_h5_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the input region centered around each peak summit.")
    

    extract_regions_modisco_fmt_parser = subparsers.add_parser("extract-regions-modisco-fmt", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Extract sequences and contributions from tfmodisco-lite input files.")

    extract_regions_modisco_fmt_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.")
    extract_regions_modisco_fmt_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.")

    extract_regions_modisco_fmt_parser.add_argument("-s", "--sequences", type=str, required=True,
        help="A .npy or .npz file containing one-hot encoded sequences.")
    
    extract_regions_modisco_fmt_parser.add_argument("-a", "--attributions", type=str, required=True, nargs='+',
        help="One or more .npy or .npz files of hypothetical contribution scores, with paths delimited by whitespace. Scores are averaged across files.")
    
    extract_regions_modisco_fmt_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .npz file.")
    
    extract_regions_modisco_fmt_parser.add_argument("-w", "--region-width", type=int, default=1000,
        help="The width of the input region centered around each peak summit.")


    tune_alphas_parser = subparsers.add_parser("tune-alphas", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Calibrate motif-specific L1 regularization weights using contribution scores on background regions.")
    
    tune_alphas_parser.add_argument("-M", "--mode", type=str, default="pp", choices={"pp", "ph", "hp", "hh"},
        help="The type of attributions to use for CWM's and input contribution scores, respectively. 'h' for hypothetical and 'p' for projected.")
    
    tune_alphas_parser.add_argument("-r", "--regions", type=str, required=True,
        help="A .npz file of background sequences and contributions.")
    tune_alphas_parser.add_argument("-m", "--modisco-h5", type=str, required=True,
        help="A tfmodisco-lite output H5 file of motif patterns.")
    
    tune_alphas_parser.add_argument("-I", "--motifs-include", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column to include. If omitted, all motifs in the modisco H5 file are used.")
    
    tune_alphas_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    
    tune_alphas_parser.add_argument("-t", "--cwm-trim-threshold", type=float, default=0.3,
        help="The threshold to determine motif start and end positions within the full CWMs.")
    
    tune_alphas_parser.add_argument("-b", "--batch-size", type=int, default=2000,
        help="The batch size used for computation.")
    tune_alphas_parser.add_argument("-d", "--device", type=str, default="cuda",
        help="The pytorch device name to use. Set to `cpu` to run without a GPU.")

    tune_alphas_parser.add_argument("-p", "--p-vals", type=float, default=[1e-3, 1e-4, 1e-5], nargs='+',
        help="One or more p-values to use for alpha calibration, deliminated by whitespace.")
    

    call_hits_parser = subparsers.add_parser("call-hits", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Call hits on provided sequences, contributions, and motif CWM's.")
    
    call_hits_parser.add_argument("-M", "--mode", type=str, default="pp", choices={"pp", "ph", "hp", "hh"},
        help="The type of attributions to use for CWM's and input contribution scores, respectively. 'h' for hypothetical and 'p' for projected.")

    call_hits_parser.add_argument("-r", "--regions", type=str, required=True,
        help="A .npz file of input sequences, contributions, and coordinates. Can be generated using `finemo extract-regions-*` subcommands.")
    call_hits_parser.add_argument("-m", "--modisco-h5", type=str, required=True,
        help="A tfmodisco-lite output H5 file of motif patterns.")
    
    call_hits_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.")
    call_hits_parser.add_argument("-C", "--chrom-order", type=str, default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.")
    
    call_hits_parser.add_argument("-I", "--motifs-include", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column to include in hit calling. If omitted, all motifs in the modisco H5 file are used.")
    call_hits_parser.add_argument("-N", "--motif-names", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom names in the second column. Omitted motifs default to tfmodisco names.")

    call_hits_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the output directory.")
    
    call_hits_parser.add_argument("-t", "--cwm-trim-threshold", type=float, default=0.3,
        help="The default threshold to determine motif start and end positions within the full CWMs.")
    call_hits_parser.add_argument("-T", "--cwm-trim-thresholds", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom trim thresholds in the second column. Omitted motifs default to the `--cwm-trim-threshold` value.")
    call_hits_parser.add_argument("-R", "--cwm-trim-coords", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom trim start and end coordinates in the second and third columns, respectively. Omitted motifs default to `--cwm-trim-thresholds` values.")
    
    call_hits_parser.add_argument("-l", "--global-lambda", type=float, default=0.7,
        help="The default L1 regularization weight determining the sparsity of hits.")
    call_hits_parser.add_argument("-L", "--motif-lambdas", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and motif-specific lambdas in the second column. Omitted motifs default to the `--global-lambda` value.")
    call_hits_parser.add_argument("-a", "--alpha", type=float, default=None,
        help="DEPRECATED: Please use the `--lambda` argument instead.")
    call_hits_parser.add_argument("-A", "--motif-alphas", type=str, default=None,
        help="DEPRECATED: Please use the `--motif-lambdas` argument instead.")
    
    call_hits_parser.add_argument("-f", "--no-post-filter", action='store_true',
        help="Do not perform post-hit-calling filtering. By default, hits are filtered based on a minimum cosine similarity of `lambda` with the input contributions.")
    call_hits_parser.add_argument("-q", "--sqrt-transform", action='store_true',
        help="Apply a signed square root transform to the input contributions and CWMs before hit calling.")
    call_hits_parser.add_argument("-s", "--step-size-max", type=float, default=3.,
        help="The maximum optimizer step size.")
    call_hits_parser.add_argument("-i", "--step-size-min", type=float, default=0.08,
        help="The minimum optimizer step size.")
    call_hits_parser.add_argument("-j", "--step-adjust", type=float, default=0.7,
        help="The optimizer step size adjustment factor. If the optimizer diverges, the step size is multiplicatively adjusted by this factor")
    call_hits_parser.add_argument("-c", "--convergence-tol", type=float, default=0.0005,
        help="The tolerance for determining convergence. The optimizer exits when the duality gap is less than the tolerance.")
    call_hits_parser.add_argument("-S", "--max-steps", type=int, default=1000,
        help="The maximum number of optimization steps.")
    call_hits_parser.add_argument("-b", "--batch-size", type=int, default=2000,
        help="The batch size used for optimization.")
    call_hits_parser.add_argument("-d", "--device", type=str, default=None,
        help="DEPRECATED: Please use the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU device.")
    call_hits_parser.add_argument("-J", "--compile", action='store_true',
        help="JIT-compile the optimizer for faster performance. This may not be supported on older GPUs.")
    

    report_parser = subparsers.add_parser("report", formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        help="Generate statistics and visualizations from hits and tfmodisco-lite motif data.")
    
    report_parser.add_argument("-r", "--regions", type=str, required=True,
        help="A .npz file containing input sequences, contributions, and coordinates. Must be the same as that used for `finemo call-hits`.")
    report_parser.add_argument("-H", "--hits", type=str, required=True,
        help="The output directory generated by the `finemo call-hits` command on the regions specified in `--regions`.")
    report_parser.add_argument("-p", "--peaks", type=str, default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.")
    report_parser.add_argument("-m", "--modisco-h5", type=str, default=None,
        help="The tfmodisco-lite output H5 file of motif patterns. Must be the same as that used for hit calling unless `--no-recall` is set. If omitted, seqlet-derived metrics will not be computed.")
    report_parser.add_argument("-I", "--motifs-include", type=str, default=None,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.")
    report_parser.add_argument("-N", "--motif-names", type=str, default=None,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.")

    report_parser.add_argument("-o", "--out-dir", type=str, required=True,
        help="The path to the report output directory.")
    
    report_parser.add_argument("-W", "--modisco-region-width", type=int, default=400,
        help="The width of the region around each peak summit used by tfmodisco-lite.")
    report_parser.add_argument("-t", "--cwm-trim-threshold", type=float, default=0.3,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.")
    report_parser.add_argument("-n", "--no-recall", action='store_true',
        help="Do not compute motif recall metrics.")
    report_parser.add_argument("-s", "--no-seqlets", action='store_true',
        help="DEPRECATED: Please omit the `--modisco-h5` argument instead.")


    collapse_hits_parser = subparsers.add_parser("collapse-hits", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Identify best hit by motif similarity among sets of overlapping hits.")
    
    collapse_hits_parser.add_argument("-i", "--hits", type=str, required=True,
        help="The `hits.tsv` or `hits_unique.tsv` file from `call-hits`.")
    collapse_hits_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .tsv file with an additional \"is_primary\" column.")
    collapse_hits_parser.add_argument("-O", "--overlap-frac", type=float, default=0.2,
        help="The threshold for determining overlapping hits. For two hits with lengths x and y, the minimum overlap is defined as `overlap_frac * (x + y) / 2`. The default value of 0.2 means that two hits must overlap by at least 20% of their average lengths to be considered overlapping.")
    

    intersect_hits_parser = subparsers.add_parser("intersect-hits", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Intersect hits across multiple runs.")
    
    intersect_hits_parser.add_argument("-i", "--hits", type=str, required=True, nargs='+',
        help="One or more hits.tsv or hits_unique.tsv files, with paths delimited by whitespace.")
    intersect_hits_parser.add_argument("-o", "--out-path", type=str, required=True,
        help="The path to the output .tsv file. Duplicate columns are suffixed with the positional index of the input file.")
    intersect_hits_parser.add_argument("-r", "--relaxed", action='store_true',
        help="Use relaxed intersection criteria, using only motif names and untrimmed coordinates. By default, the intersection assumes consistent region definitions and motif trimming. This option is not recommended if genomic coordinates are unavailable.")


    args = parser.parse_args()
    
    if args.cmd == "extract-regions-bw":
        extract_regions_bw(args.peaks, args.chrom_order, args.fasta, args.bigwigs, args.out_path, args.region_width)

    elif args.cmd == "extract-regions-chrombpnet-h5":
        extract_regions_chrombpnet_h5(args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width)

    elif args.cmd == "extract-regions-h5":
        print("WARNING: The `extract-regions-h5` command is deprecated. Use `extract-regions-chrombpnet-h5` instead.")
        extract_regions_chrombpnet_h5(args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width)

    elif args.cmd == "extract-regions-bpnet-h5":
        extract_regions_bpnet_h5(args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width)

    elif args.cmd == "extract-regions-modisco-fmt":
        extract_regions_modisco_fmt(args.peaks, args.chrom_order, args.attributions, args.sequences, args.out_path, args.region_width)

    elif args.cmd == "tune-alphas":
        tune_alphas(args.regions, args.modisco_h5, args.motifs_include, args.p_vals, args.out_dir, args.cwm_trim_threshold, 
                    args.batch_size, args.device, args.mode)
    
    elif args.cmd == "call-hits":
        if args.alpha is not None:
            warnings.warn("The `--alpha` flag is deprecated and will be removed in a future version. Please use the `--global-lambda` flag instead.")
            args.global_lambda = args.alpha
        if args.motif_alphas is not None:
            warnings.warn("The `--motif-alphas` flag is deprecated and will be removed in a future version. Please use the `--motif-lambdas` flag instead.")
            args.motif_lambdas = args.motif_alphas

        call_hits(args.regions, args.peaks, args.modisco_h5, args.chrom_order, args.motifs_include, args.motif_names, 
                  args.motif_lambdas, args.out_dir, args.cwm_trim_coords, args.cwm_trim_thresholds, args.cwm_trim_threshold, 
                  args.global_lambda, args.step_size_max, args.step_size_min, args.sqrt_transform, args.convergence_tol, 
                  args.max_steps, args.batch_size, args.step_adjust, args.device, args.mode, args.no_post_filter, args.compile)

    elif args.cmd == "report":
        if args.no_recall and not args.no_seqlets:
            raise ValueError("The `--no-seqlets` flag must be set in conjunction with `--no-recall`.")
        
        report(args.regions, args.hits, args.modisco_h5, args.peaks, args.motifs_include, 
               args.motif_names, args.out_dir, args.modisco_region_width, args.cwm_trim_threshold, 
               not args.no_recall, not args.no_seqlets)

    elif args.cmd == "collapse-hits":
        collapse_hits(args.hits, args.out_path, args.overlap_frac)

    elif args.cmd == "intersect-hits":
        intersect_hits(args.hits, args.out_path, args.relaxed)

