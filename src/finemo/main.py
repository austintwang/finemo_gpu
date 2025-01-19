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


def call_hits(regions_path, peaks_path, modisco_h5_path, chrom_order_path, motifs_include_path, motif_names_path, 
              motif_lambdas_path, out_dir, cwm_trim_threshold, lambda_default, step_size_max, step_size_min, 
              convergence_tol, max_steps, batch_size, step_adjust, device, mode, no_post_filter, compile_optimizer):
    
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
    
    motifs_df, cwms, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold, motif_type, motifs_include, 
                                                                           motif_name_map, motif_lambdas, lambda_default, True)
    num_motifs = cwms.shape[0]
    motif_width = cwms.shape[2]
    lambdas = motifs_df.get_column("lambda").to_numpy(writable=True)

    hits_df, qc_df = hitcaller.fit_contribs(cwms, contribs, sequences, trim_masks, use_hypothetical_contribs, lambdas, step_size_max, step_size_min, 
                                            convergence_tol, max_steps, batch_size, step_adjust, not no_post_filter, device, compile_optimizer)

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
    from . import evaluation        

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

        motifs_df, cwms_modisco, trim_masks, motif_names = data_io.load_modisco_motifs(modisco_h5_path, cwm_trim_threshold, "cwm", 
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
        cwm_trim_threshold = params["cwm_trim_threshold"]

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
    
    os.makedirs(out_dir, exist_ok=True)
    
    occ_path = os.path.join(out_dir, "motif_occurrences.tsv")
    data_io.write_occ_df(occ_df, occ_path)

    data_io.write_report_data(report_df, cwms, out_dir)

    evaluation.plot_hit_distributions(occ_df, motif_names, out_dir)
    evaluation.plot_completeness_distributions(hits_df, motif_names, out_dir)

    coooc_path = os.path.join(out_dir, "motif_cooocurrence.png")
    evaluation.plot_peak_motif_indicator_heatmap(coooc, motif_names, coooc_path)

    plot_dir = os.path.join(out_dir, "CWMs")
    evaluation.plot_cwms(cwms, trim_bounds, plot_dir)

    if seqlets_df is not None:
        seqlets_df = seqlets_df.collect()
        seqlets_path = os.path.join(out_dir, "seqlets.tsv")
        data_io.write_modisco_seqlets(seqlets_df, seqlets_path)

        plot_path = os.path.join(out_dir, "hit_vs_seqlet_counts.png")
        evaluation.plot_hit_vs_seqlet_counts(report_data, plot_path)

    report_path = os.path.join(out_dir, "report.html")
    evaluation.write_report(report_df, motif_names, report_path, compute_recall, seqlets_df is not None)


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
        help="The threshold to determine motif start and end positions within the full CWMs.")
    
    call_hits_parser.add_argument("-l", "--global-lambda", type=float, default=0.7,
        help="The L1 regularization weight determining the sparsity of hits.")
    call_hits_parser.add_argument("-L", "--motif-lambdas", type=str, default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and motif-specific lambdas in the second column. Omitted motifs default to the `--global-lambda` value.")
    call_hits_parser.add_argument("-a", "--alpha", type=float, default=None,
        help="DEPRECATED: Please use the `--lambda` argument instead.")
    call_hits_parser.add_argument("-A", "--motif-alphas", type=str, default=None,
        help="DEPRECATED: Please use the `--motif-lambdas` argument instead.")
    
    call_hits_parser.add_argument("-f", "--no-post-filter", action='store_true',
        help="Do not perform post-hit-calling filtering. By default, hits are filtered based on a minimum correlation of `alpha` with the input contributions.")
    call_hits_parser.add_argument("-s", "--step-size-max", type=float, default=3.,
        help="The maximum optimizer step size.")
    call_hits_parser.add_argument("-i", "--step-size-min", type=float, default=0.08,
        help="The minimum optimizer step size.")
    call_hits_parser.add_argument("-j", "--step-adjust", type=float, default=0.7,
        help="The optimizer step size adjustment factor. If the optimizer diverges, the step size is multiplicatively adjusted by this factor")
    call_hits_parser.add_argument("-c", "--convergence-tol", type=float, default=0.0005,
        help="The tolerance for determining convergence. The optimizer exits when the duality gap is less than the tolerance.")
    call_hits_parser.add_argument("-S", "--max-steps", type=int, default=10000,
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
    report_parser.add_argument("-t", "--cwm-trim-threshold", type=float, default=None,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.")
    report_parser.add_argument("-n", "--no-recall", action='store_true',
        help="Do not compute motif recall metrics.")
    report_parser.add_argument("-s", "--no-seqlets", action='store_true',
        help="DEPRECATED: Please omit the `--modisco-h5` argument instead.")
    

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
    
    elif args.cmd == "call-hits":
        if args.alpha is not None:
            warnings.warn("The `--alpha` flag is deprecated and will be removed in a future version. Please use the `--global-lambda` flag instead.")
            args.global_lambda = args.alpha
        if args.motif_alphas is not None:
            warnings.warn("The `--motif-alphas` flag is deprecated and will be removed in a future version. Please use the `--motif-lambdas` flag instead.")
            args.motif_lambdas = args.motif_alphas

        call_hits(args.regions, args.peaks, args.modisco_h5, args.chrom_order, args.motifs_include, args.motif_names, 
                  args.motif_lambdas, args.out_dir, args.cwm_trim_threshold, args.global_lambda, args.step_size_max, 
                  args.step_size_min, args.convergence_tol, args.max_steps, args.batch_size, args.step_adjust, 
                  args.device, args.mode, args.no_post_filter, args.compile)

    elif args.cmd == "report":
        if args.no_recall and not args.no_seqlets:
            raise ValueError("The `--no-seqlets` flag must be set in conjunction with `--no-recall`.")
        
        report(args.regions, args.hits, args.modisco_h5, args.peaks, args.motifs_include, 
               args.motif_names, args.out_dir, args.modisco_region_width, args.cwm_trim_threshold, 
               not args.no_recall, not args.no_seqlets)
