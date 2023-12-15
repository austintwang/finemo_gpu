# finemo_gpu

A GPU-accelerated hit caller for retrieving TFMoDISCo motif occurences from machine-learning-model-generated contribution scores.

## Installation

> **Note**
> This tool is under development and will be published to PyPI upon maturity. In the meantime, we recommend installing from source.

### Installing from source

#### Clone github repository

```sh
git clone https://github.com/austintwang/finemo_gpu.git
cd finemo_gpu
```

#### Create a conda environment with dependencies

This is optional but recommended

```sh
conda env create -f environment.yml -n $ENV_NAME
conda activate $ENV_NAME
```

#### Install python package

```sh
pip install --editable .
```

#### Update an installation

Simply pull from the Github repository

```sh
git pull
```

## Data Inputs

Required:

- Contribution scores for a set of peak sequences, either in bigWig format or [BPNet/ChromBPNet H5](https://github.com/kundajelab/chrombpnet/wiki/Generate-contribution-score-bigwigs#output-format) format.
- Motif CWM's in [tfmodisco-lite H5](https://github.com/jmschrei/tfmodisco-lite/tree/main#running-tfmodisco-lite) format.

Recommended:

- Peak region coordinates in [ENCODE NarrowPeak](https://genome.ucsc.edu/FAQ/FAQformat.html#format12) format.

## Usage

This tool contains the `finemo` command line utility.

### Preprocessing

These commands convert input contributions and sequences to a compressed `.npz` file suitable for rapid loading.

This output `.npz` file consists of two arrays:

- `sequences`: An array of one-hot-encoded sequences of type `np.int8` and size `(n, 4, w)`, where `n` is the number of regions, where `w` is the width of each region, and with bases ordered `ACGT`.
- `contribs`: An array of contribution scores of type `np.float16` and size `(n, 4, w)` OR `(n, w)`, with the former containing hypothetical scores and the latter containing only projected scores.

#### `finemo extract-regions-bw`

Extract sequences and contributions from FASTA and bigWig files.

> **Note** BigWig contribution files contain only projected scores, not hypothetical scores. As a result, the output regions file supports only analyses using projected contributions.

```console
usage: finemo extract-regions-bw [-h] -p PEAKS -f FASTA -b BIGWIGS [BIGWIGS ...] -o OUT_PATH [-w REGION_WIDTH]

options:
  -h, --help            show help message and exit
  -p PEAKS, --peaks PEAKS
                        A peak regions file in ENCODE NarrowPeak format. (*Required*)
  -f FASTA, --fasta FASTA
                        A genome FASTA file. An .fai index file will be built in the same directory as the fasta file if one does not already exist. (*Required*)
  -b BIGWIGS [BIGWIGS ...], --bigwigs BIGWIGS [BIGWIGS ...]
                        One or more bigwig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files. (*Required*)
  -o OUT_PATH, --out-path OUT_PATH
                        The path to the output .npz file. (*Required*)
  -w REGION_WIDTH, --region-width REGION_WIDTH
                        The width of the region extracted around each peak summit. (default: 1000)
```

#### `finemo extract-regions-h5`

Extract sequences and contributions from BPNet/ChromBPNet H5 files.

```console
usage: finemo extract-regions-h5 [-h] -c H5S [H5S ...] -o OUT_PATH [-w REGION_WIDTH]

options:
  -h, --help            show help message and exit
  -c H5S [H5S ...], --h5s H5S [H5S ...]
                        One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files. (*Required*)
  -o OUT_PATH, --out-path OUT_PATH
                        The path to the output .npz file. (*Required*)
  -w REGION_WIDTH, --region-width REGION_WIDTH
                        The width of the region extracted around each peak summit. (default: 1000)
```

#### `finemo extract-regions-modisco-fmt`

Extract sequences and contributions from tfmodisco-lite input `.npy`/`.npz` files.

```console
usage: finemo extract-regions-modisco-fmt [-h] -s SEQUENCES -a ATTRIBUTIONS [ATTRIBUTIONS ...] -o OUT_PATH [-w REGION_WIDTH]

options:
  -h, --help            show this help message and exit
  -s SEQUENCES, --sequences SEQUENCES
                        A .npy or .npz file containing one-hot encoded sequences. (*Required*)
  -a ATTRIBUTIONS [ATTRIBUTIONS ...], --attributions ATTRIBUTIONS [ATTRIBUTIONS ...]
                        One or more .npy or .npz files of hypothetical contribution scores, with paths delimited by whitespace. Scores are averaged across files. (*Required*)
  -o OUT_PATH, --out-path OUT_PATH
                        The path to the output .npz file. (*Required*)
  -w REGION_WIDTH, --region-width REGION_WIDTH
                        The width of the region extracted around each peak summit. (default: 1000)
```

### Hit Calling

#### `finemo call-hits`

Call hits in input regions using TFMoDISCo CWM's.

```console
usage: finemo call-hits [-h] [-M MODE] -r REGIONS -m MODISCO_H5 [-p PEAKS] -o OUT_DIR [-t CWM_TRIM_THRESHOLD] [-a ALPHA] [-l L1_RATIO] [-s STEP_SIZE] [-A STEP_ADJUST] [-c CONVERGENCE_TOL] [-S MAX_STEPS] [-b BUFFER_SIZE] [-d DEVICE]

options:
  -h, --help            show help message and exit
  -M {hh,pp,ph,hp}, --mode {hh,pp,ph,hp}
                        Type of attributions to use for CWM's and input contribution scores, respectively. 'h' for hypothetical and 'p' for projected. (default: pp)
  -r REGIONS, --regions REGIONS
                        A .npz file of input sequences and contributions. Can be generated using `finemo extract-regions-*` subcommands. (*Required*)
  -m MODISCO_H5, --modisco-h5 MODISCO_H5
                        A tfmodisco-lite output H5 file of motif patterns. (*Required*)
  -p PEAKS, --peaks PEAKS
                        A peak regions file in ENCODE NarrowPeak format. These should exactly match the regions in `--regions`. If omitted, called hits will not have absolute genomic coordinates. (Optional)
  -C CHROM_ORDER, --chrom-order CHROM_ORDER
                        A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. For missing chromosomes, order is set by order of appearance in -p/--peaks. (Optional)
  -o OUT_DIR, --out-dir OUT_DIR
                        The path to the output directory. (*Required*)
  -t CWM_TRIM_THRESHOLD, --cwm-trim-threshold CWM_TRIM_THRESHOLD
                        Trim treshold for determining motif start and end positions within the full input motif CWM's. (default: 0.3)
  -a ALPHA, --alpha ALPHA
                        Total regularization weight. (default: 0.6)
  -l L1_RATIO, --l1-ratio L1_RATIO
                        Elastic net mixing parameter. This specifies the fraction of `alpha` used for L1 regularization. (default: 1.0)
  -s STEP_SIZE, --step-size STEP_SIZE
                        Maximum optimizer step size. (default: 3.0)
  -A STEP_ADJUST, --step-adjust STEP_ADJUST
                        Optimizer step size adjustment factor. If the optimizer diverges, the step size is multiplicatively adjusted by this factor (default: 0.7)
  -c CONVERGENCE_TOL, --convergence-tol CONVERGENCE_TOL
                        Tolerance for assessing convergence. The optimizer exits when the dual gap is less than the tolerance. (default: 0.0005)
  -S MAX_STEPS, --max-steps MAX_STEPS
                        Maximum optimizer steps. (default: 10000)
  -b BUFFER_SIZE, --buffer-size BUFFER_SIZE
                        Size of buffer used for optimization. (default: 2000)
  -d DEVICE, --device DEVICE
                        Pytorch device name. Set to `cpu` to run without a GPU. (default: cuda)
```

#### Outputs

`hits.tsv`: The full set of coordinate-sorted hits with the following fields:

- `chr`: Chromosome name. `NA` if peak coordinates (`-p/--peaks`) are not provided.
- `start`: Hit start coordinate from trimmed CWM, zero-indexed. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `end`: Hit end coordinate from trimmed CWM, zero-indexed, non-inclusive. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `start_untrimmed`: Hit start coordinate from trimmed CWM, zero-indexed. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `end_untrimmed`: Hit end coordinate from trimmed CWM, zero-indexed, non-inclusive. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `motif_name`: Hit motif name, taken from the provided TFMoDisCo H5 file.
- `hit_coefficient`: The regression coefficient for the hit.
- `hit_correlation`: The correlation between the untrimmed CWM and the hit region contribution score.
- `hit_importance`: The root-mean-square contribution score within the hit region.
- `strand`: Orientation of the hit (`+`/`-`).
- `peak_name`: The name of the peak region corresponding to the hit, taken from the `name` field of the input peak data. `NA` if `-p/--peaks` is not provided.
- `peak_id`: The numerical index of the peak region corresponding to the hit.

`hits_unique.tsv`: A set of deduplicated hits in the same format as `hits.tsv`. If peak regions overlap, multiple instances of a hit may appear in `hits.tsv`, each pointing to a different peak. Here, a single instance is arbitrarily chosen for each duplicated peak. This file is created only if `-p/--peaks` is provided.

`hits.bed`: A coordinate-sorted BED file of unique hits. This file is created only if `-p/--peaks` is provided and contains the following fields:

- `chr`: Chromosome name.
- `start`: Hit start coordinate from trimmed CWM, zero-indexed.
- `end`: Hit end coordinate from trimmed CWM, zero-indexed, non-inclusive.
- `motif_name`: Hit motif name, taken from the provided TFMoDISCo H5 file.
- `score`: The `hit_correlation` score multiplied by 1000 and cast to an integer.
- `strand`: Orientation of the hit (`+`/`-`).

`peaks_qc.tsv`: Per-peak statistics with the following fields:

- `peak_id`: The numerical index of the peak region.
- `log_likelihood`: Final regression log likelihood (proportional to MSE)
- `dual_gap`: Final duality gap
- `num_steps`: Number of optimization steps
- `step_size`: Optimization step size
- `global_scale`: Scaling factor used for numerical stability
- `chr`: Chromosome name. Omitted if `-p/--peaks` not provided.
- `peak_region_start`: Peak region start coordinate, zero-indexed. Omitted if `-p/--peaks` not provided.
- `peak_name`: The name of the peak region corresponding to the hit, taken from the `name` field of the input peak data. Omitted if `-p/--peaks` not provided.

`params.json`: Parameters provided for hit calling.

#### Additional notes

- `-a/--alpha` is the primary hyperparameter to tune, with higher values giving fewer but more accurate hits. Typically, values around 0.4-0.7 serve as a good starting point. This hyperparameter can be interpreted as the maximum expected cross-correlation between a CWM and a background/uninformative track.
- `-b/--buffer-size` should be set to the highest value that fits in GPU memory. **If you encounter GPU out-of-memory errors, try reducing this value.**
- `-s/--step-size` does not substantively affect results, but should be tuned to maximize performance.
- Legacy TFMoDISCo H5's can be converted to TFMoDISCo-lite format using `modisco convert` in the [`modisco-lite`](https://github.com/jmschrei/tfmodisco-lite/tree/main) package.
- Hit calling uses only untrimmed CWM's. Trimmed CWM's are used only for calculating hit start and end offsets.

### Output reporting

#### `finemo report`

Generate an HTML report (`report.html`) with TF-MoDISCo seqlet recall and hit distribution visualizations.
The input regions must have genomic coordinates and be identical to those used for TF-MoDISCo motif construction.

```console
usage: finemo report [-h] -r REGIONS -H HITS -p PEAKS -m MODISCO_H5 -o OUT_DIR [-W MODISCO_REGION_WIDTH]

options:
  -h, --help            show this help message and exit
  -r REGIONS, --regions REGIONS
                        A .npz file of input sequences and contributions. Must be identical to the data used for hit calling and tfmodisco motif calling. (*Required*)
  -H HITS, --hits HITS  The `hits.tsv` output file from `finemo call-hits`. (*Required*)
  -p PEAKS, --peaks PEAKS
                        A sorted peak regions file in ENCODE NarrowPeak format. These should exactly match the regions in `--regions`. (*Required*)
  -m MODISCO_H5, --modisco-h5 MODISCO_H5
                        A tfmodisco-lite output H5 file of motif patterns. (*Required*)
  -o OUT_DIR, --out-dir OUT_DIR
                        The path to the output directory. (*Required*)
  -W MODISCO_REGION_WIDTH, --modisco-region-width MODISCO_REGION_WIDTH
                        The width of the region extracted around each peak summit used by tfmodisco-lite. (default: 400)
```
