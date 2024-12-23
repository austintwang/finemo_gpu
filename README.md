# finemo_gpu

**FiNeMo** (**Fi**nding **Ne**ural network **Mo**tifs) is a GPU-accelerated hit caller for identifying occurrences of TFMoDISCo motifs within contribution scores generated by machine learning models.

## Installation

> **Note**
> This software is currently in development and will be available on PyPI once mature.
For now, we suggest installing it from source.

### Installing from Source

#### Clone the GitHub Repository

```sh
git clone https://github.com/austintwang/finemo_gpu.git
cd finemo_gpu
```

#### Create a Conda Environment with Dependencies

This step is optional but recommended

```sh
conda env create -f environment.yml -n $ENV_NAME
conda activate $ENV_NAME
```

#### Install the Python Package

```sh
pip install --editable .
```

#### Update an Existing Installation

To update, simply fetch the latest changes from the GitHub repository.

```sh
git pull
```

If needed, update the conda environment with the latest dependencies and reinstall the package.

```sh
conda env update -f environment.yml -n $ENV_NAME --prune
pip install --editable .
```

## Data Inputs

Required:

- Contribution scores for peak sequences in bigWig format, [ChromBPNet H5](https://github.com/kundajelab/chrombpnet/wiki/Generate-contribution-score-bigwigs#output-format) format, [BPNet H5](https://github.com/kundajelab/bpnet-refactor?tab=readme-ov-file#3-compute-importance-scores) format, or [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/tree/main#running-tfmodisco-lite) input format.
- Motif CWMs in [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/tree/main#running-tfmodisco-lite) H5 output format.

Recommended:

- Peak region coordinates in uncompressed [ENCODE NarrowPeak](https://genome.ucsc.edu/FAQ/FAQformat.html#format12) format.

## Usage

FiNeMo includes a command-line utility named `finemo`. Here, we describe basic usage for each subcommand. For all options, run `finemo <subcommand> -h`.

### Preprocessing

The following commands transform input contributions and sequences into a compressed `.npz` file for quick loading. This file contains:

- `sequences`: A one-hot-encoded sequence array (`np.int8`) with dimensions `(n, 4, w)`, where `n` is the number of regions, and `w` is the width of each region. Bases are ordered as ACGT.
- `contribs`: A contribution score array (`np.float16`) with dimensions `(n, 4, w)` for hypothetical scores or `(n, w)` for projected scores only.

If peak region coordinates are provided, the output also includes:

- `chr`: An array of chromosome names (`np.dtype('U')`) with dimensions `(n)`.
- `chr_id`: An array of numerical chromosome IDs (`np.uint32`) with dimensions `(n)`.
- `peak_region_start`: An array of peak region start coordinates (`np.int32`) with dimensions `(n)`.
- `peak_id`: An array of numerical peak region IDs (`np.uint32`) with dimensions `(n)`.
- `peak_name`: An array of peak region names (`np.dtype('U')`) with dimensions `(n)`.

Preprocessing commands do not require GPU.

#### Shared arguments

- `-o/--out-path`: The path to the output `.npz` file. Required.
- `-w/--region-width`: The width of the input region centered around each peak summit. Default is 1000.
- `-p/--peaks`: A peak regions file in ENCODE NarrowPeak format. This is required for `finemo extract-regions-bw` and optional for all other preprocessing commands.

#### `finemo extract-regions-bw`

Extract sequences and contributions from FASTA and bigWig files.

> **Note** BigWig files only provide projected contribution scores.
Thus, the output only supports analyses based solely on projected contributions.

Usage: `finemo extract-regions-bw -p <peaks> -f <fasta> -b <bigwigs> -o <out_path> [-w <region_width>]`

- `-f/--fasta`: A genome FASTA file. If an `.fai` index file doesn't exist in the same directory, it will be created.
- `-b/--bigwigs`: One or more bigWig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.

#### `finemo extract-regions-chrombpnet-h5`

Extract sequences and contributions from ChromBPNet H5 files.

Usage: `finemo extract-regions-chrombpnet-h5 -c <h5s> -o <out_path> [-p <peaks>] [-w <region_width>]`

- `-c/--h5s`: One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.

#### `finemo extract-regions-bpnet-h5`

Extract sequences and contributions from BPNet H5 files.

Usage: `finemo extract-regions-bpnet-h5 -c <h5s> -o <out_path> [-p <peaks>] [-w <region_width>]`

- `-c/--h5s`: One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.

#### `finemo extract-regions-modisco-fmt`

Extract sequences and contributions from tfmodisco-lite input `.npy`/`.npz` files.

Usage: `finemo extract-regions-modisco-fmt -s <sequences> -a <attributions> -o <out_path> [-p <peaks>] [-w <region_width>]`

- `-s/--sequences`: A `.npy` or `.npz` file containing one-hot encoded sequences.
- `-a/--attributions`: One or more `.npy` or `.npz` files of hypothetical contribution scores, with paths delimited by whitespace. Scores are averaged across files.

### Hit Calling

#### `finemo call-hits`

Identify hits in input regions using TFMoDISCo CWM's.

Usage: `finemo call-hits -r <regions> -m <modisco_h5> -o <out_dir> [-p <peaks>] [-t <cwm_trim_threshold>] [-l <global_lambda>] [-b <batch_size>] [-J]`

- `-r/--regions`: A `.npz` file of input sequences, contributions, and coordinates. Created with a `finemo extract-regions-*` command.
- `-m/--modisco-h5`: A tfmodisco-lite output H5 file of motif patterns.
- `-o/--out-dir`: The path to the output directory.
- `-t/--cwm-trim-threshold`: The threshold to determine motif start and end positions within the full CWMs. Default is 0.3.
- `-l/--global-lambda`: The L1 regularization weight determining the sparsity of hits. Default is 0.7.
- `-b/--batch-size`: The batch size used for optimization. Default is 2000.
- `-J/--compile`: Enable JIT compilation for faster execution. This option may not work on older GPUs.

#### Outputs

`hits.tsv`: The full list of coordinate-sorted hits with the following fields:

- `chr`: Chromosome name. `NA` if peak coordinates are not provided.
- `start`: Hit start coordinate from trimmed CWM, zero-indexed. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `end`: Hit end coordinate from trimmed CWM, zero-indexed, exclusive. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `start_untrimmed`: Hit start coordinate from trimmed CWM, zero-indexed. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `end_untrimmed`: Hit end coordinate from trimmed CWM, zero-indexed, exclusive. Absolute if peak coordinates are provided, otherwise relative to the input region.
- `motif_name`: The hit motif name as specified in the provided tfmodisco H5 file.
- `hit_coefficient`: The regression coefficient for the hit, normalized per peak region.
- `hit_coefficient_global`: The regression coefficient for the hit, scaled by the overall importance of the region. **This is the primary hit score.**
- `hit_correlation`: The correlation between the untrimmed CWM and the contribution score of the motif hit.
- `hit_importance`: The total absolute contribution score within the motif hit.
- `strand`: The orientation of the hit (`+` or `-`).
- `peak_name`: The name of the peak region containing the hit, taken from the `name` field of the input peak data. `NA` if peak coordinates are not provided.
- `peak_id`: The numerical index of the peak region containing the hit.

`hits_unique.tsv`: A deduplicated list of hits in the same format as `hits.tsv`. In cases where peak regions overlap, `hits.tsv` may list multiple instances of a hit, each linked to a different peak. `hits_unique.tsv` arbitrarily selects one instance per duplicated hit. This file is generated only if peak coordinates are provided.

`hits.bed`: A coordinate-sorted BED file of unique hits. It includes:

- `chr`: Chromosome name. `NA` if peak coordinates are not provided.
- `start`: Hit start coordinate from trimmed CWM, zero-indexed.  Absolute if peak coordinates are provided, otherwise relative to the input region.
- `end`: Hit end coordinate from trimmed CWM, zero-indexed, exclusive.  Absolute if peak coordinates are provided, otherwise relative to the input region.
- `motif_name`: Hit motif name, taken from the provided tfmodisco H5 file.
- `score`: The `hit_correlation` score, multiplied by 1000.
- `strand`: The orientation of the hit (`+` or `-`).

`peaks_qc.tsv`: Per-peak statistics. It includes:

- `peak_id`: The numerical index of the peak region.
- `nll`: The final regression negative log likelihood, proportional to the mean squared error (MSE).
- `dual_gap`: The final duality gap.
- `num_steps`: The number of optimization steps taken.
- `step_size`: The optimization step size.
- `global_scale`: The peak-level scaling factor, used to normalize by overall importance.
- `peak_id`: The numerical index of the peak region containing the hit.
- `chr`: The chromosome name. `NA` if peak coordinates are not provided.
- `peak_region_start`: The start coordinate of the peak region, zero-indexed. All zero if peak coordinates are not provided.
- `peak_name`: The name of the peak region, derived from the input peak data's `name` field. `NA` if peak coordinates are not provided.

`motif_data.tsv`: Per-motif metadata. It includes:

- `motif_id`: The numerical index of the motif. Note that the forward and reverse complement of a motif are treated separately.
- `motif_name`: The motif name used.
- `motif_name_orig`: The motif name as specified in the provided tfmodisco H5 file.
- `strand`: The orientation of the motif (`+` or `-`).
- `motif_start`: The start coordinate of the trimmed motif in the untrimmed motif, zero-indexed.
- `motif_end`: The end coordinate of the trimmed motif in the untrimmed motif, zero-indexed, exclusive.
- `motif_scale`: The motif scaling factor, used to normalize by motif importance.
- `lambda`: The motif-specific L1 regularization weight.

`motif_cwms.npy`: A NumPy array of motif CWMs, with dimensions `(n, 4, w)`, where `n` is the number of motifs, and `w` is the width of each motif. Bases are ordered as ACGT. The ordering of motifs corresponds to the `motif_id` field in `motif_data.tsv`.

`params.json`: The parameters used for hit calling.

#### Additional notes

- The `-l/--global-lambda` parameter controls the sensitivity of the hit-calling algorithm, with higher values resulting in fewer but more confident hits. This parameter represents the minimum correlation between a query contribution score window and a CWM to be considered a hit. The default value of 0.7 typically works well for chromatin accessibility data. ChIP-Seq data may require a lower value (e.g. 0.6).
- The `-t/--cwm-trim-threshold` parameter sets the maximum relative contribution score in trimmed-out CWM flanks. If you find that motif flanks are being trimmed too aggressively, consider lowering this value. However, a too-low value may result in closely-spaced motif instances being missed.
- Set `-b/--batch-size` to fill a significant fraction of your GPU memory. **If you encounter GPU out-of-memory errors, try lowering this value.**
- Legacy TFMoDISCo H5 files can be updated to the newer TFMoDISCo-lite format with the `modisco convert` command found in the [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/tree/main) package.
- The hit-calling algorithm is scale-invariant. That is, whether a hit is detected depends on the shapes of the motif CWM and the contribution scores, not the absolute magnitude of the scores. If you wish to prioritize hits based on the magnitude of the contribution scores, use the `hit_coefficient_global` field in the `hits.tsv` file, which captures both the absolute importance and the closeness of match.

### Output reporting

#### `finemo report`

Generate an HTML report (`report.html`) visualizing TF-MoDISCo seqlet recall and hit distributions.
If `-n/--no-recall` is not set, the regions used for hit calling must exactly match those used during the TF-MoDISCo motif discovery process.
This command does not utilize the GPU.

Usage: `finemo report -r <regions> -H <hits> -o <out_dir> [-m <modisco_h5>] [-W <modisco_region_width>] [-n]`

- `-r/--regions`: A `.npz` file containing input sequences, contributions, and coordinates. Must be the same as that used for `finemo call-hits`.
- `-H/--hits`: The output directory generated by the `finemo call-hits` command on the regions specified in `--regions`.
- `-o/--out-dir`: The path to the report output directory.
- `-m/--modisco-h5`: The tfmodisco-lite output H5 file of motif patterns. Must be the same as that used for hit calling unless `--no-recall` is set. If omitted, seqlet-derived metrics will not be computed.
- `-W/--modisco-region-width`: The width of the region around each peak summit used by tfmodisco-lite. Default is 400.
- `-n/--no-recall`: Do not compute motif recall metrics. Default is False.

#### Additional outputs

`motif_report.tsv`: Statistics on the distribution of hits per motif. The columns and values correspond to those in the HTML report's table.

`motif_occurrences.tsv`: The number of hits of each motif in each input region. Also includes the total number of hits per region.

`CWMs`: A directory containing visualizations of motif CWMs, as well as corresponding tables with numerical CWM values.

`seqlets.tsv`: tf-modisco seqlet coordinates for each motif in each region. Only generated if `-m/--modisco-h5` is provided.
