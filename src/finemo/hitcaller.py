"""Hit caller module implementing the Fi-NeMo motif instance calling algorithm.

This module provides the core functionality for identifying transcription factor
binding motif instances in neural network contribution scores using a competitive
optimization approach based on proximal gradient descent.

The main algorithm fits a sparse linear model where contribution scores are
reconstructed as a weighted combination of motif contribution weight matrices (CWMs)
at specific genomic positions. The sparsity constraint ensures that only the most
significant motif instances are called.
"""

import warnings
from typing import Tuple, Union, Optional, Dict, List
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
from torch import Tensor
import polars as pl
from jaxtyping import Float, Int, Bool

from tqdm import tqdm

# Type aliases for tensor operations
ArrayLike = Union[ndarray, torch.Tensor]


def prox_grad_step(
    coefficients: Float[Tensor, "B M P"],
    importance_scale: Float[Tensor, "B 1 P"],
    cwms: Float[Tensor, "M 4 W"],
    contribs: Float[Tensor, "B 4 L"],
    sequences: Union[Int[Tensor, "B 4 L"], int],
    lambdas: Float[Tensor, "1 M 1"],
    step_sizes: Float[Tensor, "B 1 1"],
) -> Tuple[Float[Tensor, "B M P"], Float[Tensor, " B"], Float[Tensor, " B"]]:
    """Perform a proximal gradient descent optimization step for non-negative lasso.

    This function implements a single optimization step of the Fi-NeMo algorithm,
    which uses proximal gradient descent to solve a sparse reconstruction problem.
    The goal is to represent contribution scores as a sparse linear combination
    of motif contribution weight matrices (CWMs).

    B = batch size, M = number of motifs, L = sequence length, W = motif width.
    P = L - W + 1 (the number of positions with coefficients).

    Parameters
    ----------
    coefficients : Float[Tensor, "B M P"]
        Current coefficient matrix representing motif instance strengths.
    importance_scale : Float[Tensor, "B 1 P"]
        Scaling factors for importance-weighted reconstruction.
    cwms : Float[Tensor, "M 4 W"]
        Motif contribution weight matrices for all motifs.
        4 represents the DNA bases (A, C, G, T).
    contribs : Float[Tensor, "B 4 L"]
        Target contribution scores to reconstruct.
    sequences : Float[Tensor, "B 4 L"] | int
        One-hot encoded DNA sequences. Can be a scalar (1) for hypothetical mode.
    lambdas : Float[Tensor, "1 M 1"]
        L1 regularization weights for each motif.
    step_sizes : Float[Tensor, "B 1 1"]
        Optimization step sizes for each batch element.

    Returns
    -------
    c_next : Float[Tensor, "B M P"], shape (b, m, l - w + 1)
        Updated coefficient matrix after the optimization step.
    dual_gap : Float[Tensor, "B"]
        Duality gap for convergence assessment.
    nll : Float[Tensor, "B"]
        Negative log likelihood (proportional to MSE).

    Notes
    -----
    The algorithm uses proximal gradient descent to solve:

    minimize_c: ||contribs - conv_transpose(c * importance_scale, cwms) * sequences||²₂ + λ||c||₁

    subject to: c ≥ 0

    References
    ----------
    - Proximal gradient descent: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slide 22
    - Duality gap computation: https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """
    # Forward pass: convolution operations require specific tensor layouts
    coef_adj = coefficients * importance_scale
    pred_unmasked = F.conv_transpose1d(coef_adj, cwms)  # (b, 4, l)
    pred = (
        pred_unmasked * sequences
    )  # (b, 4, l), element-wise masking for projected mode

    # Compute gradient * -1
    residuals = contribs - pred  # (b, 4, l)
    ngrad = F.conv1d(residuals, cwms) * importance_scale  # (b, m, l - w + 1)

    # Negative log likelihood (proportional to MSE)
    nll = (residuals**2).sum(dim=(1, 2))  # (b)

    # Compute duality gap for convergence assessment
    dual_norm = (ngrad / lambdas).amax(dim=(1, 2))  # (b)
    dual_scale = (torch.clamp(1 / dual_norm, max=1.0) ** 2 + 1) / 2  # (b)
    nll_scaled = nll * dual_scale  # (b)

    dual_diff = (residuals * contribs).sum(dim=(1, 2))  # (b)
    l1_term = (torch.abs(coefficients).sum(dim=2, keepdim=True) * lambdas).sum(
        dim=(1, 2)
    )  # (b)
    dual_gap = (nll_scaled - dual_diff + l1_term).abs()  # (b)

    # Compute proximal gradient descent step
    c_next = coefficients + step_sizes * (ngrad - lambdas)  # (b, m, l - w + 1)
    c_next = F.relu(c_next)  # Ensure non-negativity constraint

    return c_next, dual_gap, nll


def optimizer_step(
    cwms: Float[Tensor, "M 4 W"],
    contribs: Float[Tensor, "B 4 L"],
    importance_scale: Float[Tensor, "B 1 P"],
    sequences: Union[Int[Tensor, "B 4 L"], int],
    coef_inter: Float[Tensor, "B M P"],
    coef: Float[Tensor, "B M P"],
    i: Float[Tensor, "B 1 1"],
    step_sizes: Float[Tensor, "B 1 1"],
    sequence_length: int,
    lambdas: Float[Tensor, "1 M 1"],
) -> Tuple[
    Float[Tensor, "B M P"],
    Float[Tensor, "B M P"],
    Float[Tensor, " B"],
    Float[Tensor, " B"],
]:
    """Perform a non-negative lasso optimizer step with Nesterov momentum.

    This function combines proximal gradient descent with momentum acceleration
    to improve convergence speed while maintaining the non-negative constraint
    on coefficients.

    B = batch size, M = number of motifs, L = sequence length, W = motif width.
    P = L - W + 1 (the number of positions with coefficients).

    Parameters
    ----------
    cwms : Float[Tensor, "M 4 W"]
        Motif contribution weight matrices.
    contribs : Float[Tensor, "B 4 L"]
        Target contribution scores.
    importance_scale : Float[Tensor, "B 1 P"]
        Importance scaling factors.
    sequences : Union[Int[Tensor, "B 4 L"], int]
        One-hot encoded sequences or scalar for hypothetical mode.
    coef_inter : Float[Tensor, "B M P"]
        Intermediate coefficient matrix (with momentum).
    coef : Float[Tensor, "B M P"]
        Current coefficient matrix.
    i : Float[Tensor, "B 1 1"]
        Iteration counter for each batch element.
    step_sizes :  Float[Tensor, "B 1 1"]
        Step sizes for optimization.
    sequence_length : int
        Sequence length for normalization.
    lambdas : Float[Tensor, "1 M 1"]
        Regularization parameters.

    Returns
    -------
    coef_inter : Float[Tensor, "B M P"]
        Updated intermediate coefficients with momentum.
    coef : Float[Tensor, "B M P"]
        Updated coefficient matrix.
    gap : Float[Tensor, " B"]
        Normalized duality gap.
    nll : Float[Tensor, " B"]
        Normalized negative log likelihood.

    Notes
    -----
    Uses Nesterov momentum with momentum coefficient i/(i+3) for improved
    convergence properties. The duality gap and NLL are normalized by
    sequence length for scale-invariant convergence assessment.

    References
    ----------
    https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slides 22, 27
    """
    coef_prev = coef

    # Proximal gradient descent step
    coef, gap, nll = prox_grad_step(
        coef_inter, importance_scale, cwms, contribs, sequences, lambdas, step_sizes
    )
    gap = gap / sequence_length
    nll = nll / (2 * sequence_length)

    # Compute updated coefficients with Nesterov momentum
    mom_term = i / (i + 3.0)
    coef_inter = (1 + mom_term) * coef - mom_term * coef_prev

    return coef_inter, coef, gap, nll


def _to_channel_last_layout(tensor: Tensor, **kwargs) -> torch.Tensor:
    """Convert tensor to channel-last memory layout for optimized convolution operations.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to convert.
    **kwargs
        Additional keyword arguments passed to tensor.to().

    Returns
    -------
    torch.Tensor
        Tensor with channel-last memory layout.
    """
    return (
        tensor[:, :, :, None].to(memory_format=torch.channels_last, **kwargs).squeeze(3)
    )


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Apply signed square root transformation to input tensor.

    This transformation preserves the sign while applying square root to the
    absolute value, which can help with numerical stability and gradient flow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Transformed tensor with same shape as input.
    """
    return torch.sign(x) * torch.sqrt(torch.abs(x))


class BatchLoaderBase(ABC):
    """Base class for loading batches of contribution scores and sequences.

    This class provides common functionality for different input formats
    including batch indexing and padding for consistent batch sizes.

    N = number of sequences, L = sequence length.

    Parameters
    ----------
    contribs : Union[Float[Tensor, "N 4 L"], Float[Tensor, "N L"]]
        Contribution scores array.
    sequences : Int[Tensor, "N 4 L"]
        One-hot encoded sequences array.
    sequence_length : int
        Sequence length.
    device : torch.device
        Target device for tensor operations.
    """

    def __init__(
        self,
        contribs: Union[Float[Tensor, "N 4 L"], Float[Tensor, "N L"]],
        sequences: Int[Tensor, "N 4 L"],
        sequence_length: int,
        device: torch.device,
    ) -> None:
        self.contribs = contribs
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.device = device

    def _get_inds_and_pad_lens(
        self, start: int, end: int
    ) -> Tuple[Int[Tensor, " Z"], Tuple[int, ...]]:
        """Get indices and padding lengths for batch loading.

        Parameters
        ----------
        start : int
            Start index for batch.
        end : int
            End index for batch.

        Returns
        -------
        inds : Int[Tensor, " Z"]
            Padded indices tensor with -1 for padding positions.
        pad_lens : tuple
            Padding specification for F.pad (left, right, top, bottom, front, back).
        """
        n = end - start
        end = min(end, self.contribs.shape[0])
        overhang = n - (end - start)
        pad_lens = (0, 0, 0, 0, 0, overhang)

        inds = F.pad(
            torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1
        ).to(device=self.device)

        return inds, pad_lens

    @abstractmethod
    def load_batch(
        self, start: int, end: int
    ) -> Tuple[
        Float[Tensor, "B 4 L"], Union[Int[Tensor, "B 4 L"], int], Int[Tensor, " B"]
    ]:
        """Load a batch of data.

        B = batch size, L = sequence length.

        Parameters
        ----------
        start : int
            Start index (used by subclasses).
        end : int
            End index (used by subclasses).

        Returns
        -------
        contribs_batch : Float[Tensor, "B 4 L"]
            Batch of contribution scores.
        sequences_batch : Union[Int[Tensor, "B 4 L"], int]
            Batch of sequences or scalar for hypothetical mode.
        inds_batch : Int[Tensor, "B"]
            Batch indices.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Parameters are intentionally unused in the base implementation.
        """
        pass


class BatchLoaderCompactFmt(BatchLoaderBase):
    """Batch loader for compact format contribution scores.

    Handles contribution scores in shape (N, L) representing projected
    scores that need to be broadcasted to (N, 4, L) format.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], Int[Tensor, "B 4 L"], Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_compact = F.pad(self.contribs[start:end, None, :], pad_lens)
        contribs_batch = _to_channel_last_layout(
            contribs_compact, device=self.device, dtype=torch.float32
        )
        sequences_batch = F.pad(self.sequences[start:end, :, :], pad_lens)  # (b, 4, l)
        sequences_batch = _to_channel_last_layout(
            sequences_batch, device=self.device, dtype=torch.int8
        )

        contribs_batch = contribs_batch * sequences_batch  # (b, 4, l)

        return contribs_batch, sequences_batch, inds


class BatchLoaderProj(BatchLoaderBase):
    """Batch loader for projected contribution scores.

    Handles contribution scores in shape (N, 4, L) where scores are
    element-wise multiplied by one-hot sequences to get projected contributions.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], Int[Tensor, "B 4 L"], Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_hyp = F.pad(self.contribs[start:end, :, :], pad_lens)
        contribs_hyp = _to_channel_last_layout(
            contribs_hyp, device=self.device, dtype=torch.float32
        )
        sequences_batch = F.pad(self.sequences[start:end, :, :], pad_lens)  # (b, 4, l)
        sequences_batch = _to_channel_last_layout(
            sequences_batch, device=self.device, dtype=torch.int8
        )
        contribs_batch = contribs_hyp * sequences_batch

        return contribs_batch, sequences_batch, inds


class BatchLoaderHyp(BatchLoaderBase):
    """Batch loader for hypothetical contribution scores.

    Handles hypothetical contribution scores in shape (N, 4, L) where
    scores represent counterfactual effects of base substitutions.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], int, Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_batch = F.pad(self.contribs[start:end, :, :], pad_lens)
        contribs_batch = _to_channel_last_layout(
            contribs_batch, device=self.device, dtype=torch.float32
        )

        return contribs_batch, 1, inds


def fit_contribs(
    cwms: Float[ndarray, "M 4 W"],
    contribs: Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]],
    sequences: Int[ndarray, "N 4 L"],
    cwm_trim_mask: Float[ndarray, "M W"],
    use_hypothetical: bool,
    lambdas: Float[ndarray, " M"],
    step_size_max: float,
    step_size_min: float,
    sqrt_transform: bool,
    convergence_tol: float,
    max_steps: int,
    batch_size: int,
    step_adjust: float,
    post_filter: bool,
    device: Optional[torch.device],
    compile_optimizer: bool,
    eps: float = 1.0,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Call motif hits by fitting sparse linear model to contribution scores.

    This is the main function implementing the Fi-NeMo algorithm. It identifies
    motif instances by solving a sparse reconstruction problem where contribution
    scores are approximated as a linear combination of motif CWMs at specific
    positions. The optimization uses proximal gradient descent with momentum.

    Parameters
    ----------
    cwms : Float[ndarray, "M 4 W"]
        Motif contribution weight matrices where M = number of motifs,
        4 = DNA bases (A, C, G, T), W = motif width.
    contribs : Float[ndarray, "N 4 L"] | Float[ndarray, "N L"]
        Neural network contribution scores where N = number of regions,
        L = sequence length. Can be hypothetical (N, 4, L) or projected (N, L).
    sequences : Float[ndarray, "N 4 L"]
        One-hot encoded DNA sequences.
    cwm_trim_mask : Float[ndarray, "M W"]
        Binary mask indicating which positions of each CWM to use.
    use_hypothetical : bool
        Whether to use hypothetical contribution scores (True) or
        projected scores (False).
    lambdas : Float[ndarray, "M"]
        L1 regularization weights for each motif.
    step_size_max : float
        Maximum optimization step size.
    step_size_min : float
        Minimum optimization step size (for convergence failure detection).
    sqrt_transform : bool
        Whether to apply signed square root transformation to inputs.
    convergence_tol : float
        Convergence tolerance based on duality gap.
    max_steps : int
        Maximum number of optimization steps.
    batch_size : int
        Number of regions to process simultaneously.
    step_adjust : float
        Factor to reduce step size when optimization diverges.
    post_filter : bool
        Whether to filter hits based on similarity threshold.
    device : torch.device, optional
        Target device for computation. Auto-detected if None.
    compile_optimizer : bool
        Whether to JIT compile the optimizer for speed.
    eps : float, default 1.0
        Small constant for numerical stability.

    Returns
    -------
    hits_df : pl.DataFrame
        DataFrame containing called motif hits with columns:
        - peak_id: Region index
        - motif_id: Motif index
        - hit_start: Start position of hit
        - hit_coefficient: Hit strength coefficient
        - hit_similarity: Cosine similarity with motif
        - hit_importance: Total contribution score in hit region
        - hit_importance_sq: Sum of squared contributions (for normalization)
    qc_df : pl.DataFrame
        DataFrame containing quality control metrics with columns:
        - peak_id: Region index
        - nll: Final negative log likelihood
        - dual_gap: Final duality gap
        - num_steps: Number of optimization steps
        - step_size: Final step size
        - global_scale: Region-level scaling factor

    Notes
    -----
    The algorithm solves the optimization problem:

    minimize_c: ||contribs - Σⱼ convolve(c * scale, cwms[j]) * sequences||²₂ + Σⱼ λⱼ||c[:,j]||₁

    subject to: c ≥ 0

    where c[i,j] represents the strength of motif j at position i.

    The importance scaling balances reconstruction across different
    motifs and positions based on the local contribution magnitude.

    Examples
    --------
    >>> hits_df, qc_df = fit_contribs(
    ...     cwms=motif_cwms,
    ...     contribs=contrib_scores,
    ...     sequences=onehot_seqs,
    ...     cwm_trim_mask=trim_masks,
    ...     use_hypothetical=False,
    ...     lambdas=np.array([0.7, 0.8]),
    ...     step_size_max=3.0,
    ...     step_size_min=0.08,
    ...     sqrt_transform=False,
    ...     convergence_tol=0.0005,
    ...     max_steps=10000,
    ...     batch_size=1000,
    ...     step_adjust=0.7,
    ...     post_filter=True,
    ...     device=None,
    ...     compile_optimizer=False
    ... )
    """
    m, _, w = cwms.shape
    n, _, sequence_length = sequences.shape

    b = batch_size

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("No GPU available. Running on CPU.", RuntimeWarning)

    # Compile optimizer if requested
    global optimizer_step
    if compile_optimizer:
        optimizer_step = torch.compile(optimizer_step, fullgraph=True)

    # Convert inputs to PyTorch tensors with proper device placement
    cwms_tensor: torch.Tensor = torch.from_numpy(cwms)
    contribs_tensor: torch.Tensor = torch.from_numpy(contribs)
    sequences_tensor: torch.Tensor = torch.from_numpy(sequences)
    cwm_trim_mask_tensor = torch.from_numpy(cwm_trim_mask)[:, None, :].repeat(1, 4, 1)
    lambdas_tensor: torch.Tensor = torch.from_numpy(lambdas)[None, :, None].to(
        device=device, dtype=torch.float32
    )

    # Convert to channel-last layout for optimized convolution operations
    cwms_tensor = _to_channel_last_layout(
        cwms_tensor, device=device, dtype=torch.float32
    )
    cwm_trim_mask_tensor = _to_channel_last_layout(
        cwm_trim_mask_tensor, device=device, dtype=torch.float32
    )
    cwms_tensor = cwms_tensor * cwm_trim_mask_tensor  # Apply trimming mask

    if sqrt_transform:
        cwms_tensor = _signed_sqrt(cwms_tensor)
        cwm_norm = (cwms_tensor**2).sum(dim=(1, 2)).sqrt()
        cwms_tensor = cwms_tensor / cwm_norm[:, None, None]

    # Initialize batch loader
    if len(contribs_tensor.shape) == 3:
        if use_hypothetical:
            batch_loader = BatchLoaderHyp(
                contribs_tensor, sequences_tensor, sequence_length, device
            )
        else:
            batch_loader = BatchLoaderProj(
                contribs_tensor, sequences_tensor, sequence_length, device
            )
    elif len(contribs_tensor.shape) == 2:
        if use_hypothetical:
            raise ValueError(
                "Input regions do not contain hypothetical contribution scores"
            )
        else:
            batch_loader = BatchLoaderCompactFmt(
                contribs_tensor, sequences_tensor, sequence_length, device
            )
    else:
        raise ValueError(
            f"Input contributions array is of incorrect shape {contribs_tensor.shape}"
        )

    # Initialize output container objects
    hit_idxs_lst: List[ndarray] = []
    coefficients_lst: List[ndarray] = []
    similarity_lst: List[ndarray] = []
    importance_lst: List[ndarray] = []
    importance_sq_lst: List[ndarray] = []
    qc_lsts: Dict[str, List[ndarray]] = {
        "nll": [],
        "dual_gap": [],
        "num_steps": [],
        "step_size": [],
        "global_scale": [],
        "peak_id": [],
    }

    # Initialize buffers for optimizer
    coef_inter: Float[Tensor, "B M P"] = torch.zeros(
        (b, m, sequence_length - w + 1)
    )  # (b, m, sequence_length - w + 1)
    coef_inter = _to_channel_last_layout(coef_inter, device=device, dtype=torch.float32)
    coef: Float[Tensor, "B M P"] = torch.zeros_like(coef_inter)
    i: Float[Tensor, "B 1 1"] = torch.zeros((b, 1, 1), dtype=torch.int, device=device)
    step_sizes: Float[Tensor, "B 1 1"] = torch.full(
        (b, 1, 1), step_size_max, dtype=torch.float32, device=device
    )

    converged: Bool[Tensor, " B"] = torch.full(
        (b,), True, dtype=torch.bool, device=device
    )
    num_load = b

    contribs_buf: Float[Tensor, "B 4 L"] = torch.zeros((b, 4, sequence_length))
    contribs_buf = _to_channel_last_layout(
        contribs_buf, device=device, dtype=torch.float32
    )

    seqs_buf: Union[Int[Tensor, "B 4 L"], int]
    if use_hypothetical:
        seqs_buf = 1
    else:
        seqs_buf = torch.zeros((b, 4, sequence_length))
        seqs_buf = _to_channel_last_layout(seqs_buf, device=device, dtype=torch.int8)

    importance_scale_buf: Float[Tensor, "B M P"] = torch.zeros(
        (b, m, sequence_length - w + 1)
    )
    importance_scale_buf = _to_channel_last_layout(
        importance_scale_buf, device=device, dtype=torch.float32
    )

    inds_buf: Int[Tensor, " B"] = torch.zeros((b,), dtype=torch.int, device=device)
    global_scale_buf: Float[Tensor, " B"] = torch.zeros(
        (b,), dtype=torch.float, device=device
    )

    with tqdm(disable=None, unit="regions", total=n, ncols=120) as pbar:
        num_complete = 0
        next_ind = 0
        while num_complete < n:
            # Retire converged peaks and fill buffer with new data
            if num_load > 0:
                load_start = next_ind
                load_end = load_start + num_load
                next_ind = min(load_end, contribs_tensor.shape[0])

                batch_data = batch_loader.load_batch(int(load_start), int(load_end))
                contribs_batch, seqs_batch, inds_batch = batch_data

                if sqrt_transform:
                    contribs_batch = _signed_sqrt(contribs_batch)

                global_scale_batch = (
                    (contribs_batch**2).sum(dim=(1, 2)) / sequence_length
                ).sqrt()
                contribs_batch = torch.nan_to_num(
                    contribs_batch / global_scale_batch[:, None, None]
                )

                importance_scale_batch = (
                    F.conv1d(contribs_batch**2, cwm_trim_mask_tensor) + eps
                ) ** (-0.5)
                importance_scale_batch = importance_scale_batch.clamp(max=10)

                contribs_buf[converged, :, :] = contribs_batch
                if not use_hypothetical:
                    seqs_buf[converged, :, :] = seqs_batch  # type: ignore

                importance_scale_buf[converged, :, :] = importance_scale_batch

                inds_buf[converged] = inds_batch
                global_scale_buf[converged] = global_scale_batch

                coef_inter[converged, :, :] *= 0
                coef[converged, :, :] *= 0
                i[converged] *= 0

                step_sizes[converged] = step_size_max

            # Optimization step
            coef_inter, coef, gap, nll = optimizer_step(
                cwms_tensor,
                contribs_buf,
                importance_scale_buf,
                seqs_buf,
                coef_inter,
                coef,
                i,
                step_sizes,
                sequence_length,
                lambdas_tensor,
            )
            i += 1

            # Assess convergence of each peak being optimized. Reset diverged peaks with lower step size.
            active = inds_buf >= 0

            diverged = ~torch.isfinite(gap) & active
            coef_inter[diverged, :, :] *= 0
            coef[diverged, :, :] *= 0
            i[diverged] *= 0
            step_sizes[diverged, :, :] *= step_adjust

            timeouts = (i > max_steps).squeeze() & active
            if timeouts.sum().item() > 0:
                timeout_inds = inds_buf[timeouts]
                for ind in timeout_inds:
                    warnings.warn(
                        f"Region {ind} has not converged within max_steps={max_steps} iterations.",
                        RuntimeWarning,
                    )

            fails = (step_sizes < step_size_min).squeeze() & active
            if fails.sum().item() > 0:
                fail_inds = inds_buf[fails]
                for ind in fail_inds:
                    warnings.warn(f"Optimizer failed for region {ind}.", RuntimeWarning)

            converged = ((gap <= convergence_tol) | timeouts | fails) & active
            num_load = converged.sum().item()

            # Extract hits from converged peaks
            if num_load > 0:
                inds_out = inds_buf[converged]
                global_scale_out = global_scale_buf[converged]

                # Compute hit scores
                coef_out = coef[converged, :, :]
                importance_scale_out_dense = importance_scale_buf[converged, :, :]
                importance_sq = importance_scale_out_dense ** (-2) - eps
                xcor_scale = importance_sq.sqrt()

                contribs_converged = contribs_buf[converged, :, :]
                importance_sum_out_dense = F.conv1d(
                    torch.abs(contribs_converged), cwm_trim_mask_tensor
                )
                xcov_out_dense = F.conv1d(contribs_converged, cwms_tensor)
                # xcov_out_dense = F.conv1d(torch.abs(contribs_converged), cwms_tensor)
                xcor_out_dense = xcov_out_dense / xcor_scale

                if post_filter:
                    coef_out = coef_out * (xcor_out_dense >= lambdas_tensor)

                # Extract hit coordinates using sparse tensor representation
                coef_out = coef_out.to_sparse()

                # Tensor indexing operations for hit extraction
                hit_idxs_out = torch.clone(coef_out.indices())  # Sparse tensor indices
                hit_idxs_out[0, :] = F.embedding(
                    hit_idxs_out[0, :], inds_out[:, None]
                ).squeeze()  # Embedding lookup with complex indexing
                # Map buffer index to peak index

                ind_tuple = torch.unbind(coef_out.indices())
                importance_out = importance_sum_out_dense[ind_tuple]
                importance_sq_out = importance_sq[ind_tuple]
                xcor_out = xcor_out_dense[ind_tuple]

                scores_out_raw = coef_out.values()

                # Store outputs
                gap_out = gap[converged]
                nll_out = nll[converged]
                step_out = i[converged, 0, 0]
                step_sizes_out = step_sizes[converged, 0, 0]

                hit_idxs_lst.append(hit_idxs_out.numpy(force=True).T)
                coefficients_lst.append(scores_out_raw.numpy(force=True))
                similarity_lst.append(xcor_out.numpy(force=True))
                importance_lst.append(importance_out.numpy(force=True))
                importance_sq_lst.append(importance_sq_out.numpy(force=True))

                qc_lsts["nll"].append(nll_out.numpy(force=True))
                qc_lsts["dual_gap"].append(gap_out.numpy(force=True))
                qc_lsts["num_steps"].append(step_out.numpy(force=True))
                qc_lsts["global_scale"].append(global_scale_out.numpy(force=True))
                qc_lsts["step_size"].append(step_sizes_out.numpy(force=True))
                qc_lsts["peak_id"].append(inds_out.numpy(force=True).astype(np.uint32))

                num_complete += num_load
                pbar.update(num_load)

    # Merge outputs into arrays
    hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
    scores_coefficient = np.concatenate(coefficients_lst, axis=0)
    scores_similarity = np.concatenate(similarity_lst, axis=0)
    scores_importance = np.concatenate(importance_lst, axis=0)
    scores_importance_sq = np.concatenate(importance_sq_lst, axis=0)

    hits: Dict[str, ndarray] = {
        "peak_id": hit_idxs[:, 0].astype(np.uint32),
        "motif_id": hit_idxs[:, 1].astype(np.uint32),
        "hit_start": hit_idxs[:, 2],
        "hit_coefficient": scores_coefficient,
        "hit_similarity": scores_similarity,
        "hit_importance": scores_importance,
        "hit_importance_sq": scores_importance_sq,
    }

    qc: Dict[str, ndarray] = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    hits_df = pl.DataFrame(hits)
    qc_df = pl.DataFrame(qc)

    return hits_df, qc_df
