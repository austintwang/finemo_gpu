import warnings

import numpy as np
import torch
import torch.nn.functional as F
import polars as pl

from tqdm import tqdm


def prox_grad_step(coefficients, importance_scale, cwms, contribs, sequences, 
                   lambdas, step_sizes):
    """
    Proximal gradient descent optimization step for non-negative lasso

    coefficients: (b, m, l - w + 1)
    importance_scale: (b, 1, l - w + 1)
    cwms: (m, 4, w)
    contribs: (b, 4, l)
    sequences: (b, 4, l) or dummy scalar
    lambdas: (1, m, 1)

    For details on proximal gradient descent: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slide 22 
    For details on duality gap computation: https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """
    # Forward pass
    coef_adj = coefficients * importance_scale
    pred_unmasked = F.conv_transpose1d(coef_adj, cwms) # (b, 4, l)
    pred = pred_unmasked * sequences # (b, 4, l)

    # Compute gradient * -1
    residuals = contribs - pred # (b, 4, l)
    ngrad = F.conv1d(residuals, cwms) * importance_scale # (b, m, l - w + 1)

    # Negative log likelihood (proportional to MSE)
    nll = (residuals**2).sum(dim=(1,2)) # (b)
    
    # Compute duality gap
    dual_norm = (ngrad / lambdas).amax(dim=(1,2)) # (b)
    dual_scale = (torch.clamp(1 / dual_norm, max=1.)**2 + 1) / 2 # (b)
    nll_scaled = nll * dual_scale # (b)

    dual_diff = (residuals * contribs).sum(dim=(1,2)) # (b)
    l1_term = (torch.abs(coefficients).sum(dim=2, keepdim=True) * lambdas).sum(dim=(1,2)) # (b)
    # l1_term = torch.linalg.vector_norm((coefficients * lambdas), ord=1, dim=(1,2)) # (b)
    dual_gap = (nll_scaled - dual_diff + l1_term).abs() # (b)

    # Compute proximal gradient descent step
    c_next = coefficients + step_sizes * (ngrad - lambdas) # (b, m, l - w + 1)
    c_next = F.relu(c_next) # (b, m, l - w + 1)

    return c_next, dual_gap, nll


def optimizer_step(cwms, contribs, importance_scale, sequences, coef_inter, coef, i, step_sizes, l, lambdas):
    """
    Non-negative lasso optimizer step with momentum. 

    cwms: (m, 4, w)
    contribs: (b, 4, l)
    importance_scale: (b, 1, l - w + 1)
    sequences: (b, 4, l) or dummy scalar
    coef_inter, coef: (b, m, l - w + 1)
    i, step_sizes: (b,)

    For details on optimization algorithm: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slides 22, 27 
    """
    coef_prev = coef

    # Proximal gradient descent step
    coef, gap, nll = prox_grad_step(coef_inter, importance_scale, cwms, contribs, sequences, 
                                    lambdas, step_sizes)
    gap = gap / l
    nll = nll / (2 * l)

    # Compute updated coefficients
    mom_term = i / (i + 3.)
    coef_inter = (1 + mom_term) * coef - mom_term * coef_prev

    return coef_inter, coef, gap, nll


def _to_channel_last_layout(tensor, **kwargs):
    return tensor[:,:,:,None].to(memory_format=torch.channels_last, **kwargs).squeeze(3)


def _signed_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


class BatchLoaderBase:
    def __init__(self, contribs, sequences, l, device):
        self.contribs = contribs
        self.sequences = sequences
        self.l = l
        self.device = device

    def _get_inds_and_pad_lens(self, start, end):
        n = end - start
        end = min(end, self.contribs.shape[0])
        overhang = n - (end - start)
        pad_lens = (0, 0, 0, 0, 0, overhang)

        inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=self.device)

        return inds, pad_lens

    def load_batch(self, start, end):
        raise NotImplementedError
    

class BatchLoaderCompactFmt(BatchLoaderBase):
    def load_batch(self, start, end):
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_compact = F.pad(self.contribs[start:end,None,:], pad_lens)
        contribs_batch = _to_channel_last_layout(contribs_compact, device=self.device, dtype=torch.float32)
        sequences_batch = F.pad(self.sequences[start:end,:,:], pad_lens) # (b, 4, l)
        sequences_batch = _to_channel_last_layout(sequences_batch, device=self.device, dtype=torch.int8)

        # global_scale = ((contribs_batch**2).sum(dim=(1,2)) / self.l).sqrt()

        contribs_batch = contribs_batch * sequences_batch # (b, 4, l)

        return contribs_batch, sequences_batch, inds


class BatchLoaderProj(BatchLoaderBase):
    def load_batch(self, start, end):
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_hyp = F.pad(self.contribs[start:end,:,:], pad_lens)
        contribs_hyp = _to_channel_last_layout(contribs_hyp, device=self.device, dtype=torch.float32) 
        sequences_batch = F.pad(self.sequences[start:end,:,:], pad_lens) # (b, 4, l)
        sequences_batch = _to_channel_last_layout(sequences_batch, device=self.device, dtype=torch.int8)
        contribs_batch = contribs_hyp * sequences_batch

        # global_scale = ((contribs_batch**2).sum(dim=(1,2)) / self.l).sqrt()
        # contribs_batch = torch.nan_to_num(contribs_batch / global_scale[:,None,None])

        return contribs_batch, sequences_batch, inds
    

class BatchLoaderHyp(BatchLoaderBase):
    def load_batch(self, start, end):
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_batch = F.pad(self.contribs[start:end,:,:], pad_lens)
        contribs_batch = _to_channel_last_layout(contribs_batch, device=self.device, dtype=torch.float32)

        # global_scale = ((contribs_batch**2).sum(dim=(1,2)) / self.l).sqrt()
        # contribs_batch = torch.nan_to_num(contribs_batch / global_scale[:,None,None])

        return contribs_batch, 1, inds


def fit_contribs(cwms, contribs, sequences, cwm_trim_mask, use_hypothetical, lambdas, step_size_max, step_size_min, sqrt_transform,
                 convergence_tol, max_steps, batch_size, step_adjust, post_filter, device, compile_optimizer, eps=1.):
    """
    Call hits by fitting sparse linear model to contributions
    
    cwms: (m, 4, w)
    contribs: (n, 4, l) or (n, l)  
    sequences: (n, 4, l)
    cwm_trim_mask: (m, w)
    """
    m, _, w = cwms.shape
    n, _, l = sequences.shape

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

    # Convert inputs to pytorch tensors
    cwms = torch.from_numpy(cwms)
    contribs = torch.from_numpy(contribs)
    sequences = torch.from_numpy(sequences)
    cwm_trim_mask = torch.from_numpy(cwm_trim_mask)[:,None,:].repeat(1,4,1)
    lambdas = torch.from_numpy(lambdas)[None,:,None].to(device=device, dtype=torch.float32)

    cwms = _to_channel_last_layout(cwms, device=device, dtype=torch.float32)
    cwm_trim_mask = _to_channel_last_layout(cwm_trim_mask, device=device, dtype=torch.float32)
    cwms = cwms * cwm_trim_mask

    if sqrt_transform:
        cwms = _signed_sqrt(cwms)
        cwm_norm = (cwms**2).sum(dim=(1,2)).sqrt()
        cwms = cwms / cwm_norm[:,None,None]

    # Initialize batch loader
    if len(contribs.shape) == 3:
        if use_hypothetical:
            batch_loader = BatchLoaderHyp(contribs, sequences, l, device)
        else:
            batch_loader = BatchLoaderProj(contribs, sequences, l, device)
    elif len(contribs.shape) == 2:
        if use_hypothetical:
            raise ValueError("Input regions do not contain hypothetical contribution scores")
        else:
            batch_loader = BatchLoaderCompactFmt(contribs, sequences, l, device)
    else:
        raise ValueError(f"Input contributions array is of incorrect shape {contribs.shape}")

    # Intialize output container objects
    hit_idxs_lst = []
    coefficients_lst = []
    correlation_lst = []
    importance_lst = []
    importance_sq_lst = []
    qc_lsts = {"nll": [], "dual_gap": [], "num_steps": [], "step_size": [], "global_scale": [], "peak_id": []}

    # Initialize buffers for optimizer
    coef_inter = torch.zeros((b, m, l - w + 1)) # (b, m, l - w + 1)
    coef_inter = _to_channel_last_layout(coef_inter, device=device, dtype=torch.float32)
    coef = torch.zeros_like(coef_inter)
    i = torch.zeros((b, 1, 1), dtype=torch.int, device=device)
    step_sizes = torch.full((b, 1, 1), step_size_max, dtype=torch.float32, device=device)
    
    converged = torch.full((b,), True, dtype=torch.bool, device=device)
    num_load = b

    contribs_buf = torch.zeros((b, 4, l))
    contribs_buf = _to_channel_last_layout(contribs_buf, device=device, dtype=torch.float32)

    if use_hypothetical:
        seqs_buf = 1
    else:
        seqs_buf = torch.zeros((b, 4, l))
        seqs_buf = _to_channel_last_layout(seqs_buf, device=device, dtype=torch.int8)

    importance_scale_buf = torch.zeros((b, m, l - w + 1))
    importance_scale_buf = _to_channel_last_layout(importance_scale_buf, device=device, dtype=torch.float32)

    inds_buf = torch.zeros((b,), dtype=torch.int, device=device)
    global_scale_buf = torch.zeros((b,), dtype=torch.float, device=device)

    with tqdm(disable=None, unit="regions", total=n, ncols=120) as pbar:
        num_complete = 0
        next_ind = 0
        while num_complete < n:
            # Retire converged peaks and fill buffer with new data
            if num_load > 0:
                load_start = next_ind
                load_end = load_start + num_load
                next_ind = min(load_end, contribs.shape[0])

                batch_data = batch_loader.load_batch(load_start, load_end)
                contribs_batch, seqs_batch, inds_batch = batch_data

                if sqrt_transform:
                    contribs_batch = _signed_sqrt(contribs_batch)
                
                global_scale_batch = ((contribs_batch**2).sum(dim=(1,2)) / l).sqrt()
                contribs_batch = torch.nan_to_num(contribs_batch / global_scale_batch[:,None,None])

                importance_scale_batch = (F.conv1d(contribs_batch**2, cwm_trim_mask) + eps)**(-0.5)
                importance_scale_batch = importance_scale_batch.clamp(max=10)

                contribs_buf[converged,:,:] = contribs_batch
                if not use_hypothetical:
                    seqs_buf[converged,:,:] = seqs_batch

                importance_scale_buf[converged,:,:] = importance_scale_batch
                
                inds_buf[converged] = inds_batch
                global_scale_buf[converged] = global_scale_batch

                coef_inter[converged,:,:] *= 0
                coef[converged,:,:] *= 0
                i[converged] *= 0

                step_sizes[converged] = step_size_max

            # Optimization step
            coef_inter, coef, gap, nll = optimizer_step(cwms, contribs_buf, importance_scale_buf, seqs_buf, coef_inter, coef, 
                                               i, step_sizes, l, lambdas)
            i += 1

            # Assess convergence of each peak being optimized. Reset diverged peaks with lower step size.
            active = inds_buf >= 0

            diverged = ~torch.isfinite(gap) & active
            coef_inter[diverged,:,:] *= 0
            coef[diverged,:,:] *= 0
            i[diverged] *= 0
            step_sizes[diverged,:,:] *= step_adjust

            timeouts = (i > max_steps).squeeze() & active
            if timeouts.sum().item() > 0:
                timeout_inds = inds_buf[timeouts]
                for ind in timeout_inds:
                    warnings.warn(f"Region {ind} has not converged within max_steps={max_steps} iterations.", RuntimeWarning)

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
                coef_out = coef[converged,:,:]
                importance_scale_out_dense = importance_scale_buf[converged,:,:]
                importance_sq = importance_scale_out_dense**(-2) - eps
                xcor_scale = importance_sq.sqrt()

                contribs_converged = contribs_buf[converged,:,:]
                importance_sum_out_dense = F.conv1d(torch.abs(contribs_converged), cwm_trim_mask)
                xcov_out_dense = F.conv1d(contribs_converged, cwms) 
                # xcov_out_dense = F.conv1d(torch.abs(contribs_converged), cwms) 
                xcor_out_dense = xcov_out_dense / xcor_scale

                if post_filter:
                    coef_out = coef_out * (xcor_out_dense >= lambdas)

                # Extract hit coordinates
                coef_out = coef_out.to_sparse()

                hit_idxs_out = torch.clone(coef_out.indices())
                hit_idxs_out[0,:] = F.embedding(hit_idxs_out[0,:], inds_out[:,None]).squeeze()
                    # Map buffer index to peak index

                ind_tuple = torch.unbind(coef_out.indices())
                importance_out = importance_sum_out_dense[ind_tuple]
                importance_sq_out = importance_sq[ind_tuple]
                xcor_out = xcor_out_dense[ind_tuple]

                scores_out_raw = coef_out.values()

                # Store outputs
                gap_out = gap[converged]
                nll_out = nll[converged]
                step_out = i[converged,0,0]
                step_sizes_out = step_sizes[converged,0,0]

                hit_idxs_lst.append(hit_idxs_out.numpy(force=True).T)
                coefficients_lst.append(scores_out_raw.numpy(force=True))
                correlation_lst.append(xcor_out.numpy(force=True))
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
    scores_correlation = np.concatenate(correlation_lst, axis=0)
    scores_importance = np.concatenate(importance_lst, axis=0)
    scores_importance_sq = np.concatenate(importance_sq_lst, axis=0)

    hits = {
        "peak_id": hit_idxs[:,0].astype(np.uint32),
        "motif_id": hit_idxs[:,1].astype(np.uint32),
        "hit_start": hit_idxs[:,2],
        "hit_coefficient": scores_coefficient,
        "hit_correlation": scores_correlation,
        "hit_importance": scores_importance,
        "hit_importance_sq": scores_importance_sq,
    }

    qc = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    hits_df = pl.DataFrame(hits)
    qc_df = pl.DataFrame(qc)

    return hits_df, qc_df