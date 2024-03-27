import warnings

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm, trange


def prox_grad_step(coefficients, importance_scale, cwms, contribs, pred_mask, 
                   alpha, step_sizes):
    """
    Proximal gradient descent optimization step for non-negative lasso

    coefficients: (b, m, l - w + 1)
    importance_scale: (b, 1, l - w + 1)
    cwms: (m, 4, w)
    contribs: (b, 4, l)
    pred_mask: (b, 4, l) or dummy scalar

    For details on proximal gradient descent: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slide 22 
    For details on duality gap computation: https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """
    # Forward pass
    coef_adj = coefficients * importance_scale
    pred_unmasked = F.conv_transpose1d(coef_adj, cwms) # (b, 4, l)
    pred = pred_unmasked * pred_mask # (b, 4, l)

    # Compute gradient * -1
    residuals = contribs - pred # (b, 4, l)
    ngrad = F.conv1d(residuals, cwms) * importance_scale # (b, m, l - w + 1)

    # Log likelihood (proportional to MSE)
    ll = (residuals**2).sum(dim=(1,2)) # (b)
    
    # Compute duality gap
    dual_norm = ngrad.amax(dim=(1,2)) # (b)
    dual_scale = (torch.clamp(alpha / dual_norm, max=1.)**2 + 1) / 2 # (b)
    ll_scaled = ll * dual_scale # (b)

    dual_diff = (residuals * contribs).sum(dim=(1,2)) # (b)
    l1_term = alpha * torch.linalg.vector_norm(coefficients, ord=1, dim=(1,2)) # (b)
    dual_gap = (ll_scaled - dual_diff + l1_term).abs() # (b)

    # Compute proximal gradient descent step
    c_next = coefficients + step_sizes * (ngrad - alpha) # (b, m, l + w - 1)
    c_next = F.relu(c_next) # (b, m, l - w + 1)

    return c_next, dual_gap, ll


def optimizer_step(cwms, contribs, importance_scale, sequences, c_a, c_b, i, step_sizes, l, alpha):
    """
    Non-negative lasso optimizer step with momentum. 

    cwms: (m, 4, w)
    contribs: (b, 4, l)
    importance_scale: (b, 1, l - w + 1)
    sequences: (b, 4, l) or dummy scalar
    c_a, c_b: (b, m, l - w + 1)
    i, step_sizes: (b,)

    For details on optimization algorithm: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slides 22, 27 
    """
    # Proximal gradient descent step
    c_b_prev = c_b
    c_b, gap, ll = prox_grad_step(c_a, importance_scale, cwms, contribs, sequences, 
                                    alpha, step_sizes)
    gap = gap / l
    ll = ll / (2 * l)

    # Compute updated coefficients
    mom_term = i / (i + 3.)
    c_a = (1 + mom_term) * c_b - mom_term * c_b_prev

    return c_a, c_b, gap, ll


def _load_batch_compact_fmt(contribs, sequences, start, end, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    pad_lens = (0, 0, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_compact = F.pad(contribs[start:end,None,:], pad_lens).float().to(device=device)
    sequences_batch = F.pad(sequences[start:end,:,:], pad_lens).to(device=device) # (b, 4, l)

    global_scale = ((contribs_compact**2).sum(dim=(1,2)) / l).sqrt()

    contribs_batch = (contribs_compact / global_scale[:,None,None]) * sequences_batch # (b, 4, l)

    return contribs_batch, sequences_batch, inds, global_scale


def _load_batch_proj(contribs, sequences, start, end, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    pad_lens = (0, 0, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_hyp = F.pad(contribs[start:end,:,:], pad_lens).float().to(device=device) 
    sequences_batch = F.pad(sequences[start:end,:,:], pad_lens).to(device=device) # (b, 4, l)
    contribs_batch = contribs_hyp * sequences_batch

    global_scale = ((contribs_batch**2).sum(dim=(1,2)) / l).sqrt()
    contribs_batch /= global_scale[:,None,None]

    return contribs_batch, sequences_batch, inds, global_scale


def _load_batch_hyp(contribs, sequences, start, end, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    pad_lens = (0, 0, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_batch = F.pad(contribs[start:end,:,:], pad_lens).float().to(device=device) 

    global_scale = ((contribs_batch**2).sum(dim=(1,2)) / l).sqrt()
    contribs_batch /= global_scale[:,None,None]

    return contribs_batch, 1, inds, global_scale


def fit_contribs(cwms, contribs, sequences, use_hypothetical, alpha, step_size_max, 
                 convergence_tol, max_steps, buffer_size, step_adjust, device):
    """
    Call hits by fitting sparse linear model to contributions
    
    cwms: (m, 4, w)
    contribs: (n, 4, l) or (n, l)  
    sequences: (n, 4, l)
    """
    m, _, w = cwms.shape
    n, _, l = sequences.shape

    b = buffer_size

    if len(contribs.shape) == 3:
        if use_hypothetical:
            load_batch_fn = _load_batch_hyp
        else:
            load_batch_fn = _load_batch_proj
    elif len(contribs.shape) == 2:
        if use_hypothetical:
            raise ValueError("Input regions do not contain hypothetical contribution scores")
        else:
            load_batch_fn = _load_batch_compact_fmt
    else:
        raise ValueError(f"Input contributions array is of incorrect shape {contribs.shape}")

    # Convert inputs to pytorch tensors
    cwms = torch.from_numpy(cwms)
    contribs = torch.from_numpy(contribs)
    sequences = torch.from_numpy(sequences)

    cwms = cwms.to(device=device).float()

    # Intialize output container objects
    hit_idxs_lst = []
    coefficients_lst = []
    correlation_lst = []
    importance_lst = []
    qc_lsts = {"log_likelihood": [], "dual_gap": [], "num_steps": [], "step_size": [], "global_scale": []}

    # Utility convolutional filter for summing values in window 
    sum_filter = torch.ones((1, 4, w), dtype=torch.float32, device=device)

    # Initialize buffers for optimizer
    c_a = torch.zeros((b, m, l - w + 1), dtype=torch.float32, device=device) # (b, m, l - w + 1)
    c_b = torch.zeros_like(c_a)
    i = torch.zeros((b, 1, 1), dtype=torch.int, device=device)
    step_sizes = torch.full((b, 1, 1), step_size_max, dtype=torch.float32, device=device)
    
    converged = torch.full((b,), True, dtype=torch.bool, device=device)
    num_load = b

    contribs_buf = torch.zeros((b, 4, l), dtype=torch.float32, device=device)
    if use_hypothetical:
        seqs_buf = 1
    else:
        seqs_buf = torch.zeros((b, 4, l), dtype=torch.int8, device=device)
    importance_scale_buf = torch.zeros((b, 1, l - w + 1), dtype=torch.float32, device=device)
    inds_buf = torch.zeros((b,), dtype=torch.int, device=device)
    global_scale_buf = torch.zeros((b,), dtype=torch.float, device=device)

    with tqdm(disable=None, unit="regions", total=n) as pbar:
        num_complete = 0
        next_ind = 0
        while num_complete < n:
            # Retire converged peaks and fill buffer with new data
            if num_load > 0:
                load_start = next_ind
                load_end = load_start + num_load
                next_ind = min(load_end, contribs.shape[0])

                batch_data = load_batch_fn(contribs, sequences, load_start, load_end, l, device)
                contribs_batch, seqs_batch, inds_batch, global_scale_batch = batch_data

                importance_scale_batch = (F.conv1d(contribs_batch**2, sum_filter) + 1)**(-0.5)
                importance_scale_batch = importance_scale_batch.clamp(max=10)

                contribs_buf[converged,:,:] = contribs_batch
                if not use_hypothetical:
                    seqs_buf[converged,:,:] = seqs_batch

                importance_scale_buf[converged,:,:] = importance_scale_batch
                
                inds_buf[converged] = inds_batch
                global_scale_buf[converged] = global_scale_batch

                c_a[converged,:,:] *= 0
                c_b[converged,:,:] *= 0
                i[converged] *= 0

                step_sizes[converged] = step_size_max

            # Optimization step
            c_a, c_b, gap, ll = optimizer_step(cwms, contribs_buf, importance_scale_buf, seqs_buf, c_a, c_b, 
                                               i, step_sizes, l, alpha)
            i += 1

            # Assess convergence of each peak being optimized. Reset diverged peaks with lower step size.
            active = inds_buf >= 0

            diverged = ~torch.isfinite(gap) & active
            c_a[diverged,:,:] *= 0
            c_b[diverged,:,:] *= 0
            i[diverged] *= 0
            step_sizes[diverged,:,:] *= step_adjust

            timeouts = (i > max_steps).squeeze()
            if timeouts.sum().item() > 0:
                warnings.warn(f"Not all regions have converged within max_steps={max_steps} iterations.", RuntimeWarning)

            converged = ((gap <= convergence_tol) | timeouts) & active
            num_load = converged.sum().item()

            # Extract hits from converged peaks
            if num_load > 0:
                inds_out = inds_buf[converged]
                global_scale_out = global_scale_buf[converged]

                coef_out = c_b[converged,:,:].to_sparse()

                # Extract hit coordinates
                hit_idxs_out = torch.clone(coef_out.indices())
                hit_idxs_out[0,:] = F.embedding(hit_idxs_out[0,:], inds_out[:,None]).squeeze()
                    # Map buffer index to peak index

                # Compute hit scores 
                importance_scale_out_dense = importance_scale_buf[converged,:]
                xcor_scale = (importance_scale_out_dense**(-2) - 1).sqrt()
                importance_out = xcor_scale[coef_out.indices()[0,:],0,coef_out.indices()[2,:]]

                xcov_out_dense = F.conv1d(contribs_buf[converged,:,:], cwms) 
                xcor_out_dense = xcov_out_dense / xcor_scale

                ind_tuple = torch.unbind(coef_out.indices())
                xcor_out = xcor_out_dense[ind_tuple]

                scores_out_raw = coef_out.values()

                # Store outputs
                gap_out = gap[converged]
                ll_out = ll[converged]
                step_out = i[converged,0,0]
                step_sizes_out = step_sizes[converged,0,0]

                hit_idxs_lst.append(hit_idxs_out.numpy(force=True).T)
                coefficients_lst.append(scores_out_raw.numpy(force=True))
                correlation_lst.append(xcor_out.numpy(force=True))
                importance_lst.append(importance_out.numpy(force=True))

                qc_lsts["log_likelihood"].append(ll_out.numpy(force=True))
                qc_lsts["dual_gap"].append(gap_out.numpy(force=True))
                qc_lsts["num_steps"].append(step_out.numpy(force=True))
                qc_lsts["global_scale"].append(global_scale_out.numpy(force=True))
                qc_lsts["step_size"].append(step_sizes_out.numpy(force=True))

                num_complete += num_load
                pbar.update(num_load)

    # Merge outputs into arrays
    hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
    scores_coefficient = np.concatenate(coefficients_lst, axis=0)
    scores_correlation = np.concatenate(correlation_lst, axis=0)
    scores_importance = np.concatenate(importance_lst, axis=0)

    hits = {
        "peak_id": hit_idxs[:,0].astype(np.uint32),
        "motif_id": hit_idxs[:,1].astype(np.uint32),
        "hit_start": hit_idxs[:,2],
        "hit_coefficient": scores_coefficient,
        "hit_correlation": scores_correlation,
        "hit_importance": scores_importance
    }

    qc = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    return hits, qc

    