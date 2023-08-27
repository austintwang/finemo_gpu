import warnings

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm, trange

def log_likelihood(coefficients, cwms_t, contribs, sequences):
    """
    coefficients: (b, m, l + w - 1)
    cwms_t: (m, 4, w)
    contribs: (b, 4, l)
    sequences: (b, 4, l)
    """
    pred = F.conv1d(coefficients, cwms_t) # (b, 4, l)
    pred_masked = pred * sequences # (b, 4, l)

    ll = F.mse_loss(pred_masked, contribs, reduction='none').sum(dim=(1,2))
    
    return ll, pred_masked


def dual_gap(coefficients, cwms, contribs, pred, ll, a_const, b_const):
    """
    coefficients: (b, m, l + 2w - 2)
    cwms_t: (m, 4, w)
    contribs: (b, 4, l + w - 1)
    pred: (b, 4, l + w - 1)
    ll: (b)

    # https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """

    residuals = contribs - pred # (b, 4, l)
    dual_norm = F.conv_transpose1d(residuals, cwms).abs().amax(dim=(1,2)) # (b)
    dual_scale = torch.clamp(a_const / dual_norm, max=1.)
    ll_scaled = ll * dual_scale

    dual_diff = torch.tensordot(residuals, contribs, dims=([1], [1]))

    l1_term = a_const * torch.linalg.vector_norm(coefficients, ord=1, dim=(1,2))
    l2_term = b_const * torch.sum(coefficients**2, dim=(1,2))

    dual_gap = (ll_scaled - dual_diff + l1_term + l2_term).abs()

    return dual_gap
    

def fit_batch(cwms, cwms_t, contribs, sequences, coef_init, clip_mask,
              a_const, b_const, step_size, convergence_tol, max_steps):
    """
    cwms: (4, m, w)
    cwms_t: (m, 4, w)
    contribs: (b, 4, l + w - 1)
    sequences: (b, 4, l + w - 1)
    coef_init: (b, m, l + 2w - 2)
    clip_mask: (1, 1, l + 2w - 2)
    """
    st_thresh = a_const * step_size
    shrink_factor = 1 + b_const * step_size

    c_a = coef_init
    c_b = torch.zeros_like(c_a)

    converged = False
    with trange(max_steps, total=np.inf, disable=None, position=1) as tbatch:
        for i in tbatch:
            c_a.requires_grad_()
            ll, pred = log_likelihood(c_a, cwms_t, contribs, sequences)
            ll_sum = ll.sum() / 2.

            c_a_grad = torch.autograd.grad(ll_sum, c_a)

            c_a.detach_()
            ll.detach_()
            pred.detach_()
            c_a_grad.detach_()

            gap = dual_gap(c_a, cwms, contribs, pred, ll, a_const, b_const)

            tbatch.set_postfix(max_gap=gap.max().item(), min_gap=gap.min().item())

            if torch.all(gap <= convergence_tol).item():
                converged = True
                break

            c_b_prev = c_b
            c_b = c_a - step_size * c_a_grad
            c_b = (c_b - torch.clamp(c_b, min=-st_thresh, max=st_thresh)) / shrink_factor

            mom_term = i / (i + 3.)
            c_a = (1 + mom_term) * c_b - mom_term * c_b_prev

    if not converged:
        warnings.warn(f"Not all regions have converged within max_steps={max_steps} iterations.", RuntimeWarning)

    coef_final = c_a * clip_mask
    coef_sparse = coef_final.to_sparse()

    ll = ll / 2.

    return coef_sparse, ll, gap, i


def fit_contribs(cwms, contribs, sequences, 
                 alpha, l1_ratio, step_size, convergence_tol, max_steps, batch_size, device):
    """
    cwms: (4, m, w)
    contribs: (n, l)
    sequences: (n, 4, l)
    """
    _, m, w = cwms.shape
    n, _, l = sequences.shape

    out_size = l + w - 1
    a_const = alpha * out_size * l1_ratio
    b_const = alpha * out_size * (1 - l1_ratio)

    cwms = cwms.to(device=device)
    cwms_t = torch.permute(cwms, (1, 0, 2)).flip(dims=(2,))

    contrib_norm = (contribs**2).mean().float().sqrt().item()

    seq_inds = torch.arange(l + 2 * w - 2)[None,None,:]
    clip_mask = (seq_inds >= (w - 1)) & (seq_inds < (l - w - 1)) # (l + 2w - 2)
    clip_mask = clip_mask.to(device=device)

    num_batches = -(n // -batch_size) # Ceiling division

    hit_idxs_lst = []
    scores_lst = []
    qc_lsts = {"log_likelihood": [], "dual_gap": [], "num_steps": []}

    for i in trange(num_batches, disable=None, unit="batches", position=0):
        start = i * batch_size
        end = min(n, start + batch_size)
        b = end - start

        sequences_batch = F.pad(sequences[start:end,:,:], (0, m - 1)).to(device=device) # (b, 4, l + w - 1)
        contribs_batch = F.pad(contribs[start:end,None,:], (0, m - 1)).to(device=device) 
        contribs_batch = (contribs_batch / contrib_norm) * sequences_batch.float() # (b, 4, l + w - 1)
        coef_init = torch.zeros((b, m, l + 2 * w - 2), dtype=torch.float16, device=device) # (b, m, l + 2w - 2)

        coef, ll, gap, steps = fit_batch(cwms, cwms_t, contribs_batch, sequences_batch, coef_init, clip_mask,
                                         a_const, b_const, step_size, convergence_tol, max_steps)
        
        hit_idxs_batch = torch.clone(coef.indices())
        hit_idxs_batch[0:] += start
        hit_idxs_batch[2:] -= m - 1

        scores_batch = coef.values()

        hit_idxs_lst.append(hit_idxs_batch.numpy(force=True))
        scores_lst.append(scores_batch.numpy(force=True))

        qc_lsts["log_likelihood"].append(ll.numpy(force=True))
        qc_lsts["dual_gap"].append(gap.numpy(force=True))
        qc_lsts["num_steps"].append(np.full(b, steps, dtype=np.int32))

    hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
    scores = np.concatenate(scores_lst, axis=0)

    hits = {
        "peak_id": hit_idxs[0:],
        "motif_id": hit_idxs[1:],
        "hit_start": hit_idxs[2:],
        "hit_score": scores
    }

    qc = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    return hits, qc, contrib_norm

