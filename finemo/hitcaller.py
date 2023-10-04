import warnings

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm, trange

from .torch_utils import compile_if_possible

# from torch.profiler import profile, record_function, ProfilerActivity ####


# @compile_if_possible(fullgraph=True)
def prox_grad_step(coefficients, cwms_t, contribs, pred_mask, 
                   a_const, b_const, st_thresh, shrink_factor, step_sizes):
    """
    coefficients: (b, m, l + 2w - 2)
    cwms_t: (m, 4, w)
    contribs: (b, 4, l + w - 1)
    pred_mask: (b, 4, l + w - 1)

    # https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """
    pred_unmasked = F.conv1d(coefficients, cwms_t) # (b, 4, l + w - 1)
    pred = pred_unmasked * pred_mask # (b, 4, l + w - 1)

    residuals = contribs - pred # (b, 4, l + w - 1)
    ngrad = F.conv_transpose1d(residuals, cwms_t) # (b, m, l + 2w - 2)
    # print(ngrad) ####

    ll = (residuals**2).sum(dim=(1,2)) # (b)
    
    # dual_norm = (ngrad - b_const * coefficients).abs().amax(dim=(1,2)) # (b)
    dual_norm = (ngrad - b_const * coefficients).amax(dim=(1,2)) # (b)
    dual_scale = (torch.clamp(a_const / dual_norm, max=1.)**2 + 1) / 2 # (b)
    ll_scaled = ll * dual_scale # (b)

    dual_diff = (residuals * contribs).sum(dim=(1,2)) # (b)

    l1_term = a_const * torch.linalg.vector_norm(coefficients, ord=1, dim=(1,2)) # (b)
    l2_term = b_const * torch.sum(coefficients**2, dim=(1,2)) # (b)

    dual_gap = (ll_scaled - dual_diff + l1_term + l2_term).abs() # (b)

    c_next = coefficients + step_sizes * ngrad # (b, m, l + 2w - 2)
    # c_next = (c_next - torch.clamp(c_next, min=-st_thresh, max=st_thresh)) / shrink_factor
    #     # (b, m, l + 2w - 2)
    c_next = F.relu(c_next - st_thresh) / shrink_factor # (b, m, l + 2w - 2)

    return c_next, dual_gap, ll


def optimizer(cwms_t, l, a_const, b_const):
    """
    cwms_t: (m, 4, w)
    """

    contribs, sequences, c_a, c_b, i, step_sizes = yield

    while True:
        st_thresh = a_const * step_sizes
        shrink_factor = 1 + b_const * step_sizes

        c_b_prev = c_b
        c_b, gap, ll = prox_grad_step(c_a, cwms_t, contribs, sequences, a_const, b_const, st_thresh, shrink_factor, step_sizes)
        gap = gap / l
        ll = ll / (2 * l)

        contribs, sequences, c_a, c_b, i, step_sizes = yield c_a, c_b, gap, ll

        mom_term = i / (i + 3.)
        c_a = (1 + mom_term) * c_b - mom_term * c_b_prev


def _load_batch_compact_fmt(contribs, sequences, start, end, motif_width, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    lpad = motif_width - 1
    pad_lens = (0, lpad, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_compact = F.pad(contribs[start:end,None,:], pad_lens).float().to(device=device)
    sequences_batch = F.pad(sequences[start:end,:,:], pad_lens).to(device=device) # (b, 4, l + w - 1)
    contribs_batch = contribs_compact * sequences_batch

    scale = ((contribs_compact**2).sum(dim=(1,2), keepdim=True) / l).sqrt()
    contribs_batch = (contribs_compact / scale) * sequences_batch # (b, 4, l + w - 1)

    return contribs_batch, sequences_batch, inds, scale.squeeze()


def _load_batch_non_hyp(contribs, sequences, start, end, motif_width, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    lpad = motif_width - 1
    pad_lens = (0, lpad, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_hyp = F.pad(contribs[start:end,:,:], pad_lens).float().to(device=device) 
    sequences_batch = F.pad(sequences[start:end,:,:], pad_lens).to(device=device) # (b, 4, l + w - 1)
    contribs_batch = contribs_hyp * sequences_batch

    scale = ((contribs_batch**2).sum(dim=(1,2), keepdim=True) / l).sqrt()
    contribs_batch /= scale

    return contribs_batch, sequences_batch, inds, scale.squeeze()


def _load_batch_hyp(contribs, sequences, start, end, motif_width, l, device):
    n = end - start
    end = min(end, contribs.shape[0])
    overhang = n - (end - start)
    lpad = motif_width - 1
    pad_lens = (0, lpad, 0, 0, 0, overhang)

    inds = F.pad(torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1).to(device=device)

    contribs_batch = F.pad(contribs[start:end,:,:], pad_lens).float().to(device=device) 

    scale = ((contribs_batch**2).sum(dim=(1,2), keepdim=True) / l).sqrt()
    contribs_batch /= scale

    return contribs_batch, 1, inds, scale.squeeze()


def fit_contribs(cwms, contribs, sequences, use_hypothetical, alpha, l1_ratio, step_size_max, 
                 convergence_tol, max_steps, buffer_size, step_adjust, device):
    """
    cwms: (4, m, w)
    contribs: (n, 4, l) or (n, l)  
    sequences: (n, 4, l)
    """
    _, m, w = cwms.shape
    n, _, l = sequences.shape

    b = buffer_size

    if len(contribs.shape) == 3:
        if use_hypothetical:
            load_batch_fn = _load_batch_hyp
        else:
            load_batch_fn = _load_batch_non_hyp
    elif len(contribs.shape) == 2:
        if use_hypothetical:
            raise ValueError("Input regions do not contain hypothetical contribution scores")
        else:
            load_batch_fn = _load_batch_compact_fmt
    else:
        raise ValueError(f"Input contributions array is of incorrect shape {contribs.shape}")

    a_const = alpha * l1_ratio
    b_const = alpha * (1 - l1_ratio)

    cwms = torch.from_numpy(cwms)
    contribs = torch.from_numpy(contribs)
    sequences = torch.from_numpy(sequences)

    cwms_t = cwms.flip(dims=(2,))
    cwms_t = cwms_t.to(device=device).float()

    seq_inds = torch.arange(l + 2 * w - 2)[None,None,:]
    clip_mask = (seq_inds >= (w - 1)) & (seq_inds < (l - w - 1)) # (l + 2w - 2)
    clip_mask = clip_mask.to(device=device)

    hit_idxs_lst = []
    scores_lst = []
    qc_lsts = {"log_likelihood": [], "dual_gap": [], "num_steps": [], "contrib_scale": []}

    opt_iter = optimizer(cwms_t, l, a_const, b_const)
    opt_iter.send(None)

    c_a = torch.zeros((b, m, l + 2 * w - 2), dtype=torch.float32, device=device) # (b, m, l + 2w - 2)
    c_b = torch.zeros_like(c_a)
    i = torch.zeros((b, 1, 1), dtype=torch.int, device=device)
    step_sizes = torch.full((b, 1, 1), step_size_max, dtype=torch.float32, device=device)
    # dual_gaps = torch.zeros((b,), dtype=torch.float32, device=device)
    
    converged = torch.full((b,), True, dtype=torch.bool, device=device)
    num_load = b

    contribs_buf = torch.zeros((b, 4, l + w - 1), dtype=torch.float32, device=device)
    if use_hypothetical:
        seqs_buf = 1
    else:
        seqs_buf = torch.zeros((b, 4, l + w - 1), dtype=torch.int8, device=device)
    inds_buf = torch.zeros((b,), dtype=torch.int, device=device)
    scale_buf = torch.zeros((b,), dtype=torch.float, device=device)

    with tqdm(disable=None, unit="regions", total=n) as pbar:
        num_complete = 0
        next_ind = 0
        while num_complete < n:
            if num_load > 0:
                load_start = next_ind
                load_end = load_start + num_load
                next_ind = min(load_end, contribs.shape[0])

                contribs_batch, seqs_batch, inds_batch, scale_batch = load_batch_fn(contribs, sequences, load_start, 
                                                                                    load_end, w, l, device)
                contribs_buf[converged,:,:] = contribs_batch
                if not use_hypothetical:
                    seqs_buf[converged,:,:] = seqs_batch

                inds_buf[converged] = inds_batch
                scale_buf[converged] = scale_batch

                c_a[converged,:,:] *= 0
                c_b[converged,:,:] *= 0
                i[converged] *= 0

            c_a, c_b, gap, ll = opt_iter.send((contribs_buf, seqs_buf, c_a, c_b, i, step_sizes),)
            i += 1

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
            print(num_load) ####

            if num_load > 0:
                inds_out = inds_buf[converged]
                scale_out = scale_buf[converged]

                coef_out = (c_a[converged,:,:] * clip_mask).to_sparse()
                print(coef_out) ####
                print(c_a[converged,:,:] * clip_mask) ####

                hit_idxs_out = torch.clone(coef_out.indices())
                hit_idxs_out[0,:] = F.embedding(hit_idxs_out[0,:], inds_out[:,None])
                hit_idxs_out[2,:] -= w - 1

                scores_out = coef_out.values()

                gap_out = gap[converged]
                ll_out = ll[converged]
                step_out = i[converged,0,0]

                hit_idxs_lst.append(hit_idxs_out.numpy(force=True).T)
                scores_lst.append(scores_out.numpy(force=True))

                qc_lsts["log_likelihood"].append(ll_out.numpy(force=True))
                qc_lsts["dual_gap"].append(gap_out.numpy(force=True))
                qc_lsts["num_steps"].append(step_out.numpy(force=True))
                qc_lsts["contrib_scale"].append(scale_out.numpy(force=True))

                num_complete += num_load
                pbar.update(num_complete)

    hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
    scores = np.concatenate(scores_lst, axis=0)

    hits = {
        "peak_id": hit_idxs[:,0].astype(np.uint32),
        "motif_id": hit_idxs[:,1].astype(np.uint32),
        "hit_start": hit_idxs[:,2],
        "hit_score": scores
    }

    qc = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    return hits, qc


    





    

# def fit_batch(cwms_t, contribs, sequences, coef_init, clip_mask,
#               l, a_const, b_const, step_size, convergence_tol, max_steps):
#     """
#     cwms: (4, m, w)
#     cwms_t: (m, 4, w)
#     contribs: (b, 4, l + w - 1)
#     sequences: (b, 4, l + w - 1)
#     coef_init: (b, m, l + 2w - 2)
#     clip_mask: (1, 1, l + 2w - 2)
#     """
#     st_thresh = a_const * step_size
#     shrink_factor = 1 + b_const * step_size

#     c_a = coef_init
#     c_b = torch.zeros_like(c_a)

#     # contribs_sum = contribs.sum(dim=(1,2)) ####
#     # contribs_sum_np = contribs_sum.numpy(force=True) ####

#     converged = False
#     with trange(max_steps, total=np.inf, disable=None, position=1) as tbatch:
#         for i in tbatch:

#             c_b_prev = c_b
#             c_b, gap, ll = prox_grad_step(c_a, cwms_t, contribs, sequences, a_const, b_const, st_thresh, shrink_factor, step_size)
#             gap = gap / l
#             # ll = ll / (2 * l)

#             # with profile(activities=[ ####
#             #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#             #     with record_function("model_inference"):
#             #         c_b, gap, ll = prox_grad_step(c_a, cwms_t, contribs, sequences, a_const, b_const, st_thresh, shrink_factor, step_size)

#             # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)) ####

#             tbatch.set_postfix(max_gap=gap.max().item(), mean_gap=gap.mean().item())
#             # num_nonzero = c_a.count_nonzero((1,2),) ####
#             # tbatch.set_postfix(max_gap=gap.max().item(), mean_gap=gap.mean().item(), max_ll=ll.max().item(), mean_ll=ll.mean().item(), 
#             #                    max_nonzero=num_nonzero.max().item(), mean_nonzero=num_nonzero.float().mean().item()) ####

#             if torch.all(gap <= convergence_tol).item():
#                 converged = True
#                 break

#             mom_term = i / (i + 3.)
#             c_a = (1 + mom_term) * c_b - mom_term * c_b_prev

#     if not converged:
#         warnings.warn(f"Not all regions have converged within max_steps={max_steps} iterations.", RuntimeWarning)

#     coef_final = c_a * clip_mask
#     coef_sparse = coef_final.to_sparse()

#     ll = ll / (2 * l)

#     return coef_sparse, ll, gap, i


# def fit_contribs(cwms, contribs, sequences, use_hypothetical, alpha, l1_ratio, 
#                  step_size, convergence_tol, max_steps, batch_size, device):
#     """
#     cwms: (4, m, w)
#     contribs: (n, 4, l) or (n, l)  
#     sequences: (n, 4, l)
#     """
#     _, m, w = cwms.shape
#     n, _, l = sequences.shape

#     if len(contribs.shape) == 3:
#         if use_hypothetical:
#             load_batch_fn = _load_batch_hyp
#         else:
#             load_batch_fn = _load_batch_non_hyp
#     elif len(contribs.shape) == 2:
#         if use_hypothetical:
#             raise ValueError("Input regions do not contain hypothetical contribution scores")
#         else:
#             load_batch_fn = _load_batch_compact_fmt
#     else:
#         raise ValueError(f"Input contributions array is of incorrect shape {contribs.shape}")

#     # out_size = l + w - 1
#     # a_const = alpha * out_size * l1_ratio
#     # b_const = alpha * out_size * (1 - l1_ratio)

#     a_const = alpha * l1_ratio
#     b_const = alpha * (1 - l1_ratio)

#     # contrib_norm = np.sqrt((contribs**2).mean())

#     cwms = torch.from_numpy(cwms)
#     contribs = torch.from_numpy(contribs)
#     sequences = torch.from_numpy(sequences)

#     cwms_t = cwms.flip(dims=(2,))
#     # cwms_t = cwms_t.to(device=device)
#     cwms_t = cwms_t.to(device=device).float()

#     seq_inds = torch.arange(l + 2 * w - 2)[None,None,:]
#     clip_mask = (seq_inds >= (w - 1)) & (seq_inds < (l - w - 1)) # (l + 2w - 2)
#     clip_mask = clip_mask.to(device=device)

#     num_batches = -(n // -batch_size) # Ceiling division

#     hit_idxs_lst = []
#     scores_lst = []
#     qc_lsts = {"log_likelihood": [], "dual_gap": [], "num_steps": [], "contrib_scale": []}

#     for i in trange(num_batches, disable=None, unit="batches", position=0):
#         start = i * batch_size
#         end = min(n, start + batch_size)
#         b = end - start

#         # sequences_batch = F.pad(sequences[start:end,:,:], (0, w - 1)).to(device=device) # (b, 4, l + w - 1)
#         # contribs_batch = F.pad(contribs[start:end,None,:], (0, w - 1)).half().to(device=device) 
#         # coef_init = torch.zeros((b, m, l + 2 * w - 2), dtype=torch.float16, device=device) # (b, m, l + 2w - 2)

#         # sequences_batch = F.pad(sequences[start:end,:,:], (0, w - 1)).to(device=device) # (b, 4, l + w - 1)
#         # contribs_batch = F.pad(contribs[start:end,None,:], (0, w - 1)).float().to(device=device) 

#         contribs_batch, sequences_batch = load_batch_fn(contribs, sequences, start, end, w, device)
#         coef_init = torch.zeros((b, m, l + 2 * w - 2), dtype=torch.float32, device=device) # (b, m, l + 2w - 2)

#         scale = ((contribs_batch**2).sum(dim=(1,2), keepdim=True) / l).sqrt()
#         contribs_batch = (contribs_batch / scale) * sequences_batch # (b, 4, l + w - 1)

#         coef, ll, gap, steps = fit_batch(cwms_t, contribs_batch, sequences_batch, coef_init, clip_mask,
#                                          l, a_const, b_const, step_size, convergence_tol, max_steps)
        
#         hit_idxs_batch = torch.clone(coef.indices())
#         hit_idxs_batch[0,:] += start
#         hit_idxs_batch[2,:] -= w - 1
#         # print(hit_idxs_batch[:,:10]) ####
#         # print(hit_idxs_batch) ####
#         # print(coef) ####

#         scores_batch = coef.values()

#         hit_idxs_lst.append(hit_idxs_batch.numpy(force=True).T)
#         scores_lst.append(scores_batch.numpy(force=True))

#         qc_lsts["log_likelihood"].append(ll.numpy(force=True))
#         qc_lsts["dual_gap"].append(gap.numpy(force=True))
#         qc_lsts["num_steps"].append(np.full(b, steps, dtype=np.int32))
#         qc_lsts["contrib_scale"].append(scale.squeeze().numpy(force=True))

#         # break ####

#     hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
#     scores = np.concatenate(scores_lst, axis=0)

#     hits = {
#         "peak_id": hit_idxs[:,0].astype(np.uint32),
#         "motif_id": hit_idxs[:,1].astype(np.uint32),
#         "hit_start": hit_idxs[:,2],
#         "hit_score": scores
#     }

#     qc = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

#     return hits, qc


