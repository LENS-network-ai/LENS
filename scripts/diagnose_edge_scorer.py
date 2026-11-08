"""
Diagnose how much the relation-aware contrast block contributes to the edge
scorer of a trained LENS / LENS+ checkpoint.

The edge descriptor is e_[ij] = [ x_i | x_j | |x_i - x_j| * a_[ij] ], where
a_[ij] is a softmax over the d feature dimensions (so it sums to 1). That
normalization alone shrinks the contrast block by roughly a factor d relative
to the two unit-norm endpoint blocks unless a is sharply peaked. This script
measures, over a few slides:

  1. block sizes: mean ||x_i||, ||x_j||, || |x_i - x_j| * a ||
  2. entropy of a_[ij] against log d (log d = uniform weights = no selection)
  3. sensitivity of the gate logit: zero out the contrast block (and, as the
     converse, the endpoint blocks) and see how much log alpha changes

Usage (needs a GPU/compute node; importing torch on the login node gets killed):
    python -m scripts.diagnose_edge_scorer \\
        --checkpoint results/.../fold1/phase_models/overall_best_model.pth \\
        --data-root <graphs dir> --list <slide list> --n-slides 6
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from model.EdgeScoring import EdgeScoringNetwork
from model.L0Utils import L0RegularizerParams, l0_test


def load_scorer(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    prefix = 'edge_scorer.'
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(f"No '{prefix}*' keys in {checkpoint_path}; is this a LENS/LENS+ checkpoint?")

    feature_dim = sub['feature_attention.0.weight'].shape[1] // 2
    first = sub.get('edge_mlp.0.weight_orig', sub.get('edge_mlp.0.weight'))
    edge_dim = first.shape[0]

    scorer = EdgeScoringNetwork(feature_dim=feature_dim, edge_dim=edge_dim)
    scorer.load_state_dict(sub)
    return scorer.to(device).eval(), feature_dim, edge_dim


def pick_slides(list_path, n_slides, seed):
    by_class = {}
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.strip().split()
            if len(parts) == 2:
                by_class.setdefault(parts[1].lower(), []).append(parts[0])
    rng = random.Random(seed)
    classes = sorted(by_class)
    chosen = []
    i = 0
    while len(chosen) < n_slides:
        pool = by_class[classes[i % len(classes)]]
        cand = rng.choice(pool)
        if cand not in chosen:
            chosen.append(cand)
        i += 1
    return chosen


def load_edges(slide_dir):
    feats = torch.load(os.path.join(slide_dir, 'features.pt'), map_location='cpu').float()
    adj = torch.load(os.path.join(slide_dir, 'adj_s.pt'), map_location='cpu')
    if adj.is_sparse:
        idx = adj.coalesce().indices()
    else:
        idx = (adj > 0).nonzero(as_tuple=False).t()
    return feats, idx[0], idx[1]


@torch.no_grad()
def analyse_edges(scorer, feats, src, tgt, feature_dim, device, params, tau, chunk=20000):
    """Returns dict of per-edge tensors (cpu) for one slide."""
    x = F.normalize(feats.to(device), p=2, dim=1)
    lin = scorer.edge_mlp[0]
    d = feature_dim
    out = {k: [] for k in ['n_s', 'n_t', 'n_diff', 'n_c', 'ent', 'amax',
                           'pre_s', 'pre_t', 'pre_c',
                           'la_full', 'la_noc', 'la_onlyc']}

    # One call so the spectral-norm weight attribute is populated in eval mode.
    lin(torch.zeros(1, 3 * d, device=device))
    W = lin.weight
    Ws, Wt, Wc = W[:, :d], W[:, d:2 * d], W[:, 2 * d:]

    for start in range(0, src.numel(), chunk):
        s = src[start:start + chunk].to(device)
        t = tgt[start:start + chunk].to(device)
        xs = F.normalize(x[s], p=2, dim=1)
        xt = F.normalize(x[t], p=2, dim=1)
        diff = torch.abs(xs - xt)
        a = scorer.feature_attention(torch.cat([xs, xt], dim=1))
        c = diff * a

        e_full = torch.cat([xs, xt, c], dim=1)
        e_noc = torch.cat([xs, xt, torch.zeros_like(c)], dim=1)
        e_onlyc = torch.cat([torch.zeros_like(xs), torch.zeros_like(xt), c], dim=1)

        ent = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=1)
        out['n_s'].append(xs.norm(dim=1)); out['n_t'].append(xt.norm(dim=1))
        out['n_diff'].append(diff.norm(dim=1)); out['n_c'].append(c.norm(dim=1))
        out['ent'].append(ent); out['amax'].append(a.max(dim=1).values)
        out['pre_s'].append(F.linear(xs, Ws).norm(dim=1))
        out['pre_t'].append(F.linear(xt, Wt).norm(dim=1))
        out['pre_c'].append(F.linear(c, Wc).norm(dim=1))
        out['la_full'].append(scorer.edge_mlp(e_full).squeeze(-1))
        out['la_noc'].append(scorer.edge_mlp(e_noc).squeeze(-1))
        out['la_onlyc'].append(scorer.edge_mlp(e_onlyc).squeeze(-1))

    return {k: torch.cat(v).cpu() for k, v in out.items()}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / den)


def summarize(name, r, d, params, tau):
    log_d = math.log(d)
    gate = lambda la: l0_test(la, params=params, temperature=tau)
    g_full, g_noc = gate(r['la_full']), gate(r['la_noc'])
    d_la = (r['la_full'] - r['la_noc']).abs()

    print(f"\n--- {name}  ({r['n_c'].numel():,} directed edges) ---")
    print(f"  block norms      ||x_i||={r['n_s'].mean():.3f}  ||x_j||={r['n_t'].mean():.3f}  "
          f"|||x_i-x_j|||={r['n_diff'].mean():.3f}  |||x_i-x_j|*a||={r['n_c'].mean():.5f}")
    print(f"  contrast / endpoint norm = {float(r['n_c'].mean() / r['n_s'].mean()):.5f}   "
          f"(uniform-a prediction ~ {float(r['n_diff'].mean()) / d / float(r['n_s'].mean()):.5f})")
    print(f"  first-layer pre-activation norm  W_s x_i={r['pre_s'].mean():.3f}  "
          f"W_t x_j={r['pre_t'].mean():.3f}  W_c c={r['pre_c'].mean():.4f}")
    print(f"  entropy(a)={r['ent'].mean():.3f}  log d={log_d:.3f}  "
          f"ratio={float(r['ent'].mean()) / log_d:.4f}   max(a)*d={float(r['amax'].mean()) * d:.2f} (1.0 = uniform)")
    print(f"  logit  std={r['la_full'].std():.3f}  mean={r['la_full'].mean():.3f}")
    print(f"  deterministic gate (tau={tau}): mean={g_full.mean():.3f}  exact-zero={float((g_full == 0).float().mean()) * 100:.1f}%  "
          f"nonzero={float((g_full > 0).float().mean()) * 100:.1f}%  >0.5={float((g_full > 0.5).float().mean()) * 100:.1f}%  "
          f"(mean gate is what density control targets; exact-zero is what 'edges removed' means)")
    print(f"  zero contrast : mean|d logit|={d_la.mean():.4f}  corr(full, no-contrast)={corr(r['la_full'], r['la_noc']):.4f}  "
          f"gate>0 flips={float(((g_full > 0) != (g_noc > 0)).float().mean()) * 100:.2f}%  "
          f"gate>0.5 flips={float(((g_full > 0.5) != (g_noc > 0.5)).float().mean()) * 100:.2f}%")
    print(f"  contrast only : corr(full, contrast-only)={corr(r['la_full'], r['la_onlyc']):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--list', required=True, help='slide list (id<TAB>label)')
    p.add_argument('--n-slides', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tau', type=float, default=0.67, help='temperature for the deterministic gate')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorer, feature_dim, edge_dim = load_scorer(args.checkpoint, device)
    params = L0RegularizerParams()
    print(f"\ncheckpoint: {args.checkpoint}\nfeature_dim={feature_dim} edge_dim={edge_dim} tau={args.tau}")

    pooled = None
    for slide in pick_slides(args.list, args.n_slides, args.seed):
        feats, src, tgt = load_edges(os.path.join(args.data_root, slide))
        r = analyse_edges(scorer, feats, src, tgt, feature_dim, device, params, args.tau)
        summarize(f"{slide} (N={feats.shape[0]})", r, feature_dim, params, args.tau)
        pooled = r if pooled is None else {k: torch.cat([pooled[k], r[k]]) for k in r}

    summarize("ALL SLIDES POOLED", pooled, feature_dim, params, args.tau)


if __name__ == '__main__':
    main()
