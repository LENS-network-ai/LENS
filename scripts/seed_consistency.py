"""
Seed consistency of the retained-edge explanation, and RoI alignment per seed.

For a set of slides (e.g. the 10 expert-annotated CPTAC-Lung slides) and several
training seeds of the same fold, this script

  1. recomputes the deterministic inference gates Z_inf from each seed's best
     checkpoint (the same forward pass the test evaluation uses) and the
     per-patch retained edge mass  w_i = sum_j z_ij;
  2. for every seed pair reports, averaged over slides:
       - Spearman rho of w_i,
       - top-10% patch Jaccard (among patches with w_i > 0),
       - retained-edge Jaccard (undirected edges with z_ij > --edge-thresh);
  3. reports Spearman rho between w_i and node degree (candidate-graph degree,
     and retained degree);
  4. optionally binarizes each seed's heatmap, compares it with the expert RoI
     and reports precision / recall as mean +- std over seeds;
  5. does the same for CLAM from precomputed per-patch attention scores (CLAM has
     no gates, so edge Jaccard is "-").

Everything is written as ONE table (markdown + CSV) plus a per-slide CSV.

Usage (LENS gates from checkpoints):
    python -m scripts.seed_consistency \\
        --slides annotated10.txt \\
        --lens-run 42=results/seed42/run_x/fold1 \\
        --lens-run 123=results/seed123/run_x/fold1 \\
        --lens-run 2024=results/seed2024/run_x/fold1 \\
        --clam-scores 42=clam/seed42 --clam-scores 123=clam/seed123 \\
        --clam-scores 2024=clam/seed2024 \\
        --roi-dir expert_rois --binarize minmax:0.5 \\
        --coord-unit-px 512 --patch-px 512 \\
        --output-dir results/seed_consistency

Inputs:
  --lens-run SEED=DIR   DIR is a fold directory holding final_model.pt (its saved
                        'config' rebuilds the model) and
                        phase_models/overall_best_model.pth (weights + tau).
  --clam-scores SEED=DIR  DIR holds <slide_id>.npy: one attention score per patch,
                        in the row order of that slide's c_idx.txt.
  --roi-dir DIR         <slide_id>.png binary masks (non-zero = expert RoI) at
                        1/--mask-downsample of level-0 resolution
                        (visualize_heatmap.py uses 10).
  --slides FILE         one slide id per line (first token; a trailing label is ignored).

Table 3's binarization rule is not stored in this repo: set --binarize to the
rule used there, otherwise precision/recall will not be comparable.
"""

import argparse
import csv
import itertools
import math
import os

import numpy as np
from scipy.stats import spearmanr


# ----------------------------------------------------------------------------
# Pure metric functions (numpy only)
# ----------------------------------------------------------------------------

def spearman(a, b):
    """Spearman rho, NaN when either input is constant (rho undefined)."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).correlation)


def jaccard(a, b):
    """Jaccard of two index arrays of unique ints; NaN when the union is empty."""
    union = len(np.union1d(a, b))
    if union == 0:
        return float('nan')
    return len(np.intersect1d(a, b, assume_unique=True)) / union


def top_fraction_set(w, frac):
    """Indices of the top `frac` of patches by w, restricted to patches with w > 0.

    Patches with zero retained mass are never "top": with a mostly-zero w the
    cut-off would otherwise land on arbitrary tied zeros. The set can therefore
    be smaller than ceil(frac * N) (and empty if no patch is active).
    """
    k = int(math.ceil(frac * len(w)))
    active = np.flatnonzero(w > 0)
    if len(active) == 0:
        return active
    order = active[np.argsort(-w[active], kind='stable')]
    return np.sort(order[:k])


def binarize(scores, rule):
    """Per-slide normalize + binarize a heatmap. `rule` is 'minmax:T' or 'top:F'.

    minmax:T  min-max normalize to [0, 1], keep patches >= T
    top:F     keep the top fraction F of patches by score
    Returns a boolean array. A constant heatmap gives an empty mask for minmax.
    """
    kind, _, value = rule.partition(':')
    value = float(value)
    if kind == 'minmax':
        rng = scores.max() - scores.min()
        norm = (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores, dtype=float)
        return norm >= value
    if kind == 'top':
        k = int(math.ceil(value * len(scores)))
        keep = np.argsort(-scores, kind='stable')[:k]
        mask = np.zeros(len(scores), dtype=bool)
        mask[keep] = True
        return mask
    raise ValueError(f"Unknown binarization rule '{rule}' (use minmax:T or top:F)")


def precision_recall(pred, roi):
    """Patch-level precision/recall; precision is NaN for an empty prediction."""
    tp = float(np.logical_and(pred, roi).sum())
    precision = tp / pred.sum() if pred.sum() > 0 else float('nan')
    recall = tp / roi.sum() if roi.sum() > 0 else float('nan')
    return precision, recall


def roi_patch_labels(mask, coords, coord_unit_px, patch_px, mask_downsample):
    """Boolean RoI membership per patch from a binary slide-level mask.

    A patch is in the RoI when at least half of its footprint in the mask is
    non-zero. coords are in units of `coord_unit_px` level-0 pixels (512 for the
    SimCLR graphs, whose c_idx.txt holds grid indices; 1 for the UNI2 graphs,
    whose c_idx.txt holds level-0 pixel coordinates).
    """
    labels = np.zeros(len(coords), dtype=bool)
    size = max(int(round(patch_px / mask_downsample)), 1)
    for i, (x, y) in enumerate(coords):
        x0 = int(round(x * coord_unit_px / mask_downsample))
        y0 = int(round(y * coord_unit_px / mask_downsample))
        box = mask[y0:y0 + size, x0:x0 + size]
        labels[i] = box.size > 0 and (box > 0).mean() >= 0.5
    return labels


def mean_std(values):
    """nan-aware mean and sample std (ddof=1) -> (mean, std)."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan')
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(mean, std, scale=1.0, digits=3):
    if np.isnan(mean):
        return 'n/a'
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


# ----------------------------------------------------------------------------
# Data and model loading (torch imported lazily)
# ----------------------------------------------------------------------------

def read_slide_ids(path):
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def read_coords(data_root, slide_id):
    path = os.path.join(data_root, slide_id, 'c_idx.txt')
    return np.loadtxt(path, dtype=float).reshape(-1, 2)


def parse_seed_pairs(items):
    out = {}
    for item in items:
        seed, _, path = item.partition('=')
        out[int(seed)] = path
    return out


def build_model_from_run(run_dir, device):
    """Rebuild ImprovedEdgeGNN from final_model.pt's config and load the
    best-validation weights (and the temperature they were trained under)."""
    import torch
    from model.LENS2 import ImprovedEdgeGNN

    final = torch.load(os.path.join(run_dir, 'final_model.pt'), map_location=device, weights_only=False)
    cfg = final['config']

    def get(name, default=None):
        return cfg.get(name, default)

    model = ImprovedEdgeGNN(
        feature_dim=cfg['n_features'],
        hidden_dim=get('hidden_dim', 256),
        num_classes=cfg['n_class'],
        lambda_reg=get('lambda_reg', get('beta')),
        reg_mode=get('reg_mode', get('egl_mode')),
        l0_method=final.get('l0_method', get('l0_method', 'hard-concrete')),
        edge_dim=cfg['edge_dim'],
        warmup_epochs=cfg['warmup_epochs'],
        graph_size_adaptation=cfg['graph_size_adaptation'],
        min_edges_per_node=cfg['min_edges_per_node'],
        dropout=cfg['dropout'],
        num_gnn_layers=cfg['num_gnn_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        use_attention_pooling=cfg['use_attention_pooling'],
        l0_gamma=get('l0_gamma', -0.1),
        l0_zeta=get('l0_zeta', 1.1),
        l0_beta=get('l0_beta', 0.66),
        baseline_ema=get('baseline_ema', 0.9),
        initial_temp=get('initial_temp', 5.0),
        lambda_density=get('lambda_density', 0.03),
        target_density=get('target_density', 0.30),
        ramp_epochs=get('ramp_epochs', 20),
        edge_logit_norm=get('edge_logit_norm', False),
    ).to(device)

    best = torch.load(os.path.join(run_dir, 'phase_models', 'overall_best_model.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    if 'temperature' in best and hasattr(model, 'temperature'):
        model.temperature = best['temperature']
    model.eval()
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)
    return model, cfg


def lens_slide_summary(model, features, adj, device, edge_thresh):
    """Deterministic inference gates for one slide -> per-patch / per-edge summaries."""
    import torch

    feats = features.unsqueeze(0).to(device)
    adjs = adj.unsqueeze(0).to(device)
    masks = torch.ones(1, feats.shape[1], device=device)
    labels = torch.zeros(1, dtype=torch.long, device=device)  # only feeds the loss
    with torch.no_grad():
        weighted_adj = model(feats, labels, adjs, masks)[3][0]

    cand = adjs[0] > 0
    # weighted_adj = A * Z, so divide by A on candidate edges to recover Z_inf.
    z = torch.where(cand, weighted_adj / adjs[0].clamp_min(1e-8), torch.zeros_like(weighted_adj))
    kept = cand & (z > edge_thresh)

    n = z.shape[0]
    upper = torch.triu(kept, diagonal=1).nonzero()
    n_cand_undirected = int(torch.triu(cand, diagonal=1).sum().item())
    return {
        'w': z.sum(dim=1).cpu().numpy(),
        'edges': np.sort((upper[:, 0] * n + upper[:, 1]).cpu().numpy()),
        'cand_degree': cand.sum(dim=1).cpu().numpy(),
        'kept_degree': kept.sum(dim=1).cpu().numpy(),
        'retention': len(upper) / max(n_cand_undirected, 1),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slides', required=True)
    parser.add_argument('--lens-run', action='append', default=[], metavar='SEED=DIR')
    parser.add_argument('--clam-scores', action='append', default=[], metavar='SEED=DIR')
    parser.add_argument('--data-root', default=None,
                        help="Graph root (features.pt, adj_s.pt, c_idx.txt per slide). "
                             "Default: the data_root saved in the first LENS checkpoint's config.")
    parser.add_argument('--roi-dir', default=None, help='Binary RoI masks <slide_id>.png (enables precision/recall)')
    parser.add_argument('--binarize', default='minmax:0.5',
                        help="Heatmap binarization: 'minmax:T' or 'top:F'. MUST match Table 3's rule.")
    parser.add_argument('--top-frac', type=float, default=0.10)
    parser.add_argument('--edge-thresh', type=float, default=0.0,
                        help='An undirected edge is retained when z_ij > this (0 = nonzero gate)')
    parser.add_argument('--coord-unit-px', type=float, default=512.0,
                        help='Level-0 pixels per c_idx.txt unit (512: SimCLR grid indices; 1: UNI2 pixel coords)')
    parser.add_argument('--patch-px', type=float, default=512.0, help='Patch edge in level-0 pixels')
    parser.add_argument('--mask-downsample', type=float, default=10.0,
                        help='Level-0 pixels per RoI-mask pixel')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slides = read_slide_ids(args.slides)
    lens_runs = parse_seed_pairs(args.lens_run)
    clam_dirs = parse_seed_pairs(args.clam_scores)
    if not lens_runs and not clam_dirs:
        raise SystemExit('Give at least one --lens-run or --clam-scores')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- per-seed, per-slide quantities --------------------------------------
    # info['LENS'][seed][slide] / info['CLAM'][seed][slide]
    info = {'LENS': {s: {} for s in lens_runs}, 'CLAM': {s: {} for s in clam_dirs}}
    data_root = args.data_root

    for seed, run_dir in lens_runs.items():
        model, cfg = build_model_from_run(run_dir, device)
        if data_root is None:
            data_root = cfg['data_root']
        for sid in slides:
            features = torch.load(os.path.join(data_root, sid, 'features.pt'), map_location='cpu')
            adj = torch.load(os.path.join(data_root, sid, 'adj_s.pt'), map_location='cpu')
            if adj.is_sparse:
                adj = adj.to_dense()
            info['LENS'][seed][sid] = lens_slide_summary(model, features, adj, device, args.edge_thresh)
            print(f"[LENS seed {seed}] {sid}: N={len(info['LENS'][seed][sid]['w'])} "
                  f"retention={info['LENS'][seed][sid]['retention']*100:.1f}%")
        del model
        torch.cuda.empty_cache()

    if data_root is None:
        raise SystemExit('--data-root is required when only --clam-scores is given')

    cand_degree = {}
    for sid in slides:
        adj = torch.load(os.path.join(data_root, sid, 'adj_s.pt'), map_location='cpu')
        if adj.is_sparse:
            adj = adj.to_dense()
        cand_degree[sid] = (adj > 0).sum(dim=1).numpy()

    for seed, d in clam_dirs.items():
        for sid in slides:
            w = np.load(os.path.join(d, f'{sid}.npy')).reshape(-1)
            if len(w) != len(cand_degree[sid]):
                raise ValueError(f"CLAM scores for {sid} (seed {seed}) have {len(w)} entries, "
                                 f"graph has {len(cand_degree[sid])} patches")
            info['CLAM'][seed][sid] = {'w': w.astype(float)}

    # ---- per-slide rows -------------------------------------------------------
    per_slide = []
    for method, by_seed in info.items():
        seeds = sorted(by_seed)
        for sid in slides:
            for a, b in itertools.combinations(seeds, 2):
                ia, ib = by_seed[a][sid], by_seed[b][sid]
                row = {
                    'method': method, 'kind': 'pair', 'slide': sid, 'seeds': f'{a}-{b}',
                    'spearman_w': spearman(ia['w'], ib['w']),
                    'top_jaccard': jaccard(top_fraction_set(ia['w'], args.top_frac),
                                           top_fraction_set(ib['w'], args.top_frac)),
                    'edge_jaccard': jaccard(ia['edges'], ib['edges']) if method == 'LENS' else float('nan'),
                }
                per_slide.append(row)
            for seed in seeds:
                it = by_seed[seed][sid]
                row = {
                    'method': method, 'kind': 'seed', 'slide': sid, 'seeds': str(seed),
                    'rho_w_cand_degree': spearman(it['w'], cand_degree[sid]),
                    'rho_w_kept_degree': spearman(it['w'], it['kept_degree']) if method == 'LENS' else float('nan'),
                    'retention': it['retention'] if method == 'LENS' else float('nan'),
                }
                if args.roi_dir is not None:
                    coords = read_coords(data_root, sid)
                    mask = _read_mask(os.path.join(args.roi_dir, f'{sid}.png'))
                    roi = roi_patch_labels(mask, coords, args.coord_unit_px, args.patch_px, args.mask_downsample)
                    pred = binarize(it['w'], args.binarize)
                    row['precision'], row['recall'] = precision_recall(pred, roi)
                    row['pred_empty'] = int(pred.sum() == 0)
                per_slide.append(row)

    with open(os.path.join(args.output_dir, 'per_slide.csv'), 'w', newline='') as f:
        fields = sorted({k for r in per_slide for k in r})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_slide)

    # ---- one table ------------------------------------------------------------
    def over_slides(method, kind, key, group):
        vals = [r[key] for r in per_slide
                if r['method'] == method and r['kind'] == kind and r['seeds'] == group and key in r]
        return float(np.nanmean(vals)) if vals and not np.all(np.isnan(vals)) else float('nan')

    header = ['Method', 'Spearman ρ(w)', f'Top-{int(args.top_frac*100)}% Jaccard', 'Edge Jaccard',
              'ρ(w, cand. degree)', 'ρ(w, kept degree)', 'Retention (%)', 'Precision (%)', 'Recall (%)']
    rows = []
    for method, by_seed in info.items():
        seeds = sorted(by_seed)
        if not seeds:
            continue
        pair_groups = [f'{a}-{b}' for a, b in itertools.combinations(seeds, 2)]
        seed_groups = [str(s) for s in seeds]

        def pair_stat(key):
            return mean_std([over_slides(method, 'pair', key, g) for g in pair_groups])

        def seed_stat(key):
            return mean_std([over_slides(method, 'seed', key, g) for g in seed_groups])

        rows.append([
            method,
            fmt(*pair_stat('spearman_w')),
            fmt(*pair_stat('top_jaccard')),
            fmt(*pair_stat('edge_jaccard')) if method == 'LENS' else '–',
            fmt(*seed_stat('rho_w_cand_degree')),
            fmt(*seed_stat('rho_w_kept_degree')) if method == 'LENS' else '–',
            fmt(*seed_stat('retention'), scale=100, digits=1) if method == 'LENS' else '–',
            fmt(*seed_stat('precision'), scale=100, digits=1) if args.roi_dir else 'n/a',
            fmt(*seed_stat('recall'), scale=100, digits=1) if args.roi_dir else 'n/a',
        ])

    with open(os.path.join(args.output_dir, 'seed_consistency_table.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(r) + ' |' for r in rows]
    note = (f"\nSlides: {len(slides)}. Seeds: LENS {sorted(lens_runs)}, CLAM {sorted(clam_dirs)}. "
            f"Pairwise columns: mean over slides per seed pair, then mean ± sample std over pairs. "
            f"Per-seed columns: mean over slides per seed, then mean ± sample std over seeds. "
            f"Edge retained iff z > {args.edge_thresh}; top set = top {args.top_frac:.0%} of patches with w > 0; "
            f"binarization {args.binarize}. 'n/a' = undefined (constant w or empty prediction).")
    table = '\n'.join(md) + '\n' + note
    with open(os.path.join(args.output_dir, 'seed_consistency_table.md'), 'w') as f:
        f.write(table + '\n')
    print('\n' + table)


def _read_mask(path):
    from PIL import Image
    return np.array(Image.open(path).convert('L'))


if __name__ == '__main__':
    main()
