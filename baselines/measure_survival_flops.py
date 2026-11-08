"""
Measure GFLOPs (forward-pass computational complexity) for LENS-Survival and
the GraphLSurv survival baseline, on one representative TCGA-BRCA slide.

Reuses baselines/FLOPcalcul.py's ComputationalProfiler unchanged -- it drives
every model the same way, model(sample_dict), where sample_dict has
'image'/'adj_s' keys (see ComputationalProfiler.create_representative_sample).
Neither survival model's own forward() matches that calling convention
directly (ImprovedEdgeGNNSurvival takes (node_feat, y_disc, censorship, adjs,
masks); GraphLSurv's adapted forward takes (model, node_feat, adjs, masks) as
a free function, not model.forward) -- the two small wrapper nn.Modules below
adapt each one to it, rather than changing the shared profiler.

Usage:
    python -m baselines.measure_survival_flops \\
        --data-root data/TCGA-BRCA-SURV/simclr_files \\
        --slide-id TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291 \\
        --n-features 2048 \\
        --output-dir results/survival_flops

Omit --slide-id to use the first slide found under --data-root.
"""

import argparse
import os

import torch
import torch.nn as nn

from baselines.FLOPcalcul import ComputationalProfiler
from baselines.graphlsurv_original_survival import build_graphlsurv, graphlsurv_forward
from model.LENS_survival import ImprovedEdgeGNNSurvival
from utils.config import add_common_args, validate_common_paths


class LensSurvivalFlopWrapper(nn.Module):
    """Adapts ImprovedEdgeGNNSurvival's (node_feat, y_disc, censorship, adjs,
    masks) forward to ComputationalProfiler's model(sample_dict) convention.
    y_disc/censorship are dummy (label values don't affect FLOP count -- only
    shapes do), constructed fresh each call so this stays device-agnostic."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample):
        node_feat = sample['image'].unsqueeze(0)     # [1, N, D]
        adjs = sample['adj_s'].unsqueeze(0)           # [1, N, N]
        device = node_feat.device
        n = node_feat.shape[1]
        masks = torch.ones(1, n, device=device)
        y_disc = torch.zeros(1, dtype=torch.long, device=device)
        censorship = torch.zeros(1, device=device)
        hazards, risk_score, loss, weighted_adj = self.model(node_feat, y_disc, censorship, adjs, masks)
        return hazards


class GraphLSurvFlopWrapper(nn.Module):
    """Adapts graphlsurv_forward(model, node_feat, adjs, masks) -- a free
    function, not model.forward -- to the same model(sample_dict) convention."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample):
        node_feat = sample['image'].unsqueeze(0)
        adjs = sample['adj_s'].unsqueeze(0)
        n = node_feat.shape[1]
        masks = torch.ones(1, n, device=node_feat.device)
        return graphlsurv_forward(self.model, node_feat, adjs, masks)


def find_first_slide(data_root):
    for name in sorted(os.listdir(data_root)):
        slide_dir = os.path.join(data_root, name)
        if os.path.isdir(slide_dir) and os.path.isfile(os.path.join(slide_dir, 'features.pt')):
            return name
    raise FileNotFoundError(f"No slide with a features.pt found under {data_root}")


def load_sample(data_root, slide_id, n_features):
    slide_dir = os.path.join(data_root, slide_id)
    features = torch.load(os.path.join(slide_dir, 'features.pt'), map_location='cpu').float()
    features = features[:, :n_features] if features.shape[1] >= n_features else features
    adj = torch.load(os.path.join(slide_dir, 'adj_s.pt'), map_location='cpu')
    if adj.is_sparse:
        adj = adj.to_dense()
    adj = adj.float()
    return {'image': features, 'adj_s': adj, 'id': slide_id, 'label': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, n_features_default=2048, output_dir_default='results/survival_flops',
                     require_train_list=False)
    parser.add_argument('--slide-id', type=str, default=None,
                         help='Slide directory name under --data-root to use as the representative '
                              'sample. Defaults to the first slide found.')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    validate_common_paths(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    slide_id = args.slide_id or find_first_slide(args.data_root)
    sample = load_sample(args.data_root, slide_id, args.n_features)
    print(f"Representative slide: {slide_id}")
    print(f"  Patches (nodes): {sample['image'].shape[0]:,}")
    print(f"  Feature dim: {sample['image'].shape[1]}")
    print(f"  Real edges: {int((sample['adj_s'] > 0).sum().item()):,}")

    lens_survival = ImprovedEdgeGNNSurvival(
        feature_dim=args.n_features, hidden_dim=args.hidden_dim, n_bins=args.n_bins,
    )
    # StatsTracker.print_stats defaults to True -- the [DEBUG] block it
    # triggers is meaningless here anyway (a freshly-constructed, untrained
    # model run once has nothing informative to show) and just repeats once
    # per warmup/profiling forward call. Silence it for a clean FLOP report.
    lens_survival.set_print_stats(False)
    graphlsurv = build_graphlsurv(feature_dim=args.n_features, hidden_dim=args.hidden_dim)

    models = {
        'LENS-Survival': LensSurvivalFlopWrapper(lens_survival),
        'GraphLSurv': GraphLSurvFlopWrapper(graphlsurv),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    profiler = ComputationalProfiler(device=str(device))
    results = profiler.measure_all_baselines(models, sample=sample)

    print("\n" + "=" * 40)
    print("GFLOPS SUMMARY (BRCA survival, 1 representative slide)")
    print("=" * 40)
    for name, gflops in results.items():
        print(f"  {name:<15}: {gflops:.3f} GFLOPs" if gflops is not None else f"  {name:<15}: FAILED (see error above)")

    profiler.save_results(os.path.join(args.output_dir, 'survival_gflops.json'))


if __name__ == '__main__':
    main()
