"""
Parameters, FLOPs, latency and peak GPU memory of LENS+ and WiKG on the same
real slides, measured the same way (torch FlopCounterMode, CUDA timing, CUDA
peak-memory counters), for slides spanning the size distribution.

What this measures is the code as implemented: both models run dense N x N
operations here (LENS+ aggregates and attends with dense adjacency / attention
matrices; WiKG builds a dense head-tail similarity matrix), so measured costs
are not the analytic sparse-graph counts. Weights are randomly initialized:
compute and memory do not depend on the weight values.

Usage:
    python -m scripts.benchmark_vs_wikg \\
        --data-root /path/to/graphs --list /path/to/train_list.txt \\
        --n-features 1536 --n-slides 12 --output-dir results/benchmark_vs_wikg
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


def n_patches(data_root, slide_id):
    with open(os.path.join(data_root, slide_id, 'c_idx.txt')) as f:
        return sum(1 for _ in f)


def pick_slides(data_root, ids, n_slides):
    """Slides at evenly spaced size quantiles (5%..95%) of the cohort."""
    sizes = {s: n_patches(data_root, s) for s in ids}
    ordered = sorted(sizes, key=sizes.get)
    quantiles = np.linspace(0.05, 0.95, n_slides)
    chosen = [ordered[int(q * (len(ordered) - 1))] for q in quantiles]
    return [(s, sizes[s]) for s in chosen]


def load_slide(data_root, slide_id, device):
    feats = torch.load(os.path.join(data_root, slide_id, 'features.pt'), map_location='cpu')
    adj = torch.load(os.path.join(data_root, slide_id, 'adj_s.pt'), map_location='cpu')
    if adj.is_sparse:
        adj = adj.to_dense()
    n = feats.shape[0]
    return (feats.unsqueeze(0).to(device), adj.unsqueeze(0).to(device),
            torch.ones(1, n, device=device), torch.zeros(1, dtype=torch.long, device=device))


def build_models(n_features, device):
    from baselines.wikg import WiKGBaseline
    from model.LENS2 import ImprovedEdgeGNN

    # LENS+ as in the paper's appendix: hidden 256, 3 GNN layers, 8 heads, edge_dim 128.
    lens = ImprovedEdgeGNN(feature_dim=n_features, hidden_dim=256, num_classes=3, edge_dim=128,
                           num_gnn_layers=3, num_attention_heads=8, reg_mode='l0').to(device)
    # WiKG with the upstream defaults used for our baseline runs.
    wikg = WiKGBaseline(feature_dim=n_features, hidden_dim=512, num_classes=3, topk=6,
                        dropout=0.3, agg_type='bi-interaction', pool='mean').to(device)
    for m in (lens, wikg):
        if hasattr(m, 'set_print_stats'):
            m.set_print_stats(False)
    return {'LENS+': lens, 'WiKG': wikg}


def run_once(model, inputs, train):
    feats, adj, masks, labels = inputs
    model.train(train)
    if train:
        model.zero_grad(set_to_none=True)
        out = model(feats, labels, adj, masks)
        out[2].backward()
    else:
        with torch.no_grad():
            model(feats, labels, adj, masks)


def measure(model, inputs, train, device, repeats=3):
    """FLOPs (one pass), median latency and peak memory for one slide."""
    run_once(model, inputs, train)  # warm-up (kernel selection, allocator)
    torch.cuda.synchronize()

    with FlopCounterMode(display=False) as counter:
        run_once(model, inputs, train)
    flops = counter.get_total_flops()

    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once(model, inputs, train)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return flops / 1e9, float(np.median(times)), peak_gb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--list', required=True, help='Slide list (first token per line is the id)')
    parser.add_argument('--n-features', type=int, default=1536)
    parser.add_argument('--n-slides', type=int, default=12)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.list) as f:
        ids = [l.split()[0] for l in f if l.strip()]
    slides = pick_slides(args.data_root, ids, args.n_slides)
    models = build_models(args.n_features, device)

    params = {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}
    print({k: f"{v:,}" for k, v in params.items()})

    rows = []
    for slide_id, n in slides:
        inputs = load_slide(args.data_root, slide_id, device)
        for name, model in models.items():
            for mode, train in (('inference', False), ('train_step', True)):
                row = {'slide': slide_id, 'n_patches': n, 'model': name, 'mode': mode}
                try:
                    gflops, sec, peak = measure(model, inputs, train, device)
                    row.update(gflops=gflops, seconds=sec, peak_gb=peak)
                except torch.cuda.OutOfMemoryError:
                    row.update(gflops=float('nan'), seconds=float('nan'), peak_gb=float('nan'))
                    print(f"OOM: {name} {mode} N={n}")
                torch.cuda.empty_cache()
                rows.append(row)
                print(row)
        del inputs
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, 'per_slide.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"Parameters: " + ", ".join(f"{k} {v:,}" for k, v in params.items()), ""]
    for mode in ('inference', 'train_step'):
        lines += [f"### {mode}", "| N patches | LENS+ GFLOPs | WiKG GFLOPs | LENS+ s | WiKG s | LENS+ GB | WiKG GB |",
                  "|---|---|---|---|---|---|---|"]
        for slide_id, n in slides:
            def get(model, key):
                r = next(r for r in rows if r['slide'] == slide_id and r['model'] == model and r['mode'] == mode)
                return r[key]
            lines.append(f"| {n} | {get('LENS+', 'gflops'):.1f} | {get('WiKG', 'gflops'):.1f} | "
                         f"{get('LENS+', 'seconds'):.3f} | {get('WiKG', 'seconds'):.3f} | "
                         f"{get('LENS+', 'peak_gb'):.2f} | {get('WiKG', 'peak_gb'):.2f} |")
        lines.append("")
    table = "\n".join(lines)
    with open(os.path.join(args.output_dir, 'benchmark_table.md'), 'w') as f:
        f.write(table + "\n")
    print(table)


if __name__ == '__main__':
    main()
