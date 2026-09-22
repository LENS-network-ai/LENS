"""
Build LENS-format graphs (features.pt / adj_s.pt / c_idx.txt) directly from
already-extracted CLAM-style patch features + coordinates, instead of
re-running LENS's own tiling + feature-extraction pipeline from raw slides.

Expected input layout:
    <patch-features-dir>/<slide_id>.pt   # [N, D] or [M>=N, D] per-patch tensor
                                          # (see load_patch_features doc for
                                          # the extra-rows-beyond-N case)
    <patches-dir>/<slide_id>.h5          # dataset 'coords' -> [N, 2] (x, y)
                                          # pixel coords, same order as the
                                          # feature rows

Adjacency connects two patches iff they are spatial 8-connected grid
neighbors -- the same rule as preprocessing/graph_construction.py's
adj_matrix(). Rather than converting pixel coords to a tile grid via a
patch_size/downsample factor (fragile: CLAM records coords in level-0 pixel
space regardless of the extraction pyramid level, so the true center-to-
center spacing depends on a downsample factor this script can't verify), the
typical nearest-neighbor spacing is estimated directly from each slide's own
coordinates (median distance to each patch's nearest neighbor), and any pair
within ~1.5x that spacing is connected -- this covers the 4 orthogonal +
4 diagonal grid neighbors (at spacing and spacing*sqrt(2)) while excluding
farther patches (at 2x spacing), and is self-calibrating regardless of the
tiling level/downsample used to produce the coordinates.

The result is saved as a *sparse* adjacency (a grid graph has at most 8 edges
per node) -- utils/dataset.py's GraphDataset already densifies on load
(`if adj_s.is_sparse: adj_s.to_dense()`).

Usage:
    python preprocessing/build_survival_graphs.py \\
        --patch-features-dir /path/to/patch_features \\
        --patches-dir        /path/to/patches \\
        --output             data/TCGA-BRCA-SURV/simclr_files
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch


def load_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f:
            raise KeyError(
                f"No 'coords' dataset in {h5_path}. Found keys: {list(f.keys())}. "
                f"Adjust build_survival_graphs.py to match your CLAM h5 layout."
            )
        return f['coords'][:].astype(np.float64)


def build_sparse_radius_adjacency(coords, neighbor_factor=1.5):
    """
    8-connected-equivalent adjacency inferred directly from patch coordinates.

    The per-slide grid spacing is estimated as the median nearest-neighbor
    distance, then any pair of patches within `neighbor_factor` x that
    spacing is connected (covers orthogonal spacing and its sqrt(2) diagonal,
    excludes the next ring of patches at 2x spacing).
    """
    n = coords.shape[0]
    if n <= 1:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long),
                                        torch.zeros((0,), dtype=torch.float32), size=(n, n))

    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(-1))

    dist_no_self = dist.copy()
    np.fill_diagonal(dist_no_self, np.inf)
    nn_dist = dist_no_self.min(axis=1)
    spacing = np.median(nn_dist)

    threshold = spacing * neighbor_factor
    adj_mask = (dist > 0) & (dist <= threshold)
    np.fill_diagonal(adj_mask, False)

    src, dst = np.nonzero(adj_mask)
    if len(src) == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
    else:
        indices = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        values = torch.ones(len(src), dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce(), spacing


def load_patch_features(feat_path, n_patches):
    """
    Load a [N, D] per-patch feature tensor for a slide with `n_patches` real
    patches (from coords.h5).

    healnet's tasks.py (the script that produced these .pt files) allocates
    one [max_patches_in_cohort, 2048] buffer per run and reuses it across
    slides without resetting it, writing each slide's real features into
    rows [0:n_patches] only (see tasks.py's `features` step). So the tensor
    is already [max_patches, feature_dim] (patches-first, no transpose
    needed) and rows [0:n_patches] are this slide's genuine features in the
    same order as coords.h5; anything beyond that is stale leftover data
    from whichever other slide was processed earlier in that run and must
    be discarded.
    """
    features = torch.load(feat_path, map_location='cpu').float()
    if features.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(features.shape)}")

    if features.shape[0] < n_patches:
        raise ValueError(
            f"Feature tensor {tuple(features.shape)} has fewer rows than the "
            f"{n_patches} real patches; features.pt and patches.h5 are out of sync."
        )

    return features[:n_patches]


def build_one_slide(slide_id, feat_path, h5_path, out_dir):
    coords = load_coords(h5_path)
    features = load_patch_features(feat_path, coords.shape[0])

    adj_sparse, spacing = build_sparse_radius_adjacency(coords)

    slide_dir = os.path.join(out_dir, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    torch.save(features, os.path.join(slide_dir, 'features.pt'))
    torch.save(adj_sparse, os.path.join(slide_dir, 'adj_s.pt'))
    with open(os.path.join(slide_dir, 'c_idx.txt'), 'w') as f:
        for x, y in coords:
            f.write(f"{x:.0f}\t{y:.0f}\n")

    return features.shape[0], int(adj_sparse._nnz()), spacing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--patch-features-dir', required=True, help='Directory of <slide_id>.pt feature files')
    parser.add_argument('--patches-dir', required=True, help='Directory of <slide_id>.h5 coordinate files')
    parser.add_argument('--output', required=True, help='Output simclr_files-style directory')
    parser.add_argument('--slide-ids', default=None,
                         help='Optional text file of slide ids (one per line, no extension) to restrict to')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    allowed = None
    if args.slide_ids:
        with open(args.slide_ids) as f:
            allowed = {line.strip() for line in f if line.strip()}

    feat_files = sorted(glob.glob(os.path.join(args.patch_features_dir, '*.pt')))
    print(f"Found {len(feat_files)} patch-feature files in {args.patch_features_dir}")

    done, skipped, failed, zero_edge = 0, 0, 0, 0
    for feat_path in feat_files:
        slide_id = os.path.splitext(os.path.basename(feat_path))[0]
        if allowed is not None and slide_id not in allowed:
            continue

        out_dir = os.path.join(args.output, slide_id)
        if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, 'adj_s.pt')):
            skipped += 1
            continue

        h5_path = os.path.join(args.patches_dir, slide_id + '.h5')
        if not os.path.exists(h5_path):
            print(f"  [skip] {slide_id}: no matching {h5_path}")
            failed += 1
            continue

        try:
            n_nodes, n_edges, spacing = build_one_slide(slide_id, feat_path, h5_path, args.output)
            done += 1
            print(f"  [{done}] {slide_id}: {n_nodes} nodes, {n_edges} directed edges "
                  f"(estimated spacing={spacing:.1f}px)")
            if n_nodes > 1 and n_edges == 0:
                zero_edge += 1
                print(f"    [warn] 0 edges despite {n_nodes} nodes -- inspect this slide's coords.h5 directly.")
        except Exception as e:
            print(f"  [fail] {slide_id}: {e}")
            failed += 1

    print(f"\nDone. Built: {done}, already existed: {skipped}, failed: {failed}, zero-edge: {zero_edge}")


if __name__ == '__main__':
    main()
