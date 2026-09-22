"""
Convert UNI2-h pre-extracted h5 features to LENS pipeline format.
For each slide: h5 -> features.pt (N x 1536) + adj_s.pt (N x N sparse 8-NN adjacency)
"""

import os
import sys
import glob
import h5py
import torch
import numpy as np
from scipy.spatial import KDTree
from collections import Counter


def build_8nn_adjacency(coords, k=8):
    """
    Build 8-nearest-neighbor adjacency matrix from patch coordinates.
    Uses spatial coordinates (not feature space).
    
    Args:
        coords: (N, 2) array of patch coordinates
        k: number of nearest neighbors (default: 8)
    
    Returns:
        adj: (N, N) sparse tensor, symmetric binary adjacency
    """
    N = coords.shape[0]
    
    if N <= k:
        # If fewer patches than k, connect all
        adj = torch.ones(N, N)
        return adj.to_sparse()
    
    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(coords)
    
    # Query k+1 neighbors (includes self)
    distances, indices = tree.query(coords, k=k+1)
    
    # Build adjacency (skip self-loop at index 0)
    rows = []
    cols = []
    for i in range(N):
        for j_idx in range(1, k+1):  # skip self
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
            # Make symmetric
            rows.append(j)
            cols.append(i)
    
    # Remove duplicates by converting to set
    edges = set(zip(rows, cols))
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    
    # Create sparse tensor
    indices_tensor = torch.LongTensor([rows, cols])
    values = torch.ones(len(rows))
    adj = torch.sparse_coo_tensor(indices_tensor, values, (N, N))
    
    return adj


def convert_h5_to_lens(h5_path, output_dir, slide_name):
    """
    Convert a single UNI2-h h5 file to LENS format.
    
    Args:
        h5_path: path to h5 file
        output_dir: base output directory
        slide_name: slide identifier (folder name)
    """
    # Create output folder
    slide_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_dir, exist_ok=True)
    
    # Check if already converted
    feat_path = os.path.join(slide_dir, 'features.pt')
    adj_path = os.path.join(slide_dir, 'adj_s.pt')
    if os.path.exists(feat_path) and os.path.exists(adj_path):
        return True, "already exists"
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Extract features: (1, N, 1536) -> (N, 1536)
            features = torch.from_numpy(f['features'][0]).float()
            
            # Extract coordinates: (1, N, 2) -> (N, 2)
            coords = f['coords'][0]  # numpy array (N, 2)
        
        N = features.shape[0]
        
        # Build 8-NN adjacency from spatial coordinates
        adj = build_8nn_adjacency(coords, k=8)
        
        # Save features.pt
        torch.save(features, feat_path)
        
        # Save adj_s.pt
        torch.save(adj, adj_path)
        
        # Save coordinates for reference
        coord_path = os.path.join(slide_dir, 'c_idx.txt')
        with open(coord_path, 'w') as f:
            for i in range(N):
                f.write(f"{coords[i, 0]}\t{coords[i, 1]}\n")
        
        return True, f"N={N}, features={features.shape}, edges={adj._nnz()}"
    
    except Exception as e:
        return False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-dir', type=str, required=True,
                        help='Directory containing h5 files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for LENS format')
    parser.add_argument('--train-list', type=str, required=True,
                        help='Train list file to filter which slides to convert')
    args = parser.parse_args()
    
    # Load slide names from train list
    with open(args.train_list) as f:
        lines = f.readlines()
    
    slides = [l.strip().split()[0] for l in lines if l.strip()]
    labels = [l.strip().split()[1] for l in lines if l.strip()]
    
    print(f"Slides to convert: {len(slides)}")
    print(f"Class distribution: {Counter(labels)}")
    
    # Find h5 files
    h5_files = glob.glob(os.path.join(args.h5_dir, '*.h5'))
    h5_map = {os.path.basename(f).replace('.h5', ''): f for f in h5_files}
    
    print(f"H5 files available: {len(h5_map)}")
    
    # Convert
    success = 0
    failed = 0
    skipped = 0
    
    for i, slide in enumerate(slides):
        if slide not in h5_map:
            print(f"  [{i+1}/{len(slides)}] MISSING: {slide}")
            failed += 1
            continue
        
        ok, msg = convert_h5_to_lens(h5_map[slide], args.output_dir, slide)
        
        if ok:
            if "already exists" in msg:
                skipped += 1
            else:
                success += 1
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(slides)}] {slide}: {msg}")
        else:
            print(f"  [{i+1}/{len(slides)}] FAILED {slide}: {msg}")
            failed += 1
    
    print(f"\nDone! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
