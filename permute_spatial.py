import os
import sys
import torch
import numpy as np
from scipy.spatial import KDTree
import argparse


def build_8nn_adjacency(coords, k=8):
    N = coords.shape[0]
    if N <= k:
        adj = torch.ones(N, N)
        return adj.to_sparse()
    
    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=k+1)
    
    rows = []
    cols = []
    for i in range(N):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    
    edges = set(zip(rows, cols))
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    
    indices_tensor = torch.LongTensor([rows, cols])
    values = torch.ones(len(rows))
    adj = torch.sparse_coo_tensor(indices_tensor, values, (N, N))
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True,
                        help='Original data directory with features.pt, adj_s.pt, c_idx.txt')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for permuted data')
    parser.add_argument('--train-list', type=str, required=True,
                        help='Train list file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    with open(args.train_list) as f:
        lines = f.readlines()
    slides = [l.strip().split()[0] for l in lines if l.strip()]
    
    print(f"Permuting {len(slides)} slides...")
    
    success = 0
    failed = 0
    
    for i, slide in enumerate(slides):
        src_dir = os.path.join(args.data_root, slide)
        dst_dir = os.path.join(args.output_dir, slide)
        os.makedirs(dst_dir, exist_ok=True)
        
        try:
            # Load original coordinates
            coord_path = os.path.join(src_dir, 'c_idx.txt')
            coords = np.loadtxt(coord_path)
            N = coords.shape[0]
            
            # Load original features (keep unchanged)
            feat_path = os.path.join(src_dir, 'features.pt')
            features = torch.load(feat_path, map_location='cpu')
            
            # Permute: shuffle which coordinates go with which features
            perm = np.random.permutation(N)
            coords_permuted = coords[perm]
            
            # Rebuild 8-NN from permuted coordinates
            adj_permuted = build_8nn_adjacency(coords_permuted, k=8)
            
            # Save features unchanged
            torch.save(features, os.path.join(dst_dir, 'features.pt'))
            
            # Save permuted adjacency
            torch.save(adj_permuted, os.path.join(dst_dir, 'adj_s.pt'))
            
            # Save permuted coordinates
            np.savetxt(os.path.join(dst_dir, 'c_idx.txt'), coords_permuted)
            
            success += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(slides)}] {slide}: N={N}")
        
        except Exception as e:
            print(f"  FAILED {slide}: {e}")
            failed += 1
    
    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
