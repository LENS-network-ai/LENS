"""
baselines/train_wikg.py

WiKG baseline under the exact LENS classification protocol: same GraphDataset,
same cross-validation split as main.py (patient-grouped by default; same seed -> same folds), same shared
train_and_evaluate loop (AdamW + cosine schedule, best-validation checkpoint,
held-out --test-list evaluated once per fold), same per-sample prediction CSVs.

WiKG constructs its own graph from features (it never sees the 8-NN
adjacency); see baselines/wikg.py.

Setup:
    bash baselines/setup_wikg_baseline.sh

Usage (from the repo root, as a module so `baselines.*` and `utils.*` resolve):
    python -m baselines.train_wikg \\
        --data-root <graphs dir with features.pt per slide> \\
        --train-list <trainVal list> --test-list <held-out list> \\
        --n-features 1536 --n-class 3 --epochs 80 --output-dir results/wikg_cptac
"""

import argparse
import gc
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from baselines.wikg import WiKGBaseline
from helper import collate
from training.training_loop import train_and_evaluate
from utils.config import add_common_args, validate_common_paths
from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from utils.splits import make_folds


def train_wikg(dataset, train_idx, val_idx, args, output_dir, test_dataset=None,
               use_wandb=True, wandb_project='lens-training', fold=None, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate)

    model = WiKGBaseline(
        feature_dim=args.n_features, hidden_dim=args.hidden_dim,
        num_classes=args.n_class, topk=args.topk, dropout=args.dropout,
        agg_type=args.agg_type, pool=args.pool,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nWiKG: feature_dim={args.n_features} hidden_dim={args.hidden_dim} "
          f"topk={args.topk} agg={args.agg_type} pool={args.pool} "
          f"dropout={args.dropout} params={n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(mode='cos', base_lr=args.lr, num_epochs=args.epochs,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=args.warmup_epochs)

    wandb_config = None
    if use_wandb:
        fold_str = f"_fold{fold}" if fold is not None else ""
        wandb_config = {
            'project': wandb_project,
            'name': run_name or f"wikg{fold_str}_seed{args.seed}",
            'config': {
                'method': 'wikg', 'feature_dim': args.n_features,
                'hidden_dim': args.hidden_dim, 'topk': args.topk,
                'agg_type': args.agg_type, 'pool': args.pool,
                'dropout': args.dropout, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'learning_rate': args.lr,
                'weight_decay': args.weight_decay, 'seed': args.seed,
                'num_classes': args.n_class, 'train_size': len(train_idx),
                'val_size': len(val_idx), 'fold': fold, 'params': n_params,
            },
            'tags': ['wikg', f'fold{fold}' if fold is not None else 'single'],
            'group': 'wikg',
            'job_type': 'train',
        }

    results = train_and_evaluate(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        num_epochs=args.epochs, n_features=args.n_features, output_dir=output_dir,
        warmup_epochs=args.warmup_epochs, wandb_config=wandb_config,
        l0_method='wikg', test_loader=test_loader,
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_accuracy': results['best_val_acc'],
        'best_epoch': results['best_epoch'],
        'method': 'wikg',
        'config': vars(args),
    }, os.path.join(output_dir, 'final_model.pt'))

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return {"results": results, "method_name": "wikg"}


def _read_ids(path):
    with open(path, 'r') as f:
        return f.readlines()


def _labels_from_ids(dataset):
    # Same parsing as GraphDataset.__getitem__, without loading every
    # features.pt/adj_s.pt just to read labels. Order and values are
    # identical to main.py's [dataset[i]['label'] ...], so the folds match.
    labels = []
    for line in dataset.ids:
        info = line.replace('\n', '')
        parts = info.split('\t') if '\t' in info else info.split()
        label = parts[1].strip().lower()
        labels.append(dataset.classdict[label] if dataset.classdict is not None else int(label))
    return labels


def get_parser():
    parser = argparse.ArgumentParser(description='WiKG baseline (official model) under the LENS protocol')
    add_common_args(parser, n_features_default=1536)

    parser.add_argument('--n-class', type=int, default=3, help='Number of classes')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Stop after this many folds (same split as a full run)')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4, help='Upstream WiKG default: 1e-4')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Upstream WiKG default: 1e-5')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='LR warm-up epochs (shared scheduler)')

    parser.add_argument('--hidden-dim', type=int, default=512, help='Upstream WiKG default: 512')
    parser.add_argument('--topk', type=int, default=6, help='Neighbours per patch (upstream default: 6)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Upstream WiKG default: 0.3')
    parser.add_argument('--agg-type', choices=['bi-interaction', 'gcn', 'sage'], default='bi-interaction')
    parser.add_argument('--pool', choices=['mean', 'max', 'attn'], default='mean',
                        help="Readout (upstream train.py uses 'mean')")

    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='lens-training')
    return parser


def main():
    args = get_parser().parse_args()
    validate_common_paths(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = GraphDataset(root=args.data_root, ids=_read_ids(args.train_list))
    all_labels = _labels_from_ids(dataset)

    test_dataset = None
    if args.test_list:
        test_dataset = GraphDataset(root=args.data_root, ids=_read_ids(args.test_list))
        print(f"Held-out test set: {len(test_dataset)} samples from {args.test_list}")

    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    folds = make_folds(dataset.ids, all_labels, args.n_folds,
                       args.seed if args.split_seed is None else args.split_seed,
                       args.group_by_patient)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\nFold {fold + 1}/{args.n_folds}")
        fold_dir = os.path.join(args.output_dir, f"fold{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        out = train_wikg(dataset, train_idx, val_idx, args, fold_dir,
                         test_dataset=test_dataset, use_wandb=not args.no_wandb,
                         wandb_project=args.wandb_project, fold=fold + 1)
        fold_results.append(out['results'])
        if args.max_folds is not None and (fold + 1) >= args.max_folds:
            print(f"\n--max-folds={args.max_folds} reached, stopping after fold {fold + 1}.")
            break

    val_accs = [r['best_val_acc'] for r in fold_results]
    test_accs = [r['test_accuracy'] for r in fold_results if 'test_accuracy' in r]
    lines = [f"WiKG | seed={args.seed} | folds={len(fold_results)}",
             f"val  acc: {np.mean(val_accs) * 100:.2f} +/- {np.std(val_accs) * 100:.2f}"]
    if test_accs:
        lines.append(f"test acc: {np.mean(test_accs) * 100:.2f} +/- {np.std(test_accs) * 100:.2f}")
    summary = "\n".join(lines)
    print("\n" + "=" * 60 + "\n" + summary + "\n" + "=" * 60)
    with open(os.path.join(args.output_dir, 'cv_summary.txt'), 'w') as f:
        f.write(summary + "\n")


if __name__ == '__main__':
    main()
