"""
Train the ORIGINAL GraphLSurv (https://github.com/liupei101/GraphLSurv) on
the same data, splits, and evaluation protocol as training/train_survival.py,
for a direct, controlled comparison against LENS.

Same 5-fold CV on --train-list + one held-out fixed --test-list, same
concordance-index metric, same checkpoint-by-best-val-c-index selection.
GraphLSurv keeps its own original training objective (Cox partial
likelihood over a continuous risk score) rather than LENS's discrete-time
NLL, since a fair baseline comparison should let each method use the
loss it was actually designed and validated for -- concordance index is
loss-agnostic, so it's still an apples-to-apples comparison point.

IMPORTANT: the Cox partial-likelihood loss needs multiple samples per batch
to form a meaningful risk set (with batch_size=1 the loss is degenerate --
always exactly 0, no gradient). Use a real batch size here (default 16),
unlike train_survival.py's default of 1.

Usage:
    python training/train_graphlsurv_survival.py \\
        --data-root  data/TCGA-BRCA-SURV/simclr_files \\
        --train-list data/TCGA-BRCA-SURV/train_list_surv.txt \\
        --test-list  data/TCGA-BRCA-SURV/test_list_surv.txt \\
        --n-features 2048 --epochs 60 --n-folds 5 --output-dir results/graphlsurv_brca
"""

import argparse
import copy
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.graphlsurv_original_survival import (  # noqa: E402
    build_graphlsurv, graphlsurv_forward, to_cox_label,
    nlog_partial_likelihood, graphlsurv_concordance_index,
)
from utils.survival_dataset import SurvivalGraphDataset, collate_survival, prepare_survival_batch  # noqa: E402


def read_ids(path):
    with open(path) as f:
        return [line for line in f if line.strip()]


def get_patient_id(slide_id):
    """TCGA patient barcode = first 3 hyphen-separated segments of the slide
    barcode (e.g. 'TCGA-3C-AALI-01Z-...' -> 'TCGA-3C-AALI'). Some patients in
    this cohort have multiple slides that must stay in the same CV fold."""
    return '-'.join(slide_id.split('-')[:3])


def make_loader(ids, args, shuffle):
    dataset = SurvivalGraphDataset(root=args.data_root, ids=ids)
    return dataset, DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                collate_fn=collate_survival, drop_last=shuffle)


def run_epoch(model, loader, optimizer, device, n_features, train: bool, track_retention: bool = False):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_risk, all_y = [], []
    retentions = []

    for batch in loader:
        node_feat, adjs, masks, _, censorship, survival_months = prepare_survival_batch(
            batch['image'], batch['adj_s'], batch['y_disc'], batch['censorship'],
            batch['survival_months'], n_features=n_features, device=device,
        )
        y = to_cox_label(survival_months, censorship)

        with torch.set_grad_enabled(train):
            if track_retention:
                risk, retention = graphlsurv_forward(model, node_feat, adjs, masks, return_retention=True)
                retentions.append(retention)
            else:
                risk = graphlsurv_forward(model, node_feat, adjs, masks)
            loss = nlog_partial_likelihood(risk, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_risk.append(risk.detach().cpu())
        all_y.append(y.detach().cpu())

    all_risk = torch.cat(all_risk, dim=0)
    all_y = torch.cat(all_y, dim=0)
    c_index = graphlsurv_concordance_index(all_risk, all_y)
    metrics = {'loss': total_loss / max(1, n_batches), 'c_index': c_index}
    if track_retention and retentions:
        metrics['node_anchor_retention'] = float(np.mean(retentions))
    return metrics


def run_fold(fold, fold_train_ids, fold_val_ids, test_loader, args, device, output_dir):
    print(f"\n{'='*60}\nFOLD {fold}/{args.n_folds} "
          f"(train={len(fold_train_ids)}, val={len(fold_val_ids)})\n{'='*60}")

    _, train_loader = make_loader(fold_train_ids, args, shuffle=True)
    _, val_loader = make_loader(fold_val_ids, args, shuffle=False)

    model = build_graphlsurv(feature_dim=args.n_features, hidden_dim=args.hidden_dim,
                              dropout_ratio=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"graphlsurv_fold{fold}", group='graphlsurv-cv', job_type='train',
            tags=['graphlsurv', 'baseline', f'fold{fold}'],
            config={**vars(args), 'fold': fold, 'train_size': len(fold_train_ids), 'val_size': len(fold_val_ids)},
        )

    best_val_c_index = 0.0
    best_state = None

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.n_features, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args.n_features, train=False)

        print(f"Fold {fold} Epoch {epoch+1}/{args.epochs} | "
              f"train loss={train_metrics['loss']:.4f} c-index={train_metrics['c_index']:.4f} | "
              f"val loss={val_metrics['loss']:.4f} c-index={val_metrics['c_index']:.4f}")

        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'], 'train/c_index': train_metrics['c_index'],
                'val/loss': val_metrics['loss'], 'val/c_index': val_metrics['c_index'],
            })

        if val_metrics['c_index'] > best_val_c_index:
            best_val_c_index = val_metrics['c_index']
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, optimizer, device, args.n_features,
                              train=False, track_retention=True)

    print(f"Fold {fold} DONE | best val c-index={best_val_c_index:.4f} | "
          f"test c-index={test_metrics['c_index']:.4f} | "
          f"test node-anchor retention={test_metrics.get('node_anchor_retention', float('nan'))*100:.1f}% "
          f"(NOT the same quantity as LENS's edge retention -- see graphlsurv_forward docstring)")

    if args.use_wandb:
        wandb.log({
            'test/c_index': test_metrics['c_index'], 'test/loss': test_metrics['loss'],
            'test/node_anchor_retention_pct': test_metrics.get('node_anchor_retention', float('nan')) * 100,
        })
        wandb.summary['best_val_c_index'] = best_val_c_index
        wandb.summary['test_c_index'] = test_metrics['c_index']
        wandb.finish()

    torch.save({
        'fold': fold, 'model_state_dict': best_state if best_state is not None else model.state_dict(),
        'val_c_index': best_val_c_index, 'test_c_index': test_metrics['c_index'],
        'test_node_anchor_retention': test_metrics.get('node_anchor_retention'),
        'config': vars(args),
    }, os.path.join(output_dir, f'fold{fold}_best_model.pt'))

    return best_val_c_index, test_metrics['c_index']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--test-list', required=True)
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-features', type=int, default=2048)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16,
                         help='Cox loss needs >1 sample per batch to form a risk set; unlike LENS, do not use 1.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='lens-survival-brca')
    parser.add_argument('--wandb-entity', default=None)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ids = read_ids(args.train_list)
    test_ids = read_ids(args.test_list)
    _, test_loader = make_loader(test_ids, args, shuffle=False)
    print(f"CV pool: {len(train_ids)} samples | Fixed held-out test: {len(test_ids)} samples")

    censorships = [int(line.strip().split('\t')[2]) for line in train_ids]

    # Group by patient so a patient's slides can't be split across
    # fold-train/fold-val -- see the identical comment in train_survival.py
    # for the empirical confirmation that this cohort needs it (62/838
    # patients have >1 slide, both in the training pool).
    patient_ids = [get_patient_id(line.strip().split('\t')[0]) for line in train_ids]
    n_unique_patients = len(set(patient_ids))
    print(f"CV pool spans {n_unique_patients} unique patients across {len(train_ids)} slides "
          f"({len(train_ids) - n_unique_patients} patients contribute >1 slide)")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    val_c_indices, test_c_indices = [], []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(train_ids)), censorships, groups=patient_ids), start=1):
        fold_train_ids = [train_ids[i] for i in train_idx]
        fold_val_ids = [train_ids[i] for i in val_idx]

        train_patients = {get_patient_id(line.strip().split('\t')[0]) for line in fold_train_ids}
        val_patients = {get_patient_id(line.strip().split('\t')[0]) for line in fold_val_ids}
        overlap = train_patients & val_patients
        assert not overlap, f"Fold {fold}: {len(overlap)} patients leak across train/val: {overlap}"

        val_c, test_c = run_fold(fold, fold_train_ids, fold_val_ids, test_loader, args, device, output_dir)
        val_c_indices.append(val_c)
        test_c_indices.append(test_c)

    summary_lines = [
        "GraphLSurv 5-FOLD CV SUMMARY",
        "=" * 40,
        f"Val c-index   : {np.mean(val_c_indices):.4f} +/- {np.std(val_c_indices):.4f}  {['%.4f' % v for v in val_c_indices]}",
        f"Test c-index  : {np.mean(test_c_indices):.4f} +/- {np.std(test_c_indices):.4f}  {['%.4f' % v for v in test_c_indices]}",
    ]
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    with open(os.path.join(output_dir, 'cv_summary.txt'), 'w') as f:
        f.write(summary + "\n")


if __name__ == '__main__':
    main()
