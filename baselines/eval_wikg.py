"""
baselines/eval_wikg.py

Evaluate a saved WiKG checkpoint (phase_models/overall_best_model.pth written by
the shared training loop) on a held-out list, without retraining. Reports
accuracy, macro one-vs-rest AUC, per-class recall and the confusion matrix, and
writes a per-slide prediction CSV.

Usage (from the repo root):
    python -m baselines.eval_wikg \\
        --checkpoint results/wikg_full/run_x/fold1/phase_models/overall_best_model.pth \\
        --data-root /path/to/uni2_graphs/CPTAC --test-list test.txt \\
        --n-features 1536 --n-class 3 --output-dir results/wikg_eval

The architecture flags must match the ones used for training (defaults are the
train_wikg.py defaults); the checkpoint is loaded with strict=True, so a
mismatch fails instead of silently evaluating a different model.
"""

import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from baselines.wikg import WiKGBaseline
from helper import collate, preparefeatureLabel
from utils.dataset import GraphDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--test-list', required=True)
    parser.add_argument('--n-features', type=int, default=1536)
    parser.add_argument('--n-class', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--agg-type', choices=['bi-interaction', 'gcn', 'sage'], default='bi-interaction')
    parser.add_argument('--pool', choices=['mean', 'max', 'attn'], default='mean')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WiKGBaseline(
        feature_dim=args.n_features, hidden_dim=args.hidden_dim, num_classes=args.n_class,
        topk=args.topk, dropout=args.dropout, agg_type=args.agg_type, pool=args.pool,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"Loaded {args.checkpoint} (saved epoch {ckpt.get('epoch')}, "
          f"val acc {ckpt.get('val_accuracy')})")

    with open(args.test_list) as f:
        ids = f.readlines()
    dataset = GraphDataset(args.data_root, ids, site='LUAD')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

    rows, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for sample in loader:
            node_feat, labels, adjs, masks = preparefeatureLabel(
                sample['image'], sample['label'], sample['adj_s'], n_features=args.n_features)
            logits = model(node_feat, labels, adjs, masks)[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for i, slide_id in enumerate(sample['id']):
                pred = int(probs[i].argmax())
                true = int(labels[i].item())
                rows.append({'slide_id': slide_id, 'true_label': true, 'predicted_label': pred,
                             'confidence': float(probs[i].max()), 'correct': int(pred == true)})
                probs_all.append(probs[i])
                labels_all.append(true)
            del node_feat, adjs, masks
            torch.cuda.empty_cache()

    y = np.array(labels_all)
    p = np.stack(probs_all)
    pred = p.argmax(axis=1)
    acc = float((pred == y).mean())
    auc = float(roc_auc_score(y, p, multi_class='ovr', average='macro'))
    cm = confusion_matrix(y, pred, labels=list(range(args.n_class)))
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    with open(os.path.join(args.output_dir, 'test_predictions.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = (f"n_test={len(y)}\naccuracy={acc:.4f}\nauc_macro_ovr={auc:.4f}\n"
               f"per_class_recall={np.round(recall, 4).tolist()}\nconfusion_matrix (rows=true):\n{cm}\n")
    with open(os.path.join(args.output_dir, 'test_summary.txt'), 'w') as f:
        f.write(summary)
    print(summary)


if __name__ == '__main__':
    main()
