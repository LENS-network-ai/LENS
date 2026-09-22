"""
Standalone re-evaluation of a saved fold checkpoint's test-set edge
retention/weights, without retraining -- for when the retention-counting
logic in evaluate_test_with_edge_details (training/train_survival.py)
changes after a run has already completed (e.g. the >0 -> >0.5 "kept edge"
threshold fix), or any time you just want the per-slide detail/sparse
adjacencies regenerated from an existing checkpoint.

Reuses train_survival.py's build_model/make_loader/read_ids/
evaluate_test_with_edge_details directly rather than reimplementing them, so
this can never silently drift from what the actual training script does.
The checkpoint's own saved 'config' (the exact argparse Namespace used to
train it) is used to rebuild the model and dataloader, so no hyperparameters
need to be re-specified by hand.

Usage:
    python training/reevaluate_test_details.py \\
        --checkpoint results/survival_brca_single_run/run_.../fold5_best_model.pt \\
        --output-dir results/survival_brca_single_run/run_.../fold5_test_edge_details_v2
"""

import argparse
import os

import torch

from training.train_survival import build_model, make_loader, read_ids, evaluate_test_with_edge_details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Path to a fold{N}_best_model.pt saved by train_survival.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default=None,
                         help="Override the data root stored in the checkpoint's config, if paths moved")
    parser.add_argument('--test-list', default=None,
                         help="Override the test list stored in the checkpoint's config, if paths moved")
    args_cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    args = argparse.Namespace(**ckpt['config'])

    if args_cli.data_root is not None:
        args.data_root = args_cli.data_root
    if args_cli.test_list is not None:
        args.test_list = args_cli.test_list

    print(f"Loaded fold {ckpt.get('fold')} checkpoint | "
          f"val c-index={ckpt.get('val_c_index', float('nan')):.4f} | "
          f"test c-index={ckpt.get('test_c_index', float('nan')):.4f}")

    model = build_model(args, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, test_loader = make_loader(read_ids(args.test_list), args, shuffle=False)

    os.makedirs(args_cli.output_dir, exist_ok=True)
    evaluate_test_with_edge_details(model, test_loader, device, args.n_features, args_cli.output_dir)


if __name__ == '__main__':
    main()
