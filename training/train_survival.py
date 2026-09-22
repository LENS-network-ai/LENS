"""
Train LENS's edge-sparsification model on discrete-time survival prediction,
using 5-fold cross-validation on the training pool plus one held-out fixed
test set -- mirroring the mean/std-across-folds reporting convention used in
healnet/healnet/main.py, but with a genuinely fixed test set (never touched
by any fold's training or model selection) rather than a fresh random test
split drawn per fold.

Per fold:
    1. Split --train-list into a CV-train / CV-val partition (stratified by
       censorship, since ~87% of this cohort is censored).
    2. Train a fresh model; track the best checkpoint by *validation*
       c-index (never the test set) -- this is what training/train_survival.py
       got wrong in an earlier version: it selected checkpoints by peeking at
       the test set every epoch, which is test-set leakage.
    3. Evaluate that fold's best-val checkpoint on --test-list exactly once.

After all folds: report mean +/- std of both val c-index (how well model
selection worked) and test c-index (the actual generalization estimate),
and save every fold's best checkpoint plus a cv_summary.txt.

Kept as a standalone script (rather than folding into training/training.py)
so it can't regress the existing classification training path: it reuses the
shared edge-sparsification core (model/LENS_survival.py) but has its own
loss (discrete-time NLL), its own metric (concordance index), and its own
label format (utils/survival_dataset.SurvivalGraphDataset).

Usage:
    python training/train_survival.py \\
        --data-root  data/TCGA-BRCA-SURV/simclr_files \\
        --train-list data/TCGA-BRCA-SURV/train_list_surv.txt \\
        --test-list  data/TCGA-BRCA-SURV/test_list_surv.txt \\
        --n-features 2048 --epochs 30 --n-folds 5 --output-dir results/survival_brca
"""

import argparse
import copy
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from model.LENS_survival import ImprovedEdgeGNNSurvival
from utils.survival_dataset import SurvivalGraphDataset, collate_survival, prepare_survival_batch
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import concordance_index


def read_ids(path):
    with open(path) as f:
        return [line for line in f if line.strip()]


def get_patient_id(slide_id):
    """
    TCGA slide barcodes are '<patient_barcode>-<sample-vial>-<portion>-<slide>.<uuid>',
    e.g. 'TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-...' -> patient 'TCGA-3C-AALI'.
    The first 3 hyphen-separated segments are the patient barcode; some
    patients in this cohort have multiple slides (e.g. DX1 and DX2), which
    must stay together in the same CV fold to avoid patient-level leakage.
    """
    return '-'.join(slide_id.split('-')[:3])


def build_model(args, device):
    return ImprovedEdgeGNNSurvival(
        feature_dim=args.n_features, hidden_dim=args.hidden_dim, n_bins=args.n_bins,
        lambda_reg=args.lambda_reg, lambda_density=args.lambda_density, target_density=args.target_density,
        l0_method=args.l0_method, warmup_epochs=args.warmup_epochs, ramp_epochs=args.ramp_epochs,
        nll_alpha=args.nll_alpha,
    ).to(device)


def make_loader(ids, args, shuffle):
    dataset = SurvivalGraphDataset(root=args.data_root, ids=ids)
    return dataset, DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_survival)


def run_epoch(model, loader, optimizer, scheduler, device, n_features, epoch, train: bool, print_every: int = 50):
    model.train(train)
    if train:
        model.set_epoch(epoch)

    total_loss, n_batches = 0.0, 0
    risk_scores, event_times, censorships = [], [], []
    densities, lambda_effs, ambiguities = [], [], []

    for batch_idx, batch in enumerate(loader):
        # Only during training, only every `print_every` batches -- see the
        # debug block in LENS_survival.py's forward() for what gets printed
        # (raw logit and gate-value distributions, not just epoch averages).
        model.set_print_stats(train and print_every > 0 and batch_idx % print_every == 0)

        node_feat, adjs, masks, y_disc, censorship, survival_months = prepare_survival_batch(
            batch['image'], batch['adj_s'], batch['y_disc'], batch['censorship'],
            batch['survival_months'], n_features=n_features, device=device,
        )

        with torch.set_grad_enabled(train):
            hazards, risk_score, loss, _ = model(node_feat, y_disc, censorship, adjs, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler(optimizer, batch_idx, epoch, 0)

        # L0 schedule diagnostics -- only meaningful to track while training,
        # since that's what actually drives lambda/edge-retention over time.
        if train and hasattr(model.stats_tracker, 'current_density_history'):
            if model.stats_tracker.current_density_history:
                densities.append(model.stats_tracker.current_density_history[-1])
            if model.stats_tracker.lambda_eff_history:
                lambda_effs.append(model.stats_tracker.lambda_eff_history[-1])

        # Ambiguity, unlike density/lambda_eff, is collected on EVAL passes
        # too (not just train) -- the hyperparameter sweep's objective needs
        # this measured on the held-out val set at the selected checkpoint,
        # not just averaged over training batches.
        if hasattr(model.stats_tracker, 'ambiguity_history') and model.stats_tracker.ambiguity_history:
            ambiguities.append(model.stats_tracker.ambiguity_history[-1])

        total_loss += loss.item()
        n_batches += 1
        risk_scores.extend(risk_score.detach().cpu().tolist())
        event_times.extend(survival_months.detach().cpu().tolist())
        censorships.extend(censorship.detach().cpu().tolist())

    c_index = concordance_index(risk_scores, event_times, censorships)
    metrics = {'loss': total_loss / max(1, n_batches), 'c_index': c_index}
    if densities:
        metrics['density'] = float(np.mean(densities))          # kept-edges fraction
    if lambda_effs:
        metrics['lambda_eff'] = float(np.mean(lambda_effs))
    if ambiguities:
        metrics['ambiguity'] = float(np.mean(ambiguities))
    return metrics


def plot_lambda_density(epoch_lambdas, epoch_densities, target_density, output_path):
    """Plot the L0 effective-lambda schedule and kept-edges (%) over training epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(epoch_lambdas) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(epochs, epoch_lambdas, color='tab:blue', marker='o', markersize=3)
    ax1.set_ylabel('Effective lambda ($\\lambda_{eff}$)')
    ax1.set_title('L0 Regularization Schedule')
    ax1.grid(alpha=0.3)

    kept_pct = [d * 100 for d in epoch_densities]
    ax2.plot(epochs, kept_pct, color='tab:green', marker='o', markersize=3, label='Kept edges (%)')
    ax2.axhline(target_density * 100, color='tab:red', linestyle='--',
                label=f'Target density ({target_density*100:.0f}%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Kept edges (%)')
    ax2.set_title('Edge Retention Rate')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_test_with_edge_details(model, test_loader, device, n_features, save_dir):
    """
    Detailed per-slide test-time pass, for interpretability work rather than
    just an aggregate metric: for every test slide, records how many of its
    original edges were retained, the actual kept-edge weight distribution
    (not just a count), and saves the sparse pruned adjacency to disk (e.g.
    for visualize_heatmap.py later).
    """
    model.eval()
    model.set_print_stats(False)

    adj_dir = os.path.join(save_dir, 'pruned_adjacencies')
    os.makedirs(adj_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'edge_retention_summary.csv')

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            node_feat, adjs, masks, y_disc, censorship, survival_months = prepare_survival_batch(
                batch['image'], batch['adj_s'], batch['y_disc'], batch['censorship'],
                batch['survival_months'], n_features=n_features, device=device,
            )
            slide_ids = batch['id']

            _, risk_score, _, weighted_adj = model(node_feat, y_disc, censorship, adjs, masks)

            batch_size = node_feat.shape[0]
            for b in range(batch_size):
                n_nodes = int(masks[b].sum().item())
                adj_orig_b = adjs[b, :n_nodes, :n_nodes]
                adj_weighted_b = weighted_adj[b, :n_nodes, :n_nodes]

                orig_edge_mask = adj_orig_b > 0
                original_weights = adj_weighted_b[orig_edge_mask]
                num_original_edges = int(orig_edge_mask.sum().item())

                # l0_test's deterministic gate isn't binary -- it clips to
                # exactly 0 or exactly 1 only outside the [gamma, zeta]
                # logit range; in between it's a continuous value in (0, 1).
                # ">0" only catches exact zeros and silently counts a
                # barely-open edge (e.g. weight 0.05) as fully "kept", which
                # overstates retention. ">0.5" is the standard binarization
                # threshold for a gate value and is what "kept" should mean.
                kept_edge_mask = adj_weighted_b > 0.5
                num_kept_edges = int(kept_edge_mask.sum().item())
                retention_pct = 100.0 * num_kept_edges / max(1, num_original_edges)

                # Full weight-distribution breakdown among original edges,
                # to see whether weights are genuinely bimodal (near 0 or
                # near 1, hard-concrete working as intended) or clustered in
                # the middle (gates never pushed to their extremes).
                num_exact_zero = int((original_weights == 0).sum().item())
                num_weak = int(((original_weights > 0) & (original_weights <= 0.5)).sum().item())
                num_strong = num_kept_edges

                kept_weights = adj_weighted_b[kept_edge_mask]
                if kept_weights.numel() > 0:
                    w_mean = kept_weights.mean().item()
                    w_min = kept_weights.min().item()
                    w_max = kept_weights.max().item()
                    w_std = kept_weights.std().item() if kept_weights.numel() > 1 else 0.0
                else:
                    w_mean = w_min = w_max = w_std = 0.0

                slide_id = slide_ids[b]
                sparse_adj = adj_weighted_b.detach().cpu().to_sparse().coalesce()
                torch.save(sparse_adj, os.path.join(adj_dir, f'{slide_id}.pt'))

                rows.append({
                    'slide_id': slide_id,
                    'num_nodes': n_nodes,
                    'num_original_edges': num_original_edges,
                    'num_kept_edges': num_kept_edges,
                    'retention_pct': retention_pct,
                    'num_exact_zero': num_exact_zero,
                    'num_weak_0_to_0.5': num_weak,
                    'num_strong_above_0.5': num_strong,
                    'kept_weight_mean': w_mean,
                    'kept_weight_min': w_min,
                    'kept_weight_max': w_max,
                    'kept_weight_std': w_std,
                    'risk_score': risk_score[b].item(),
                    'survival_months': survival_months[b].item(),
                    'censorship': int(censorship[b].item()),
                })

    with open(summary_path, 'w', newline='') as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        retentions = [r['retention_pct'] for r in rows]
        print(f"\nTest-set edge retention detail saved to {save_dir}")
        print(f"  {len(rows)} slides | retention: mean={np.mean(retentions):.1f}% "
              f"min={np.min(retentions):.1f}% max={np.max(retentions):.1f}% "
              f"std={np.std(retentions):.1f}%")
        print(f"  Per-slide sparse pruned adjacencies: {adj_dir}/<slide_id>.pt")
        print(f"  Per-slide summary table: {summary_path}")

    return rows


def run_fold(fold, fold_train_ids, fold_val_ids, test_loader, args, device, output_dir):
    print(f"\n{'='*60}\nFOLD {fold}/{args.n_folds} "
          f"(train={len(fold_train_ids)}, val={len(fold_val_ids)})\n{'='*60}")

    _, train_loader = make_loader(fold_train_ids, args, shuffle=True)
    _, val_loader = make_loader(fold_val_ids, args, shuffle=False)

    model = build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(mode='cos', base_lr=args.lr, num_epochs=args.epochs,
                              iters_per_epoch=len(train_loader), warmup_epochs=args.warmup_epochs)

    if args.use_wandb:
        # No wandb.init here -- all folds share the single run opened once in
        # main(), with metrics prefixed fold{N}/... so they don't overwrite
        # each other in that one run's history. unwatch() first since each
        # fold gets a fresh model instance and watch() doesn't auto-replace
        # a previous target within the same run.
        wandb.unwatch()
        wandb.watch(model, log='all', log_freq=100)

    best_val_c_index = 0.0
    best_state = None
    best_epoch = 0
    epoch_lambdas, epoch_densities = [], []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device,
                                   args.n_features, epoch, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scheduler, device,
                                 args.n_features, epoch, train=False)

        lambda_eff = train_metrics.get('lambda_eff', 0.0)
        density = train_metrics.get('density', 0.0)
        epoch_lambdas.append(lambda_eff)
        epoch_densities.append(density)

        print(f"Fold {fold} Epoch {epoch+1}/{args.epochs} | "
              f"train loss={train_metrics['loss']:.4f} c-index={train_metrics['c_index']:.4f} | "
              f"val loss={val_metrics['loss']:.4f} c-index={val_metrics['c_index']:.4f} | "
              f"lambda_eff={lambda_eff:.6f} kept_edges={density*100:.1f}%")

        if args.use_wandb:
            # Prefixed per fold so all 5 folds' curves live in one shared
            # run's history without overwriting each other. Also log a
            # fold-local 'epoch' so you can set that as a chart's x-axis to
            # overlay folds -- the run's own auto-incrementing step keeps
            # climbing across the whole run (fold1: 0-N, fold2: N+1-2N, ...)
            # rather than resetting per fold.
            wandb.log({
                f'fold{fold}/epoch': epoch,
                f'fold{fold}/train/loss': train_metrics['loss'], f'fold{fold}/train/c_index': train_metrics['c_index'],
                f'fold{fold}/val/loss': val_metrics['loss'], f'fold{fold}/val/c_index': val_metrics['c_index'],
                f'fold{fold}/l0/lambda_eff': lambda_eff, f'fold{fold}/l0/kept_edges_pct': density * 100,
                f'fold{fold}/l0/target_density_pct': args.target_density * 100,
                f'fold{fold}/lr': optimizer.param_groups[0]['lr'],
            })

        if val_metrics['c_index'] > best_val_c_index:
            best_val_c_index = val_metrics['c_index']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Evaluate the best-val checkpoint on the held-out fixed test set exactly once.
    #
    # model.state_dict() only captures nn.Module parameters/buffers -- it
    # does NOT capture model.current_epoch / model.temperature / the
    # regularizer's lambda schedules, which are plain Python attributes.
    # Without restoring them here, they stay at whatever the LAST training
    # epoch left them (epoch=args.epochs-1, temperature at its annealed
    # floor), regardless of which epoch best_state actually came from --
    # silently evaluating those frozen weights under a temperature they were
    # never trained with. Confirmed to matter in practice: best_val_c_index
    # consistently peaks early (epoch ~15-25) before overfitting sets in,
    # while the L0 gate's exact-0/exact-1 saturation only develops much
    # later as temperature anneals to its floor -- so without this restore,
    # an early best-val checkpoint gets evaluated at a temperature/schedule
    # state it was never actually trained under.
    if best_state is not None:
        model.load_state_dict(best_state)
        model.set_epoch(best_epoch)
    print(f"Fold {fold}: best val c-index={best_val_c_index:.4f} at epoch {best_epoch+1}/{args.epochs} "
          f"(temperature restored to {model.temperature:.3f} for test evaluation)")

    # Re-evaluate the val set at the restored checkpoint/temperature (rather
    # than reusing the value captured mid-loop) so val_loss and ambiguity are
    # measured under the exact same state test_metrics below is -- both feed
    # the sweep's combined objective, so they need to be internally
    # consistent with each other and with what test/interpretability below
    # actually sees.
    final_val_metrics = run_epoch(model, val_loader, None, None, device, args.n_features, epoch=0, train=False)
    val_ambiguity = final_val_metrics.get('ambiguity', 0.0)
    # Sweep objective (see main()'s docstring/CLI help): a plain sum, not
    # normalized -- val_loss is O(0.5-2) and ambiguity is bounded in [0,
    # 0.25], so ambiguity acts as a tie-breaker/soft constraint among
    # settings with similar val_loss rather than dominating the objective.
    # Re-weight here if a sweep shows one term swamping the other.
    combined_objective = final_val_metrics['loss'] + val_ambiguity

    test_metrics = run_epoch(model, test_loader, None, None, device, args.n_features, epoch=0, train=False)

    print(f"Fold {fold} DONE | best val c-index={best_val_c_index:.4f} | "
          f"val loss={final_val_metrics['loss']:.4f} val ambiguity={val_ambiguity:.4f} "
          f"combined_objective={combined_objective:.4f} | "
          f"test c-index={test_metrics['c_index']:.4f} | "
          f"final kept_edges={epoch_densities[-1]*100:.1f}%")

    if args.use_wandb:
        wandb.log({f'fold{fold}/test/c_index': test_metrics['c_index'],
                    f'fold{fold}/test/loss': test_metrics['loss'],
                    f'fold{fold}/best_val_c_index': best_val_c_index,
                    f'fold{fold}/val/ambiguity': val_ambiguity,
                    f'fold{fold}/combined_objective': combined_objective})
        wandb.summary[f'fold{fold}_best_val_c_index'] = best_val_c_index
        wandb.summary[f'fold{fold}_best_epoch'] = best_epoch
        wandb.summary[f'fold{fold}_test_c_index'] = test_metrics['c_index']
        wandb.summary[f'fold{fold}_val_ambiguity'] = val_ambiguity
        wandb.summary[f'fold{fold}_combined_objective'] = combined_objective
        # No wandb.finish() here -- the run stays open across all folds,
        # closed once at the end of main() after the cross-fold summary.

    with open(os.path.join(output_dir, f'fold{fold}_lambda_density.csv'), 'w') as f:
        f.write("epoch,lambda_eff,kept_edges_pct\n")
        for e, (lam, dens) in enumerate(zip(epoch_lambdas, epoch_densities), start=1):
            f.write(f"{e},{lam},{dens*100}\n")

    plot_lambda_density(epoch_lambdas, epoch_densities, args.target_density,
                         os.path.join(output_dir, f'fold{fold}_lambda_density.png'))

    # Detailed per-slide edge retention/weights + saved sparse pruned
    # adjacencies for the fixed test set, using this fold's best-val
    # checkpoint (already loaded into `model` above).
    test_detail_dir = os.path.join(output_dir, f'fold{fold}_test_edge_details')
    evaluate_test_with_edge_details(model, test_loader, device, args.n_features, test_detail_dir)

    torch.save({
        'fold': fold,
        'model_state_dict': best_state if best_state is not None else model.state_dict(),
        'best_epoch': best_epoch,
        'val_c_index': best_val_c_index,
        'test_c_index': test_metrics['c_index'],
        'lambda_history': epoch_lambdas,
        'density_history': epoch_densities,
        'val_ambiguity': val_ambiguity,
        'combined_objective': combined_objective,
        'config': vars(args),
    }, os.path.join(output_dir, f'fold{fold}_best_model.pt'))

    return best_val_c_index, test_metrics['c_index'], combined_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--test-list', required=True)
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-features', type=int, default=2048)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    # Matches the classification pipeline's validated schedule (README's
    # standard-training example): a faster ramp than this let lambda_eff
    # outrun the adaptive-lambda feedback and collapse density well past
    # --target-density before it could correct (observed: density fell to
    # 12.6% by epoch 9 with warmup=5/ramp=10, loss diverging).
    parser.add_argument('--warmup-epochs', type=int, default=15)
    parser.add_argument('--ramp-epochs', type=int, default=20)

    # model/LENS_survival.py now sums the L0 penalty over real edges only
    # (masked, not averaged, and not over the full dense N^2 tensor) -- see
    # that file's comment for why an averaged version silently diluted the
    # per-edge gradient by ~E (edge count) and made lambda_reg increases
    # nearly ineffective. This masked-sum scale is close to (a bit smaller
    # than, since it excludes non-edges) what the classification pipeline's
    # own default targets, so start near its documented default (0.01) --
    # still a reasoned starting point, not empirically tuned for this
    # dataset; watch kept_edges_pct over the first ~15 epochs and move by
    # 10x in whichever direction it's clearly headed away from
    # --target-density.
    parser.add_argument('--lambda-reg', type=float, default=0.01)
    parser.add_argument('--lambda-density', type=float, default=0.03)
    parser.add_argument('--target-density', type=float, default=0.30)
    parser.add_argument('--l0-method', default='hard-concrete', choices=['hard-concrete', 'arm'])
    parser.add_argument('--nll-alpha', type=float, default=0.0)

    parser.add_argument('--use-wandb', action='store_true',
                         help='Log the whole 5-fold CV run as a single WandB run (metrics prefixed fold{N}/...)')
    parser.add_argument('--wandb-project', default='lens-survival')
    parser.add_argument('--wandb-entity', default=None, help='WandB team/entity (default: your default entity)')

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ids = read_ids(args.train_list)
    test_ids = read_ids(args.test_list)
    _, test_loader = make_loader(test_ids, args, shuffle=False)
    print(f"CV pool: {len(train_ids)} samples | Fixed held-out test: {len(test_ids)} samples")

    # Stratify folds by censorship (the label field most likely to be
    # imbalanced -- ~87% censored in this cohort) rather than the discrete
    # time bin, so every fold sees a representative mix of events/censoring.
    censorships = [int(line.strip().split('\t')[2]) for line in train_ids]

    # Group by patient so a patient's slides can never be split across
    # fold-train and fold-val within the same fold -- confirmed 62/838
    # patients in this cohort have >1 slide, both landing in the training
    # pool (e.g. TCGA-3C-AALI has DX1 and DX2). Plain StratifiedKFold splits
    # by row/slide with no patient awareness, so it can (and, checked
    # empirically, does) put one of a patient's slides in train and the
    # other in val for the same fold -- real patient-level leakage.
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

    if args.use_wandb:
        # One run for the whole 5-fold experiment (not one run per fold):
        # run_fold logs each fold's metrics under a fold{N}/... prefix into
        # this same run instead of calling wandb.init/finish itself.
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"survival_cv_{timestamp}", job_type='train',
            tags=['survival', args.l0_method, f'{args.n_folds}fold-cv'],
            config=vars(args),
        )

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    val_c_indices, test_c_indices, combined_objectives = [], [], []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(train_ids)), censorships, groups=patient_ids), start=1):
        fold_train_ids = [train_ids[i] for i in train_idx]
        fold_val_ids = [train_ids[i] for i in val_idx]

        # Sanity-check every fold, every run: no patient may appear on both
        # sides. This must never fail; if it does, something upstream (e.g.
        # get_patient_id's barcode parsing) is wrong.
        train_patients = {get_patient_id(line.strip().split('\t')[0]) for line in fold_train_ids}
        val_patients = {get_patient_id(line.strip().split('\t')[0]) for line in fold_val_ids}
        overlap = train_patients & val_patients
        assert not overlap, f"Fold {fold}: {len(overlap)} patients leak across train/val: {overlap}"

        val_c, test_c, combined_obj = run_fold(fold, fold_train_ids, fold_val_ids, test_loader, args, device, output_dir)
        val_c_indices.append(val_c)
        test_c_indices.append(test_c)
        combined_objectives.append(combined_obj)

    summary_lines = [
        "5-FOLD CV SUMMARY",
        "=" * 40,
        f"Val c-index   : {np.mean(val_c_indices):.4f} +/- {np.std(val_c_indices):.4f}  {['%.4f' % v for v in val_c_indices]}",
        f"Test c-index  : {np.mean(test_c_indices):.4f} +/- {np.std(test_c_indices):.4f}  {['%.4f' % v for v in test_c_indices]}",
        f"Combined objective (val_loss + ambiguity): {np.mean(combined_objectives):.4f} +/- {np.std(combined_objectives):.4f}",
        f"Best fold (by test c-index): {int(np.argmax(test_c_indices)) + 1}",
    ]
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    with open(os.path.join(output_dir, 'cv_summary.txt'), 'w') as f:
        f.write(summary + "\n")

    if args.use_wandb:
        wandb.summary['mean_val_c_index'] = np.mean(val_c_indices)
        wandb.summary['std_val_c_index'] = np.std(val_c_indices)
        wandb.summary['mean_test_c_index'] = np.mean(test_c_indices)
        wandb.summary['std_test_c_index'] = np.std(test_c_indices)
        wandb.summary['best_fold'] = int(np.argmax(test_c_indices)) + 1
        # Sweep target -- name matches training/sweep_survival.yaml's
        # metric.name so `wandb agent` can read this back per run.
        wandb.summary['combined_objective'] = np.mean(combined_objectives)
        wandb.finish()


if __name__ == '__main__':
    main()
