"""
baselines/train_abmil.py

Training wrapper for ABMIL baseline — mirrors train_edge_gnn.py exactly:
  - Same function signature: train_abmil(dataset, train_idx, val_idx, args, ...)
  - Same DataLoader / collate_fn from helper.py
  - Same LR_Scheduler from utils/lr_scheduler.py
  - Same WandB logging structure and config dict
  - Same optimizer (AdamW), same output file conventions
  - Same calculate_class_weights / plot_metrics from training/analysis.py
  - Same train_and_evaluate loop from training/training_loop.py

Drop into baselines/ and call from main.py exactly like train_edge_gnn.
"""

import os
import gc
import torch
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import DataLoader

from training.analysis import plot_metrics, calculate_class_weights
from training.training_loop import train_and_evaluate
from helper import collate
from utils.lr_scheduler import LR_Scheduler
from baselines.abmil import ABMIL


def train_abmil(
    dataset,
    train_idx,
    val_idx,
    args,
    output_dir,
    use_wandb=True,
    wandb_project='lens-training',
    fold=None,
    run_name=None
):
    """
    Train the ABMIL baseline model with WandB logging.

    Mirrors train_edge_gnn() signature exactly so it can be called
    from main.py with the same arguments.

    Args:
        dataset      : Full dataset (same object passed to train_edge_gnn)
        train_idx    : Training indices
        val_idx      : Validation indices
        args         : Training arguments (argparse Namespace)
        output_dir   : Output directory for checkpoints and logs
        use_wandb    : Whether to use WandB logging
        wandb_project: WandB project name
        fold         : Fold number (for cross-validation)
        run_name     : Custom WandB run name
    Returns:
        dict with keys: results, model, method_name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # DataLoaders — identical to train_edge_gnn
    # -----------------------------------------------------------------------
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate
    )

    print("\n" + "="*60)
    print("Training ABMIL Baseline")
    print("="*60)

    # -----------------------------------------------------------------------
    # Class weights — same as train_edge_gnn
    # -----------------------------------------------------------------------
    class_weights = calculate_class_weights(dataset, train_idx).to(device)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    hidden_dim = getattr(args, 'hidden_dim', 256)
    dropout    = getattr(args, 'dropout', 0.25)

    model = ABMIL(
        feature_dim=512,
        hidden_dim=hidden_dim,
        num_classes=args.n_class,
        dropout=dropout
    ).to(device)

    print(f"\nModel Configuration:")
    print(f"  Method     : ABMIL (Gated Attention MIL)")
    print(f"  Feature dim: 512  (ResNet18-SimCLR)")
    print(f"  Hidden dim : {hidden_dim}")
    print(f"  Num classes: {args.n_class}")
    print(f"  Dropout    : {dropout}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Optimizer and scheduler — identical to train_edge_gnn
    # -----------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LR_Scheduler(
        mode='cos',
        base_lr=args.lr,
        num_epochs=args.epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=5
    )

    # -----------------------------------------------------------------------
    # WandB setup — mirrors train_edge_gnn config structure
    # -----------------------------------------------------------------------
    wandb_config = None
    if use_wandb:
        if run_name is None:
            fold_str = f"_fold{fold}" if fold is not None else ""
            run_name = f"abmil{fold_str}"

        config_dict = {
            # Model
            'method'     : 'abmil',
            'feature_dim': 512,
            'hidden_dim' : hidden_dim,
            'num_classes': args.n_class,
            'dropout'    : dropout,

            # Training
            'epochs'        : args.epochs,
            'warmup_epochs' : getattr(args, 'warmup_epochs', 5),
            'batch_size'    : args.batch_size,
            'learning_rate' : args.lr,
            'weight_decay'  : args.weight_decay,

            # Data
            'train_size': len(train_idx),
            'val_size'  : len(val_idx),
        }
        if fold is not None:
            config_dict['fold'] = fold

        wandb_config = {
            'project' : wandb_project,
            'name'    : run_name,
            'config'  : config_dict,
            'tags'    : ['abmil', f'fold{fold}' if fold is not None else 'single'],
            'group'   : 'abmil',
            'job_type': 'train',
        }

        print(f"\nWandB Logging: Enabled")
        print(f"  Project: {wandb_project}")
        print(f"  Run    : {run_name}")
    else:
        print(f"\nWandB Logging: Disabled")

    # -----------------------------------------------------------------------
    # Train — reuses the same training_loop as LENS
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")

    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        n_features=args.n_features,
        output_dir=output_dir,
        warmup_epochs=getattr(args, 'warmup_epochs', 5),
        wandb_config=wandb_config,
        l0_method='abmil'          # passed as method tag; training_loop uses
                                    # it only for logging, no l0 logic applied
    )

    # -----------------------------------------------------------------------
    # Save final checkpoint — same format as train_edge_gnn
    # -----------------------------------------------------------------------
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch'              : args.epochs,
        'model_state_dict'   : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy'       : results["val_accs"][-1],
        'best_val_accuracy'  : results["best_val_acc"],
        'best_epoch'         : results["best_epoch"],
        'method'             : 'abmil',
        'config'             : vars(args),
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # -----------------------------------------------------------------------
    # Plots and results — same as train_edge_gnn
    # -----------------------------------------------------------------------
    metrics_path = os.path.join(output_dir, 'training_metrics.png')
    plot_metrics(
        results["train_accs"],
        results["val_accs"],
        results["train_losses"],
        results["val_losses"],
        metrics_path,
        title="ABMIL Training Metrics",
        warmup_epochs=getattr(args, 'warmup_epochs', 5)
    )
    print(f"Training metrics saved: {metrics_path}")

    _save_training_results(args, output_dir, results)
    _print_training_summary(results)

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "results"    : results,
        "model"      : model,
        "method_name": "abmil"
    }


# ---------------------------------------------------------------------------
# Helpers — mirror save_training_results / print_training_summary
# ---------------------------------------------------------------------------

def _save_training_results(args, output_dir, results):
    """Save training results to text file — same format as train_edge_gnn."""
    path = os.path.join(output_dir, 'training_results.txt')
    with open(path, 'w') as f:
        f.write("ABMIL BASELINE TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Method         : ABMIL (Gated Attention MIL)\n")
        f.write(f"Feature dim    : 512  (ResNet18-SimCLR)\n")
        f.write(f"Hidden dim     : {getattr(args, 'hidden_dim', 256)}\n")
        f.write(f"Dropout        : {getattr(args, 'dropout', 0.25)}\n")
        f.write(f"Num classes    : {args.n_class}\n")

        f.write(f"\nTraining:\n")
        f.write(f"  Epochs       : {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Weight decay : {args.weight_decay}\n")
        f.write(f"  Batch size   : {args.batch_size}\n")

        f.write(f"\n{'='*60}\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Val Accuracy : {results['best_val_acc']:.4f}\n")
        f.write(f"Best Epoch        : {results['best_epoch']}\n")
        f.write(f"\nFinal Epoch:\n")
        f.write(f"  Train Accuracy  : {results['train_accs'][-1]:.4f}\n")
        f.write(f"  Val Accuracy    : {results['val_accs'][-1]:.4f}\n")
        f.write(f"  Train Loss      : {results['train_losses'][-1]:.4f}\n")
        f.write(f"  Val Loss        : {results['val_losses'][-1]:.4f}\n")

        f.write(f"\n{'='*60}\n")
        f.write("METHOD NOTES\n")
        f.write("-" * 40 + "\n")
        f.write("ABMIL uses no graph structure.\n")
        f.write("It pools patch features via gated attention (no edges).\n")
        f.write("This baseline isolates the contribution of the graph\n")
        f.write("topology learned by LENS from the patch features alone.\n")

    print(f"Detailed results saved: {path}")


def _print_training_summary(results):
    """Print training summary — same format as train_edge_gnn."""
    print("\n" + "="*60)
    print("ABMIL TRAINING SUMMARY")
    print("="*60)
    print(f"\nPerformance:")
    print(f"  Best Val Accuracy : {results['best_val_acc']:.4f}  (Epoch {results['best_epoch']})")
    print(f"\nFinal Epoch:")
    print(f"  Train Accuracy    : {results['train_accs'][-1]:.4f}")
    print(f"  Val Accuracy      : {results['val_accs'][-1]:.4f}")
    print(f"  Gap               : {abs(results['train_accs'][-1] - results['val_accs'][-1]):.4f}")
    print("\n" + "="*60)
