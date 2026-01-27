#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from model.LENSwithGAT import LENSWithGAT
from helper import collate
from utils.dataset import GraphDataset  # Your existing dataset class
from utils.config import get_parser
from training.training_loop import train_and_evaluate
from utils.lr_scheduler import LR_Scheduler  # Your existing LR scheduler

def main():
    # Get the base parser and add GAT-specific parameters
    base_parser = get_parser()
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False,
                                  description='LENS with Graph Attention Networks')
    
    # Add GAT-specific arguments
    parser.add_argument('--gat-heads', type=int, default=8,
                        help='Number of attention heads in GAT layers')
    parser.add_argument('--gat-layers', type=int, default=2,
                        help='Number of GAT layers')
    parser.add_argument('--gat-hidden', type=int, default=64,
                        help='Hidden dimension for GAT layers')
    parser.add_argument('--use-l0-gat', action='store_true',
                        help='Use L0 regularization in GAT layers')
    parser.add_argument('--gat-bias-l0', type=float, default=-0.5,
                        help='Initial bias for L0 regularization in GAT')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"gat_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Load dataset
    with open(args.train_list, 'r') as f:
        all_ids = f.readlines()
    
    dataset = GraphDataset(root=args.data_root, ids=all_ids)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Create train/val split (for simplicity, using first 80% for training)
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create dataloaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = LENSWithGAT(
        feature_dim=args.n_features,
        hidden_dim=args.hidden_dims if isinstance(args.hidden_dims, int) else args.hidden_dims[0],
        num_classes=args.n_class,
        beta=args.beta,
        egl_mode=args.egl_mode,
        edge_dim=args.edge_dim,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        gat_hidden_dim=args.gat_hidden,
        use_l0_gat=args.use_l0_gat,
        gat_bias_l0=args.gat_bias_l0
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(
        mode='cos',
        base_lr=args.lr,
        num_epochs=args.epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=5
    )
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("LENS WITH GAT CONFIGURATION\n")
        f.write("=" * 30 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    # Train and evaluate
    print("\n" + "="*60)
    print("🔹 Training LENS Model with Graph Attention Networks")
    print("="*60)
    
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
        warmup_epochs=args.warmup_epochs
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_accuracy': results["val_accs"][-1],
        'best_val_accuracy': results["best_val_acc"],
        'best_epoch': results["best_epoch"],
    }, final_model_path)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f} (Epoch {results['best_epoch']})")
    print(f"Model saved to: {final_model_path}")
    
    # Also save the edge weight evolution plot
    if hasattr(model, 'plot_stats'):
        plot_path = os.path.join(output_dir, 'edge_weight_evolution.png')
        model.plot_stats(save_path=plot_path)
        print(f"Edge weight evolution plot saved to: {plot_path}")
    
    print(f"\n Detailed analysis and reports can be found in:")
    print(f"   {output_dir}")

if __name__ == "__main__":
    main()
