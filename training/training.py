# training/train_edge_gnn.py
"""
Training wrapper for LENS model with WandB logging and ARM support
"""

import os
import torch
import torch.optim as optim
import gc
import numpy as np
import wandb
from torch.utils.data import DataLoader

from training.analysis import plot_metrics, calculate_class_weights
from training.training_loop import train_and_evaluate
from helper import collate
from utils.lr_scheduler import LR_Scheduler


def train_edge_gnn(dataset, train_idx, val_idx, args, output_dir, 
                   use_wandb=True, wandb_project='lens-training',
                   fold=None, run_name=None):
    """
    Train the ImprovedEdgeGNN (LENS) model with WandB logging
    
    Args:
        dataset: Full dataset
        train_idx: Training indices
        val_idx: Validation indices
        args: Training arguments
        output_dir: Output directory
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        fold: Fold number (for cross-validation)
        run_name: Custom run name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get L0 method
    l0_method = args.l0_method if hasattr(args, 'l0_method') else 'hard-concrete'
    
    # Create dataloaders
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
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
    print(f"Training LENS Model ({l0_method.upper()})")
    print("="*60)
    
    # Calculate class weights
    class_weights = calculate_class_weights(dataset, train_idx).to(device)
    
    # Import model
    from model.LENS2 import ImprovedEdgeGNN
    
    # Get regularization parameters
    lambda_reg = args.lambda_reg if hasattr(args, 'lambda_reg') else args.beta
    reg_mode = args.reg_mode if hasattr(args, 'reg_mode') else args.egl_mode
    
    # L0 parameters
    l0_gamma = args.l0_gamma if hasattr(args, 'l0_gamma') else -0.1
    l0_zeta = args.l0_zeta if hasattr(args, 'l0_zeta') else 1.1
    l0_beta = args.l0_beta if hasattr(args, 'l0_beta') else 0.66
    baseline_ema = args.baseline_ema if hasattr(args, 'baseline_ema') else 0.9
    initial_temp = args.initial_temp if hasattr(args, 'initial_temp') else 5.0
    
    # Initialize model
    model = ImprovedEdgeGNN(
        feature_dim=512,
        hidden_dim=256,
        num_classes=args.n_class,
        lambda_reg=lambda_reg,
        reg_mode=reg_mode,
        l0_method=l0_method,
        edge_dim=args.edge_dim,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout,
        num_gnn_layers=args.num_gnn_layers,
        num_attention_heads=args.num_attention_heads,
        use_attention_pooling=args.use_attention_pooling,
        l0_gamma=l0_gamma,
        l0_zeta=l0_zeta,
        l0_beta=l0_beta,
        baseline_ema=baseline_ema,
        initial_temp=initial_temp
    ).to(device)
    
    # Print model info
    print(f"\nModel Configuration:")
    print(f"  L0 Method: {l0_method}")
    print(f"  Regularization: {reg_mode}, λ={lambda_reg}")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    print(f"  L0 Params: γ={l0_gamma}, ζ={l0_zeta}, β={l0_beta}")
    if l0_method == 'arm':
        print(f"  ARM Baseline EMA: {baseline_ema}")
    
    # Initialize optimizer and scheduler
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
    
    # Setup WandB
    wandb_config = None
    if use_wandb:
        # Generate run name
        if run_name is None:
            fold_str = f"_fold{fold}" if fold is not None else ""
            run_name = f"{l0_method}{fold_str}"
        
        # Create config dict
        config_dict = {
            # Model
            'l0_method': l0_method,
            'reg_mode': reg_mode,
            'lambda_reg': lambda_reg,
            'l0_gamma': l0_gamma,
            'l0_zeta': l0_zeta,
            'l0_beta': l0_beta,
            'baseline_ema': baseline_ema,
            
            # Training
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'initial_temp': initial_temp,
            
            # Architecture
            'feature_dim': 512,
            'hidden_dim': 256,
            'edge_dim': args.edge_dim,
            'num_classes': args.n_class,
            'dropout': args.dropout,
            
            # Graph
            'graph_size_adaptation': args.graph_size_adaptation,
            'min_edges_per_node': args.min_edges_per_node,
            
            # Data
            'train_size': len(train_idx),
            'val_size': len(val_idx),
        }
        
        if fold is not None:
            config_dict['fold'] = fold
        
        # Create WandB config
        wandb_config = {
            'project': wandb_project,
            'name': run_name,
            'config': config_dict,
            'tags': [l0_method, f'fold{fold}' if fold is not None else 'single'],
            'group': l0_method,
            'job_type': 'train',
        }
        
        print(f"\nWandB Logging: Enabled")
        print(f"  Project: {wandb_project}")
        print(f"  Run: {run_name}")
    else:
        print(f"\nWandB Logging: Disabled")
    
    # Train and evaluate
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
        warmup_epochs=args.warmup_epochs,
        wandb_config=wandb_config,
        l0_method=l0_method
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': results["val_accs"][-1],
        'best_val_accuracy': results["best_val_acc"],
        'best_epoch': results["best_epoch"],
        'l0_method': l0_method,
        'config': vars(args),
    }, final_model_path)
    
    print(f"\nFinal model saved: {final_model_path}")
    
    # Save edge weight evolution plot
    if hasattr(model, 'plot_stats'):
        stats_path = os.path.join(output_dir, 'edge_weight_evolution.png')
        model.plot_stats(save_path=stats_path)
        print(f"Edge weight evolution saved: {stats_path}")
    
    # Plot training metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.png')
    plot_metrics(
        results["train_accs"],
        results["val_accs"],
        results["train_losses"],
        results["val_losses"],
        metrics_path,
        title=f"LENS Training Metrics ({l0_method})",
        warmup_epochs=args.warmup_epochs
    )
    print(f"Training metrics saved: {metrics_path}")
    
    # Save detailed results
    save_training_results(args, output_dir, results, model, l0_method)
    
    # Print summary
    print_training_summary(results, model, l0_method)
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Return results
    return {
        "results": results,
        "model": model,
        "l0_method": l0_method
    }


def save_training_results(args, output_dir, results, model, l0_method):
    """Save comprehensive training results to file"""
    results_path = os.path.join(output_dir, 'training_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("LENS MODEL TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"L0 Method: {l0_method}\n")
        
        lambda_value = args.lambda_reg if hasattr(args, 'lambda_reg') else args.beta
        f.write(f"Lambda: {lambda_value}\n")
        
        reg_mode = args.reg_mode if hasattr(args, 'reg_mode') else args.egl_mode
        f.write(f"Regularization Mode: {reg_mode}\n")
        
        # L0 parameters
        if reg_mode == 'l0':
            l0_gamma = args.l0_gamma if hasattr(args, 'l0_gamma') else -0.1
            l0_zeta = args.l0_zeta if hasattr(args, 'l0_zeta') else 1.1
            l0_beta = args.l0_beta if hasattr(args, 'l0_beta') else 0.66
            
            f.write(f"L0 Parameters:\n")
            f.write(f"  - gamma: {l0_gamma}\n")
            f.write(f"  - zeta: {l0_zeta}\n")
            f.write(f"  - beta: {l0_beta}\n")
            
            if l0_method == 'arm':
                baseline_ema = args.baseline_ema if hasattr(args, 'baseline_ema') else 0.9
                f.write(f"  - baseline_ema: {baseline_ema}\n")
        
        # Other parameters
        f.write(f"\nArchitecture:\n")
        f.write(f"  - Edge Dimension: {args.edge_dim}\n")
        f.write(f"  - Hidden Dimensions: {args.hidden_dims if hasattr(args, 'hidden_dims') else 512}\n")
        f.write(f"  - Dropout: {args.dropout}\n")
        
        f.write(f"\nTraining:\n")
        f.write(f"  - Epochs: {args.epochs}\n")
        f.write(f"  - Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"  - Learning Rate: {args.lr}\n")
        f.write(f"  - Weight Decay: {args.weight_decay}\n")
        f.write(f"  - Batch Size: {args.batch_size}\n")
        
        f.write(f"\nGraph Processing:\n")
        f.write(f"  - Graph Size Adaptation: {args.graph_size_adaptation}\n")
        f.write(f"  - Min Edges Per Node: {args.min_edges_per_node}\n")
        
        # Performance
        f.write(f"\n{'='*60}\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"Best Validation Accuracy: {results['best_val_acc']:.4f}\n")
        f.write(f"Best Epoch: {results['best_epoch']}\n")
        
        if 'best_edge_sparsity' in results:
            f.write(f"Edge Sparsity at Best: {results['best_edge_sparsity']:.1f}%\n")
        
        f.write(f"\nFinal Epoch:\n")
        f.write(f"  - Train Accuracy: {results['train_accs'][-1]:.4f}\n")
        f.write(f"  - Val Accuracy: {results['val_accs'][-1]:.4f}\n")
        f.write(f"  - Train Loss: {results['train_losses'][-1]:.4f}\n")
        f.write(f"  - Val Loss: {results['val_losses'][-1]:.4f}\n")
        
        if hasattr(model, 'stats_tracker'):
            if hasattr(model.stats_tracker, 'edge_density_history'):
                if len(model.stats_tracker.edge_density_history) > 0:
                    final_density = model.stats_tracker.edge_density_history[-1]
                    f.write(f"  - Final Edge Density: {final_density:.4f}\n")
        
        # Method-specific notes
        f.write(f"\n{'='*60}\n")
        f.write(f"METHOD-SPECIFIC NOTES ({l0_method.upper()})\n")
        f.write("-" * 40 + "\n")
        
        if l0_method == 'hard-concrete':
            f.write("Hard-Concrete Relaxation:\n")
            f.write("  - Continuous relaxation provides stable gradients\n")
            f.write("  - Proven method for graph sparsification\n")
            f.write("  - Consider ARM if you need binary gates during training\n")
            
        elif l0_method == 'arm':
            f.write("ARM (Augment-REINFORCE-Merge):\n")
            f.write("  - Direct binary sampling during training\n")
            f.write("  - Control variates reduce gradient variance\n")
            f.write("  - Check gradient variance in WandB logs\n")
            f.write("  - If unstable, try increasing baseline_ema\n")
    
    print(f"\nDetailed results saved: {results_path}")


def print_training_summary(results, model, l0_method):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print(f"LENS TRAINING SUMMARY ({l0_method.upper()})")
    print("="*60)
    
    print(f"\nPerformance:")
    print(f"  Best Val Accuracy: {results['best_val_acc']:.4f} (Epoch {results['best_epoch']})")
    
    if 'best_edge_sparsity' in results:
        print(f"  Edge Sparsity: {results['best_edge_sparsity']:.1f}%")
    
    print(f"\nFinal Epoch:")
    print(f"  Train Accuracy: {results['train_accs'][-1]:.4f}")
    print(f"  Val Accuracy: {results['val_accs'][-1]:.4f}")
    print(f"  Gap: {abs(results['train_accs'][-1] - results['val_accs'][-1]):.4f}")
    
    if hasattr(model, 'stats_tracker'):
        if hasattr(model.stats_tracker, 'edge_density_history'):
            if len(model.stats_tracker.edge_density_history) > 0:
                final_density = model.stats_tracker.edge_density_history[-1]
                print(f"\nFinal Edge Density: {final_density:.4f}")
    
    print("\n" + "="*60)
