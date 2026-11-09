#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import argparse
import optuna
from datetime import datetime
from torch.utils.data import Subset

from training.training import train_edge_gnn
from utils.dataset import GraphDataset
from utils.config import get_parser, validate_common_paths
from utils.splits import make_folds, patient_id

def run_bayesian_optimization(args):
    """Run Bayesian optimization to find optimal parameters"""
    print("\n" + "="*60)
    print("STARTING BAYESIAN OPTIMIZATION")
    print("="*60)
    print(f"Target sparsity: {args.target_sparsity}")
    print(f"Sparsity penalty weight: {args.sparsity_penalty}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Number of CV folds: {args.n_folds}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"bayesian_opt_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("BAYESIAN OPTIMIZATION CONFIGURATION\n")
        f.write("=" * 40 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    # Load dataset
    with open(args.train_list, 'r') as f:
        all_ids = f.readlines()
    
    dataset = GraphDataset(root=args.data_root, ids=all_ids, no_edges=args.no_edges)
    all_labels = dataset.labels()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Define the objective function for optimization
    def objective(trial):
        # Define parameters to optimize
        trial_params = {
            'lambda_reg': trial.suggest_float('lambda_reg', 0.001, 0.05, log=True),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 3, 10),
            'initial_temp': trial.suggest_float('initial_temp', 2.0, 10.0),
            'l0_gamma': trial.suggest_float('l0_gamma', -0.2, -0.05),
            'l0_zeta': trial.suggest_float('l0_zeta', 1.05, 1.2),
            'l0_beta': trial.suggest_float('l0_beta', 0.5, 1.0),
        }
        
        # Create a copy of args and update with trial parameters
        trial_args = argparse.Namespace(**vars(args))
        for param, value in trial_params.items():
            setattr(trial_args, param, value)
        
        # Ensure we're using L0 regularization
        trial_args.reg_mode = 'l0'
        
        # Create trial output directory
        trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Set up cross-validation
        folds = make_folds(dataset.ids, all_labels, args.n_folds,
                           args.seed if args.split_seed is None else args.split_seed,
                           args.group_by_patient)
        fold_results = {}
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"\nTrial {trial.number}, Fold {fold+1}/{args.n_folds}")
            fold_dir = os.path.join(trial_dir, f"fold{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            fold_results[f"fold{fold+1}"] = train_edge_gnn(
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                args=trial_args,
                output_dir=fold_dir
            )
        
        # Calculate mean metrics across folds
        val_accs = [fold_results[f"fold{fold+1}"]["results"]["best_val_acc"] 
                    for fold in range(args.n_folds)]
        edge_sparsities = []
        for fold in range(args.n_folds):
            if "best_edge_sparsity" in fold_results[f"fold{fold+1}"]["results"]:
                # Convert percentage to fraction for objective function
                edge_sparsity = fold_results[f"fold{fold+1}"]["results"]["best_edge_sparsity"] / 100.0
                edge_sparsities.append(edge_sparsity)
        
        mean_acc = np.mean(val_accs)
        
        # If sparsity couldn't be computed, heavily penalize
        if not edge_sparsities:
            return mean_acc - args.sparsity_penalty
        
        mean_sparsity = np.mean(edge_sparsities)
        
        # Calculate objective with target sparsity penalty
        objective_value = mean_acc - args.sparsity_penalty * abs(mean_sparsity - args.target_sparsity)
        
        # Log the components
        trial.set_user_attr('mean_accuracy', mean_acc)
        trial.set_user_attr('mean_sparsity', mean_sparsity)
        trial.set_user_attr('sparsity_penalty', args.sparsity_penalty * abs(mean_sparsity - args.target_sparsity))
        
        # Save trial summary
        with open(os.path.join(trial_dir, 'trial_summary.txt'), 'w') as f:
            f.write(f"Trial {trial.number} Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write("Parameters:\n")
            for param, value in trial_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nResults:\n")
            f.write(f"  Mean Validation Accuracy: {mean_acc:.4f}\n")
            f.write(f"  Mean Edge Sparsity: {mean_sparsity:.4f}\n")
            f.write(f"  Target Sparsity: {args.target_sparsity:.4f}\n")
            f.write(f"  Sparsity Penalty: {args.sparsity_penalty * abs(mean_sparsity - args.target_sparsity):.4f}\n")
            f.write(f"  Objective Value: {objective_value:.4f}\n")
        
        print(f"\nTrial {trial.number} finished with objective value: {objective_value:.4f}")
        print(f"  Accuracy: {mean_acc:.4f}, Sparsity: {mean_sparsity:.4f}, Target: {args.target_sparsity:.4f}")
        
        return objective_value
    
    # Create and run the study
    study = optuna.create_study(direction='maximize', study_name="L0_optimization")
    study.optimize(objective, n_trials=args.n_trials)
    
    # Get best parameters and results
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Print and save results
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best lambda value: {best_params['lambda_reg']:.6f}")
    print(f"Best warmup_epochs: {best_params['warmup_epochs']}")
    print(f"Best initial_temp: {best_params['initial_temp']:.2f}")
    print(f"Best L0 parameters: gamma={best_params['l0_gamma']:.3f}, zeta={best_params['l0_zeta']:.3f}, beta={best_params['l0_beta']:.3f}")
    print(f"Best validation accuracy: {best_trial.user_attrs['mean_accuracy']:.4f}")
    print(f"Achieved sparsity: {best_trial.user_attrs['mean_sparsity']:.4f}")
    print(f"Target sparsity: {args.target_sparsity:.4f}")
    print(f"Final objective value: {best_trial.value:.4f}")
    
    # Save final results to file
    with open(os.path.join(output_dir, 'optimization_results.txt'), 'w') as f:
        f.write("BAYESIAN OPTIMIZATION RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Target sparsity: {args.target_sparsity}\n")
        f.write(f"Sparsity penalty weight: {args.sparsity_penalty}\n\n")
        
        f.write("BEST PARAMETERS:\n")
        for param_name, param_value in best_params.items():
            f.write(f"{param_name}: {param_value}\n")
        
        f.write("\nPERFORMANCE:\n")
        f.write(f"Validation accuracy: {best_trial.user_attrs['mean_accuracy']:.4f}\n")
        f.write(f"Achieved sparsity: {best_trial.user_attrs['mean_sparsity']:.4f}\n")
        f.write(f"Sparsity penalty: {best_trial.user_attrs['sparsity_penalty']:.4f}\n")
        f.write(f"Objective value: {best_trial.value:.4f}\n")
    
    # Try to save the optimization plots if plotly is available
    try:
        import plotly
        # Save the optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        # Save the parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(output_dir, 'param_importances.png'))
        
        # Save the parameter relationships
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_image(os.path.join(output_dir, 'param_relationships.png'))
    except:
        print("Warning: Could not save optimization plots. Ensure plotly is installed.")
    
    return best_params, best_trial.value, output_dir

def run_standard_training(args):
    """Run standard cross-validation training"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    with open(args.train_list, 'r') as f:
        all_ids = f.readlines()

    dataset = GraphDataset(root=args.data_root, ids=all_ids, no_edges=args.no_edges)
    all_labels = dataset.labels()

    # Optional fixed held-out test set (disjoint from --train-list, never
    # used for model selection): built once and shared across every fold,
    # same as --train-list's dataset above. Each fold evaluates its own
    # selected checkpoint against this same set right after training --
    # see train_edge_gnn/train_and_evaluate -- instead of a separate test
    # script run manually afterward.
    test_dataset = None
    if args.test_list:
        with open(args.test_list, 'r') as f:
            test_ids = f.readlines()
        test_dataset = GraphDataset(root=args.data_root, ids=test_ids, no_edges=args.no_edges)
        print(f"Held-out test set: {len(test_dataset)} samples from {args.test_list}")

    # Handle parameter name conversion
    if hasattr(args, 'lambda_reg') and not hasattr(args, 'beta'):
        args.beta = args.lambda_reg
    elif hasattr(args, 'beta') and not hasattr(args, 'lambda_reg'):
        args.lambda_reg = args.beta
        
    if hasattr(args, 'reg_mode') and not hasattr(args, 'egl_mode'):
        args.egl_mode = args.reg_mode
    elif hasattr(args, 'egl_mode') and not hasattr(args, 'reg_mode'):
        args.reg_mode = args.egl_mode
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write("CONFIGURATION\n")
        f.write("=" * 20 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    # Set up cross-validation
    folds = make_folds(dataset.ids, all_labels, args.n_folds,
                       args.seed if args.split_seed is None else args.split_seed,
                       args.group_by_patient)
    n_patients = len({patient_id(i.split()[0]) for i in dataset.ids})
    print(f"CV split: {'patient-grouped' if args.group_by_patient else 'slide-level'} "
          f"({len(dataset.ids)} slides, {n_patients} patients)")
    
    fold_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\nFold {fold+1}/{args.n_folds}")
        fold_dir = os.path.join(args.output_dir, f"fold{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Train model for this fold
        fold_results[f"fold{fold+1}"] = train_edge_gnn(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            args=args,
            output_dir=fold_dir,
            test_dataset=test_dataset,
        )

        # --max-folds stops early (e.g. for a hyperparameter sweep, where
        # training all --n-folds per trial is too expensive) while still
        # using the SAME StratifiedKFold split every trial would use for a
        # full run -- fold 1 of a 1-fold run is fold 1 of a 5-fold run,
        # not a different, ad-hoc train/val split.
        if args.max_folds is not None and (fold + 1) >= args.max_folds:
            print(f"\n--max-folds={args.max_folds} reached, stopping after fold {fold+1}.")
            break

    summarize_cross_validation(fold_results, args)

def summarize_cross_validation(fold_results, args):
    """
    Summarize cross-validation results.

    Iterates over len(fold_results), NOT args.n_folds -- they can differ
    (--max-folds stopping early for a sweep, or any future early-exit) and
    indexing fold_results["fold{n_folds}"] when fewer folds actually ran
    raised a bare KeyError with no useful message.
    """
    n_trained = len(fold_results)
    val_accs = [fold_results[f"fold{fold+1}"]["results"]["best_val_acc"]
                for fold in range(n_trained)]

    # Present only when --test-list was given (see run_standard_training).
    test_accs = [fold_results[f"fold{fold+1}"]["results"]["test_accuracy"]
                 for fold in range(n_trained)
                 if "test_accuracy" in fold_results[f"fold{fold+1}"]["results"]]

    edge_sparsities = []
    for fold in range(n_trained):
        if "best_edge_sparsity" in fold_results[f"fold{fold+1}"]["results"]:
            edge_sparsities.append(fold_results[f"fold{fold+1}"]["results"]["best_edge_sparsity"])
    
    avg_acc = sum(val_accs) / len(val_accs)
    std_acc = np.std(val_accs)

    avg_test_acc = np.mean(test_accs) if test_accs else None
    std_test_acc = np.std(test_accs) if test_accs else None

    if edge_sparsities:
        avg_sparsity = sum(edge_sparsities) / len(edge_sparsities)
        std_sparsity = np.std(edge_sparsities)
    else:
        avg_sparsity = None
        std_sparsity = None

    # Get regularization parameter name
    reg_param_name = "Lambda" if hasattr(args, 'lambda_reg') else "Beta"
    reg_param_value = args.lambda_reg if hasattr(args, 'lambda_reg') else args.beta

    # Get regularization mode name
    reg_mode_name = "Regularization Mode" if hasattr(args, 'reg_mode') else "EGL Mode"
    reg_mode_value = args.reg_mode if hasattr(args, 'reg_mode') else args.egl_mode

    write_cv_summary(args, fold_results, avg_acc, std_acc, avg_sparsity, std_sparsity,
                    reg_param_name, reg_param_value, reg_mode_name, reg_mode_value,
                    avg_test_acc, std_test_acc)
    print_cv_summary(avg_acc, std_acc, avg_sparsity, std_sparsity, fold_results, args,
                   reg_param_name, reg_param_value, reg_mode_name, reg_mode_value,
                   avg_test_acc, std_test_acc)

def write_cv_summary(args, fold_results, avg_acc, std_acc, avg_sparsity, std_sparsity,
                    reg_param_name, reg_param_value, reg_mode_name, reg_mode_value,
                    avg_test_acc=None, std_test_acc=None):
    """Write cross-validation summary to a file"""
    with open(os.path.join(args.output_dir, 'cv_summary.txt'), 'w') as f:
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        f.write("IMPROVED EDGE GNN\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Validation Accuracy: {avg_acc:.4f} +/- {std_acc:.4f}\n")
        if avg_test_acc is not None:
            f.write(f"Average Test Accuracy: {avg_test_acc:.4f} +/- {std_test_acc:.4f}\n")
        if avg_sparsity is not None:
            f.write(f"Average Edge Sparsity: {avg_sparsity:.1f}% +/- {std_sparsity:.1f}%\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"{reg_param_name}: {reg_param_value}\n")
        f.write(f"{reg_mode_name}: {reg_mode_value}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Graph Size Adaptation: {args.graph_size_adaptation}\n")
        f.write(f"Min Edges Per Node: {args.min_edges_per_node}\n")
        
        # Include L0 parameters if using L0 regularization
        if reg_mode_value == 'l0' and hasattr(args, 'l0_gamma'):
            f.write(f"\nL0 PARAMETERS\n")
            f.write("-" * 20 + "\n")
            f.write(f"L0 Gamma: {args.l0_gamma}\n")
            f.write(f"L0 Zeta: {args.l0_zeta}\n")
            f.write(f"L0 Beta: {args.l0_beta}\n")
            if hasattr(args, 'initial_temp'):
                f.write(f"Initial Temperature: {args.initial_temp}\n")
        
        f.write("\nFOLD RESULTS\n")
        f.write("-" * 20 + "\n")
        for fold in range(len(fold_results)):
            results = fold_results[f"fold{fold+1}"]
            f.write(f"Fold {fold+1}: Acc={results['results']['best_val_acc']:.4f} (Epoch {results['results']['best_epoch']})")
            if "test_accuracy" in results["results"]:
                f.write(f", Test Acc={results['results']['test_accuracy']:.4f}")
            if "best_edge_sparsity" in results["results"]:
                f.write(f", Sparsity={results['results']['best_edge_sparsity']:.1f}%\n")
            else:
                f.write("\n")

def print_cv_summary(avg_acc, std_acc, avg_sparsity, std_sparsity, fold_results, args,
                    reg_param_name, reg_param_value, reg_mode_name, reg_mode_value,
                    avg_test_acc=None, std_test_acc=None):
    """Print cross-validation summary and recommendations"""
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE")
    print("="*60)
    print(f"ImprovedEdgeGNN: {avg_acc:.4f} +/- {std_acc:.4f}")
    if avg_test_acc is not None:
        print(f"Held-out Test Accuracy: {avg_test_acc:.4f} +/- {std_test_acc:.4f}")
    if avg_sparsity is not None:
        print(f"Average Edge Sparsity: {avg_sparsity:.1f}% +/- {std_sparsity:.1f}%")
    
    # Print L0 specific parameters if applicable
    if reg_mode_value == 'l0' and hasattr(args, 'l0_gamma'):
        print(f"\nL0 Parameters:")
        print(f"  Gamma: {args.l0_gamma}")
        print(f"  Zeta: {args.l0_zeta}")
        print(f"  Beta: {args.l0_beta}")
        if hasattr(args, 'initial_temp'):
            print(f"  Initial Temperature: {args.initial_temp}")
    
    # Final recommendations
    print("\nFINAL RECOMMENDATIONS:")
    
    # Analyze sparsity
    if avg_sparsity is not None:
        if avg_sparsity < 5.0:
            print("Model is pruning too aggressively:")
            print(f"  - Only {avg_sparsity:.1f}% of edges have weight > 0.1")
            print(f"  - Consider decreasing {reg_param_name.lower()} or increasing warmup epochs")
            if reg_mode_value == 'l0':
                print("  - Try adjusting L0 parameters (increase beta_l0, decrease gamma)")
        elif avg_sparsity > 50.0:
            print("Model is not pruning enough:")
            print(f"  - {avg_sparsity:.1f}% of edges have weight > 0.1")
            print(f"  - Consider increasing {reg_param_name.lower()}")
            if reg_mode_value == 'l0':
                print("  - Try adjusting L0 parameters (decrease beta_l0)")
        else:
            print(f"Edge sparsity looks good: {avg_sparsity:.1f}% of edges have weight > 0.1")
    
    # Output analysis location
    print(f"\nDetailed graph analysis and reports can be found in:")
    print(f"  {args.output_dir}")

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    validate_common_paths(args)
    if args.no_edges:
        # With an empty graph there is nothing to gate, so the L0/density terms
        # are meaningless; train as a plain per-patch model.
        args.reg_mode = args.egl_mode = 'none'
        print("--no-edges: empty graphs, regularization disabled (reg-mode none)")

    # Check whether to run Bayesian optimization or standard training
    if args.run_bayesian_opt:
        # Run Bayesian optimization
        best_params, best_value, output_dir = run_bayesian_optimization(args)
        print(f"\nBayesian optimization complete! Best parameters saved to {output_dir}")
    else:
        # Run standard cross-validation training
        run_standard_training(args)

if __name__ == "__main__":
    main()
