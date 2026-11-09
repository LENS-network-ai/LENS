import os
import torch
import numpy as np
import optuna
from datetime import datetime
from utils.splits import make_folds

from training.training import train_edge_gnn
from utils.dataset import GraphDataset
from utils.config import get_parser

def objective(trial, dataset, all_labels, args):
    """
    Objective function for Bayesian optimization: Accval − λ · |SparsityRate − Target|
    
    Args:
        trial: Optuna trial object
        dataset: GraphDataset instance
        all_labels: Labels for all examples
        args: Base arguments
        
    Returns:
        Objective value to maximize
    """
    # Define parameters for this trial
    lambda_reg = trial.suggest_float('lambda_reg', 0.001, 0.05, log=True)
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 10)
    initial_temp = trial.suggest_float('initial_temp', 2.0, 10.0)
    l0_gamma = trial.suggest_float('l0_gamma', -0.2, -0.05)
    l0_zeta = trial.suggest_float('l0_zeta', 1.05, 1.2)
    l0_beta = trial.suggest_float('l0_beta', 0.5, 1.0)
    
    # Update args for this trial
    args.lambda_reg = lambda_reg
    args.warmup_epochs = warmup_epochs
    args.initial_temp = initial_temp
    args.l0_gamma = l0_gamma
    args.l0_zeta = l0_zeta
    args.l0_beta = l0_beta
    args.reg_mode = 'l0'  # Ensure L0 mode is used
    
    # Create trial output directory
    trial_dir = f"./optuna_trials/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Set up cross-validation
    folds = make_folds(dataset.ids, all_labels, args.n_folds,
                       getattr(args, 'split_seed', None) or args.seed,
                       getattr(args, 'group_by_patient', True))
    fold_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        fold_dir = os.path.join(trial_dir, f"fold{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        fold_results[f"fold{fold+1}"] = train_edge_gnn(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            args=args,
            output_dir=fold_dir
        )
    
    # Calculate mean accuracy and sparsity across folds
    val_accs = [fold_results[f"fold{fold+1}"]["results"]["best_val_acc"] 
                for fold in range(args.n_folds)]
    edge_sparsities = []
    for fold in range(args.n_folds):
        if "best_edge_sparsity" in fold_results[f"fold{fold+1}"]["results"]:
            # Convert percentage to fraction (0-1)
            edge_sparsity = fold_results[f"fold{fold+1}"]["results"]["best_edge_sparsity"] / 100.0
            edge_sparsities.append(edge_sparsity)
    
    mean_acc = np.mean(val_accs)
    mean_sparsity = np.mean(edge_sparsities) if edge_sparsities else 0
    
    # Calculate objective with target sparsity penalty: O = Accval − penalty_weight · |SparsityRate − Target|
    objective_value = mean_acc - args.sparsity_penalty * abs(mean_sparsity - args.target_sparsity)
    
    # Log the components
    trial.set_user_attr('mean_accuracy', mean_acc)
    trial.set_user_attr('mean_sparsity', mean_sparsity)
    trial.set_user_attr('sparsity_penalty', args.sparsity_penalty * abs(mean_sparsity - args.target_sparsity))
    trial.set_user_attr('lambda_value', lambda_reg)
    
    return objective_value

def run_bayesian_optimization(args):
    """Run Bayesian optimization to find optimal parameters"""
    # Load dataset
    with open(args.train_list, 'r') as f:
        all_ids = f.readlines()
    
    dataset = GraphDataset(root=args.data_root, ids=all_ids, no_edges=getattr(args, 'no_edges', False))
    all_labels = dataset.labels()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"bayesian_opt_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name="L0_optimization")
    study.optimize(lambda trial: objective(trial, dataset, all_labels, args), 
                   n_trials=args.n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Print and save results
    print("\n" + "="*60)
    print("🏆 BAYESIAN OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best λ value: {best_params['lambda_reg']:.6f}")
    print(f"Best warmup_epochs: {best_params['warmup_epochs']}")
    print(f"Best initial_temp: {best_params['initial_temp']:.2f}")
    print(f"Best L0 parameters: gamma={best_params['l0_gamma']:.3f}, zeta={best_params['l0_zeta']:.3f}, beta={best_params['l0_beta']:.3f}")
    print(f"Best validation accuracy: {best_trial.user_attrs['mean_accuracy']:.4f}")
    print(f"Achieved sparsity: {best_trial.user_attrs['mean_sparsity']:.4f}")
    print(f"Target sparsity: {args.target_sparsity:.4f}")
    print(f"Final objective value: {best_trial.value:.4f}")
    
    # Save results to file
    with open(os.path.join(output_dir, 'optimization_results.txt'), 'w') as f:
        f.write("BAYESIAN OPTIMIZATION RESULTS\n")
        f.write("="*40 + "\n\n")
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
    
    # Save the optimization history
    optuna.visualization.plot_optimization_history(study).write_image(
        os.path.join(output_dir, 'optimization_history.png'))
    optuna.visualization.plot_param_importances(study).write_image(
        os.path.join(output_dir, 'param_importances.png'))
    
    return best_params, best_trial.value

if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Bayesian optimization trials')
    args = parser.parse_args()
    
    # Run Bayesian optimization
    best_params, best_value = run_bayesian_optimization(args)
