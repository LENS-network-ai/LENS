import argparse

def get_parser():
    """Create and return the argument parser for the model"""
    parser = argparse.ArgumentParser(description='ImprovedEdgeGNN for Graph Classification with Smart Sparsification')
    
    # Basic parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-features', type=int, default=512, help='Number of node features')
    parser.add_argument('--n-class', type=int, default=3, help='Number of classes')
    parser.add_argument('--data-root', type=str, required=True, help='Data root directory')
    parser.add_argument('--train-list', type=str, required=True, help='Training list file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    
    # Model parameters
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=128, 
                        help='Hidden dimensions for layers')
    parser.add_argument('--edge-dim', type=int, default=128,
                        help='Hidden dimension for edge scoring network')
    
    # Lambda/beta regularization
    parser.add_argument('--lambda-reg', type=float, default=0.01,
                        help='Regularization strength (default: 0.01)')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='[DEPRECATED] Use --lambda-reg instead')
    
    # Regularization mode
    parser.add_argument('--reg-mode', type=str, default='l0',
                        choices=['l0', 'egl', 'none'],
                        help='Regularization mode (default: l0)')
    parser.add_argument('--egl-mode', type=str, default='l0',
                        choices=['l0', 'none', 'egl', 'l1', 'l2', 'entropy'],
                        help='[DEPRECATED] Use --reg-mode instead')
    
    # Common regularization parameters
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of epochs for warmup')
    parser.add_argument('--graph-size-adaptation', action='store_true',
                        help='Enable adaptive pruning based on graph size')
    parser.add_argument('--min-edges-per-node', type=float, default=2.0,
                        help='Minimum number of edges to keep per node')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for edge scoring network')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--initial-temp', type=float, default=5.0,
                        help='Initial temperature for edge gating')
    
    # L0 regularization specific parameters
    parser.add_argument('--l0-gamma', type=float, default=-0.1,
                        help='L0 regularization gamma parameter (default: -0.1)')
    parser.add_argument('--l0-zeta', type=float, default=1.1,
                        help='L0 regularization zeta parameter (default: 1.1)')
    parser.add_argument('--l0-beta', type=float, default=0.66,
                        help='L0 regularization beta parameter (default: 0.66)')
    
    # Target sparsity parameters
    parser.add_argument('--target-sparsity', type=float, default=0.7,
                        help='Target sparsity rate (0.0-1.0)')
    parser.add_argument('--sparsity-penalty', type=float, default=5.0,
                        help='Weight for sparsity deviation penalty')
    
    # Bayesian optimization parameters
    parser.add_argument('--run-bayesian-opt', action='store_true',
                        help='Run Bayesian optimization instead of regular training')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for Bayesian optimization')
    
    # Analysis parameters
    parser.add_argument('--detailed-analysis', action='store_true',
                        help='Enable more frequent detailed graph analysis')
    parser.add_argument('--l0_method', type=str, default='hard-concrete',
                        choices=['hard-concrete', 'arm', 'ste'])
    parser.add_argument('--baseline_ema', type=float, default=0.9)
    
    # Target density control parameters
    parser.add_argument('--target-density', type=float, default=0.30,
                        help='Target edge retention rate (default: 0.30)')
    
    parser.add_argument('--lambda-density', type=float, default=0.03,
                        help='Density loss weight (default: 0.03)')
    
    parser.add_argument('--enable-adaptive-lambda', action='store_true', default=True,
                        help='Enable adaptive lambda mechanism (default: True)')
    
    parser.add_argument('--enable-density-loss', action='store_true', default=True,
                        help='Enable density loss term (default: True)')
    
    parser.add_argument('--alpha-min', type=float, default=0.2,
                        help='Minimum adaptive scaling factor (default: 0.2)')
    
    parser.add_argument('--alpha-max', type=float, default=2.0,
                        help='Maximum adaptive scaling factor (default: 2.0)')
    
    parser.add_argument('--ramp-epochs', type=int, default=20,
                        help='Number of epochs for lambda ramp after warmup (default: 20)')
    
    # Multi-layer GNN and attention pooling
    parser.add_argument('--num-gnn-layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--num-attention-heads', type=int, default=4,
                        help='Number of attention heads for pooling (default: 4)')
    parser.add_argument('--use-attention-pooling', action='store_true', default=True,
                        help='Use multi-head attention pooling (default: True)')
    parser.add_argument('--no-attention-pooling', action='store_false', dest='use_attention_pooling',
                        help='Disable attention pooling, use standard pooling')

    return parser
