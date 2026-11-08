import argparse
import os


def add_common_args(parser, n_features_default=512, batch_size_default=1,
                     output_dir_default='results', require_train_list=True,
                     require_test_list=False, require_output_dir=False):
    """
    Data/IO/setup flags shared across every LENS entrypoint (main.py,
    training/train_survival.py, training/train_graphlsurv_survival.py,
    baselines/reproductionScript.py). Call this once per entrypoint's parser,
    then add task-specific flags on top -- keeps names/help text/defaults
    consistent everywhere instead of each script re-inventing its own (this
    repo used to have --data_root/--train_ids/--val_ids with underscores and
    no --n-features at all in baselines/reproductionScript.py, alongside
    --data-root/--train-list/--test-list with hyphens everywhere else).

    n_features_default/batch_size_default/output_dir_default: entrypoints
    with a different sensible default (e.g. train_survival.py's 2048-dim
    features) pass their own value here rather than editing this function.
    require_train_list: set False for scripts where a single positional
    dataset makes more sense than train/test lists (none currently do, but
    keeps this from being classification-pipeline-specific by construction).
    require_test_list: the survival pipeline's whole protocol (a fixed
    held-out set evaluated once per fold) depends on this existing from the
    start, so train_survival.py passes True; main.py's --test-list is a
    later, optional addition and leaves this False.
    """
    parser.add_argument('--data-root', type=str, required=True,
                         help='Root directory containing per-slide graph data (features.pt/adj_s.pt/c_idx.txt)')
    parser.add_argument('--train-list', type=str, required=require_train_list,
                         help='Training list file (tab-separated: slide_id, label/y_disc, ...)')
    parser.add_argument('--test-list', type=str, default=None, required=require_test_list,
                         help='Held-out test list file, disjoint from --train-list. '
                              'The selected best checkpoint is automatically evaluated '
                              'against it once training finishes -- no separate test '
                              'script needed.')
    parser.add_argument('--n-features', type=int, default=n_features_default,
                         help=f'Node feature dimension (512=SimCLR-ResNet18, 1536=UNI2-h) '
                              f'(default: {n_features_default})')
    parser.add_argument('--batch-size', type=int, default=batch_size_default, help='Batch size')
    parser.add_argument('--output-dir', type=str, default=output_dir_default, required=require_output_dir,
                         help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split-seed', type=int, default=None,
                         help='Seed for the cross-validation split only (default: --seed). '
                              'Set it to a fixed value to keep the folds identical while '
                              '--seed varies the initialization.')
    parser.add_argument('--group-by-patient', action='store_true', default=True,
                         help='Keep all slides of a patient in the same fold (default)')
    parser.add_argument('--no-group-by-patient', action='store_false', dest='group_by_patient',
                         help='Legacy slide-level StratifiedKFold split (slides of one patient '
                              'can land in both train and validation)')
    return parser


def validate_common_paths(args):
    """
    Fail fast with a clear, actionable message instead of a bare
    FileNotFoundError/stack trace deep inside GraphDataset or open().
    Call right after parser.parse_args() in every entrypoint that used
    add_common_args (or otherwise takes --data-root/--train-list/--test-list).

    Raises SystemExit(2) on the first problem found, matching argparse's own
    error exit code and error-message style.
    """
    errors = []

    if not os.path.isdir(args.data_root):
        errors.append(f"--data-root does not exist or is not a directory: {args.data_root}")

    train_list = getattr(args, 'train_list', None)
    if train_list is not None and not os.path.isfile(train_list):
        errors.append(f"--train-list does not exist or is not a file: {train_list}")

    test_list = getattr(args, 'test_list', None)
    if test_list is not None and not os.path.isfile(test_list):
        errors.append(f"--test-list does not exist or is not a file: {test_list}")

    if errors:
        message = "\n".join(f"  - {e}" for e in errors)
        raise SystemExit(f"error: invalid path argument(s):\n{message}")


def get_parser():
    """Create and return the argument parser for the classification model (main.py)."""
    parser = argparse.ArgumentParser(description='ImprovedEdgeGNN for Graph Classification with Smart Sparsification')

    add_common_args(parser)

    # Basic parameters
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs (paper: 80)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n-class', type=int, default=3, help='Number of classes')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--max-folds', type=int, default=None,
                         help='Stop after training this many folds (still uses the same '
                              '--n-folds StratifiedKFold split, just trains fewer of them) -- '
                              'e.g. --max-folds 1 for cheap per-trial hyperparameter sweeps. '
                              'Default: train all --n-folds.')

    # Model parameters. Was --hidden-dims (nargs='+', accepting a list) but
    # training/training.py always hardcoded hidden_dim=256 regardless of what
    # was passed here -- this flag did nothing. Replaced with a plain scalar
    # that's now actually wired through (see training/training.py).
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for GNN/pooling/classifier layers (default: 256)')
    parser.add_argument('--edge-dim', type=int, default=128,
                        help='Hidden dimension for edge scoring network')
    parser.add_argument('--no-edges', action='store_true',
                        help='No-edge ablation: train and evaluate on empty graphs (implies --reg-mode none). '
                             'The readout falls back to all real patches.')

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
    parser.add_argument('--warmup-epochs', type=int, default=15,
                        help='Number of warm-up epochs (paper: 15); also the LR warm-up length')
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
    parser.add_argument('--l0-method', type=str, default='hard-concrete',
                        choices=['hard-concrete', 'arm', 'ste'])
    parser.add_argument('--baseline-ema', type=float, default=0.9)

    # Target density control parameters
    parser.add_argument('--target-density', type=float, default=0.30,
                        help='Target edge retention rate (default: 0.30)')

    parser.add_argument('--edge-logit-norm', action='store_true',
                        help='Standardize each graph\'s edge logits and place the '
                             '--target-density quantile at the gate midpoint, so about that '
                             'fraction of every graph\'s edges is kept by construction '
                             '(off by default; hard-concrete only)')

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
    parser.add_argument('--num-attention-heads', type=int, default=8,
                        help='Number of attention heads for pooling (default: 8; paper: 8 heads of dim 32 at hidden-dim 256)')
    parser.add_argument('--use-attention-pooling', action='store_true', default=True,
                        help='Use multi-head attention pooling (default: True)')
    parser.add_argument('--no-attention-pooling', action='store_false', dest='use_attention_pooling',
                        help='Disable attention pooling, use standard pooling')

    return parser
