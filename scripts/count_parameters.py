"""
Print exact parameter counts for the shared edge scorer, LENS (efficiency),
and LENS+ (accuracy), broken down by component -- for filling in the
architecture appendix's \\VERIFY{} parameter-count entries.

Both model/LENS.py and model/LENS2.py define a class named ImprovedEdgeGNN
(same name, different files) -- imported here under distinct aliases to
avoid collision.

IMPORTANT: parameter counts here depend entirely on the --n-features,
--hidden-dim, --edge-dim, --n-class values passed below. These defaults
match: feature_dim=1536 (UNI2-h, per the appendix text), hidden_dim=128 for
LENS / 256 for LENS+ (256 is the value training/training.py hardcoded for
every run before today's fix -- see that file's history), edge_dim=128
(main.py/utils/config.py's --edge-dim CLI default -- NOT the same as
EdgeScoringNetwork's own internal default of 32, which only applies if a
caller never passes edge_dim explicitly). Edit the CONFIG block below if
your actual reported runs used different values -- these numbers are only
correct for the config actually used to produce the reported results.

Usage:
    python -m scripts.count_parameters
"""

from model.LENS import ImprovedEdgeGNN as LENSEfficiency
from model.LENS2 import ImprovedEdgeGNN as LENSPlus

# ---- CONFIG: edit to match your actual reported run ------------------------
FEATURE_DIM = 1536      # UNI2-h
NUM_CLASSES = 3
EDGE_DIM = 128           # main.py's --edge-dim CLI default (NOT EdgeScoringNetwork's own default of 32)
LENS_HIDDEN_DIM = 128    # matches classifier "128 -> 64 -> C" in the appendix
LENS_PLUS_HIDDEN_DIM = 256  # hardcoded in training/training.py for every run before today's fix
LENS_PLUS_NUM_GNN_LAYERS = 3
LENS_PLUS_NUM_ATTENTION_HEADS = 4  # CLI default -- change to 8 if that was passed explicitly for your run
# -----------------------------------------------------------------------------


def count(module):
    return sum(p.numel() for p in module.parameters())


def report(name, model, component_names):
    total = count(model)
    print(f"\n{name}: {total:,} parameters total")
    for attr in component_names:
        sub = getattr(model, attr, None)
        if sub is not None:
            print(f"  {attr:<15}: {count(sub):,}")


def main():
    lens = LENSEfficiency(
        feature_dim=FEATURE_DIM, hidden_dim=LENS_HIDDEN_DIM, num_classes=NUM_CLASSES,
        edge_dim=EDGE_DIM,
    )
    lens_plus = LENSPlus(
        feature_dim=FEATURE_DIM, hidden_dim=LENS_PLUS_HIDDEN_DIM, num_classes=NUM_CLASSES,
        edge_dim=EDGE_DIM, num_gnn_layers=LENS_PLUS_NUM_GNN_LAYERS,
        num_attention_heads=LENS_PLUS_NUM_ATTENTION_HEADS,
    )

    print("=" * 70)
    print("PARAMETER COUNTS")
    print(f"(feature_dim={FEATURE_DIM}, edge_dim={EDGE_DIM}, "
          f"LENS hidden_dim={LENS_HIDDEN_DIM}, LENS+ hidden_dim={LENS_PLUS_HIDDEN_DIM})")
    print("=" * 70)

    # Edge scorer is identical in structure between LENS and LENS+ (same
    # EdgeScoringNetwork class, same feature_dim/edge_dim) -- reporting once
    # via the LENS instance is representative of both, and LENS+'s own count
    # below confirms it matches.
    print(f"\nShared edge scorer: {count(lens.edge_scorer):,} parameters")
    print(f"  feature_attention: {count(lens.edge_scorer.feature_attention):,}")
    print(f"  edge_mlp         : {count(lens.edge_scorer.edge_mlp):,}")

    report("LENS (efficiency)", lens, ["edge_scorer", "conv", "classifier"])
    report("LENS+ (accuracy)", lens_plus, ["edge_scorer", "gnn", "pooling", "classifier"])


if __name__ == "__main__":
    main()
