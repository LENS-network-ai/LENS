<!-- Project Title Banner -->
<h1 align="center">LENS: Learnable Edge Network Sparsification for Interpretable Histopathology</h1>
<!-- Overview Image -->
<div align="center">
<img width="3721" height="3371" alt="lens_verti" src="https://github.com/user-attachments/assets/61a89f7b-0584-47fe-842e-953f97e9c3c7" />
  <p><em>LENS: A graph neural network approach for interpretable histopathology analysis through learnable edge sparsification</em></p>
</div>





##  Pipeline Overview


##  Installation

```bash
# Clone the repository
git@github.com:LENS-network-ai/LENS.git
cd LENS

# Install dependencies
pip install -r requirements.txt
```

### Dataset Information
## Dataset Information

The **CPTAC (Clinical Proteomic Tumor Analysis Consortium)** dataset serves as the main benchmark in this work. CPTAC data are publicly available from the U.S. National Cancer Institute via The Cancer Imaging Archive (TCIA):

https://www.cancerimagingarchive.net/

The CPTAC sample IDs used in our experiments are provided in `CPTAC_IDs.txt`.

To assess the generalizability of our approach, we additionally evaluate on multiple **TCGA** cohorts:
- TCGA-BRCA
- TCGA-RCC
- TCGA Lung Cancer (LUAD and LUSC)

TCGA data are obtained from the Genomic Data Commons (GDC) portal : https://portal.gdc.cancer.gov/.


### Step 1: WSI Tiling
```bash
# -s 512      # Tile size: 512x512 pixels  
#-e 0        # Overlap: 0px (no extra pixels added on edges)  
#-j 32       # Threads: use 32 parallel threads  
#-B 50       # Max background: skip tiles with >50% background  
#-o [path]   # Output path: where to save the tiles  
#-M -1       # Magnification: -1 = all levels, or set a specific one  

python preprocessing/tiling.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```

### Step 2: Graph Construction
```python
# Create a graph dataset from the tiled images

python preprocessing/graph_construction.py  --weights "path_to_pretrained_feature_extractor" --dataset "path_to_patches" --output "../graphs"
```
In our work we have mainly used the pretrained feature extractor from GTP work: [ResNetSimCLR](https://github.com/vkola-lab/tmi2022/tree/main/feature_extractor)

**Expected structure**:
```
    data/
    ├── slides/                    # Original .svs files (optional for training)
    ├── graphs/LUAD/simclr_files/  # Preprocessed graph data
    │   ├── C3N-03093-21/          # CPTAC sample ID
    │   │   ├── adj_s.pt           # Adjacency matrix [N×N]
    │   │   ├── features.pt        # Node features [N×512] 
    │   │   └── c_idx.txt          # Patch coordinates
    │   ├── C3N-01179-21/
    │   │   └── ...
    │   └── C3N-01334-21/
    │       └── ...
    ├── trainVal5.txt              # Training/validation samples 70% (CPTAC IDs)
    └── test_list.txt              # Test samples 30 %(CPTAC IDs)
```
### Step 3: Model Training

There are two ways to run the LENS model:

1. **Standard Training**: Train the model with fixed hyperparameters
2. **Bayesian Optimization**: Automatically find optimal hyperparameters based on validation accuracy and sparsity

### Standard Training

To run standard training with cross-validation:
```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --batch-size 1 \
  --epochs 80 \
  --lambda-reg 0.001 \
  --lambda-density 0.03 \
  --target-density 0.30 \
  --reg-mode l0 \
  --l0-method hard-concrete \
  --warmup-epochs 15 \
  --ramp-epochs 20 \
 
```

#### Key Parameters

- `--data-root`: Directory containing the graph data
- `--train-list`: File with list of training examples (format: `filename\tlabel`)
- `--lambda-reg`: Base regularization strength (λ₀) for L0 penalty (default: 0.01)
- `--lambda-density`: Density loss weight (λ_ρ) for target density enforcement (default: 0.03)
- `--target-density`: Target edge retention rate (0.0-1.0, default: 0.30 = keep 30% of edges)
- `--reg-mode`: Regularization type (`l0`, or `none`)
- `--l0-method`: L0 relaxation method (`hard-concrete`, `arm`, or `ste`)
- `--warmup-epochs`: Number of epochs for linear warmup (default: 15)
- `--ramp-epochs`: Number of epochs for linear ramp after warmup (default: 20)


#### L0 Regularization Methods

LENS supports three L0 relaxation methods:

**1. Hard-Concrete (default, recommended)**
```bash
python main.py \
  --reg-mode l0 \
  --l0-method hard-concrete \
  --l0-gamma -0.1 \
  --l0-zeta 1.1 \
  --l0-beta 0.66 \
  --initial-temp 5.0
```

**2. ARM (Augment-REINFORCE-Merge)**
```bash
python main.py \
  --reg-mode l0 \
  --l0-method arm \

```

**3. STE (Straight-Through Estimator)**
```bash
python main.py \
  --reg-mode l0 \
  --l0-method ste \

```

#### L0 Parameters

- `--l0-gamma`: Lower bound of Hard-Concrete distribution (default: -0.1, must be < 0)
- `--l0-zeta`: Upper bound of Hard-Concrete distribution (default: 1.1, must be > 1)
- `--initial-temp`: Initial temperature for edge gating with cosine annealing (default: 5.0)


#### Adaptive Density Control

LENS uses adaptive lambda scaling to achieve target density:
```bash
python main.py \
  --enable-adaptive-lambda \
  --enable-density-loss \
  --target-density 0.30 \
  --lambda-density 0.03 \
  --alpha-min 0.2 \
  --alpha-max 2.0
```

- `--enable-adaptive-lambda`: Enable adaptive lambda mechanism (default: True)
- `--enable-density-loss`: Enable density loss term (default: True)
- `--alpha-min`: Minimum adaptive scaling factor (default: 0.2)
- `--alpha-max`: Maximum adaptive scaling factor (default: 2.0)

The effective lambda is computed as: **λ_eff = α · λ₀**, where **α = clip(1 + (ρ - ρ*), α_min, α_max)**

#### Multi-Layer GNN and Attention Pooling
```bash
python main.py \
  --num-gnn-layers 3 \
  --num-attention-heads 4 \
  --use-attention-pooling \
  --hidden-dim 256 \
  --edge-dim 128 \
  --dropout 0.2
```

- `--num-gnn-layers`: Number of graph convolution layers (default: 3)
- `--num-attention-heads`: Number of attention heads for pooling (default: 4)
- `--use-attention-pooling`: Use multi-head attention pooling (default: True)
- `--no-attention-pooling`: Disable attention pooling, use standard pooling
- `--hidden-dim`: Hidden dimension for GNN layers (default: 256)
- `--edge-dim`: Hidden dimension for edge scoring network (default: 128)
- `--dropout`: Dropout rate (default: 0.2)

#### Lambda Schedule

LENS uses a three-phase lambda schedule:

1. **Warmup (epochs 0 to T_w)**: Linear increase from 0 to λ₀
2. **Ramp (epochs T_w to T_w + T_r)**: Linear increase from λ₀ to 2λ₀
3. **Plateau (epochs > T_w + T_r)**: Constant at 2λ₀

Temperature uses cosine annealing from `initial-temp` to 0.67.

### Bayesian Optimization

To automatically find optimal parameters balancing accuracy and sparsity:
```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --batch-size 1 \
  --epochs 80 \
  --run-bayesian-opt \
  --n-trials 30 \
  --target-sparsity 0.7 \
  --sparsity-penalty 5.0
```

#### Optimization Parameters

- `--run-bayesian-opt`: Flag to activate Bayesian optimization
- `--n-trials`: Number of optimization trials to run (default: 50)
- `--target-sparsity`: Target sparsity rate to aim for (0.0-1.0, where 0.7 means 70% of edges pruned)
- `--sparsity-penalty`: Weight for sparsity deviation penalty in objective function (higher = stricter adherence to target)

The optimization objective is: **O = Acc_val - λ_penalty · |Sparsity - Target|**

Bayesian optimization automatically searches over the following hyperparameters:
- `lambda_reg`: [0.0001, 0.05] - Regularization strength
- `lambda_density`: [0.01, 0.1] - Density loss weight
- `target_density`: [0.2, 0.5] - Target edge retention rate
- `warmup_epochs`: [5, 20] - Warmup duration
- `ramp_epochs`: [10, 30] - Ramp duration
- `initial_temp`: [2.0, 10.0] - Initial temperature for annealing
- `dropout`: [0.1, 0.4] - Dropout rate
- `learning_rate`: [0.0001, 0.01] - Learning rate

**Note**: L0 parameters (gamma=-0.1, zeta=1.1, beta=0.66) use standard values from literature and are not optimized.



### Tips for Hyperparameter Tuning

1. **Start with default parameters** for initial experiments
2. **Adjust `lambda-reg`** to control overall sparsity level:
   - Lower values (0.0001-0.001) → less aggressive pruning
   - Higher values (0.005-0.01) → more aggressive pruning
3. **Tune `target-density`** based on your dataset:
   - Dense graphs: 0.20-0.30 (keep 20-30% of edges)
   - Sparse graphs: 0.40-0.50 (keep 40-50% of edges)
4. **Use `lambda-density`** to enforce target density:
   - Start with 0.003 and increase if density deviates from target
5. **Increase `warmup-epochs`** if training is unstable early on
6. **Try different L0 methods**:
   - `hard-concrete`: Best for most cases (stable, differentiable)
   - `arm`: For true binary sampling during training
   - `ste`: Simplest, but may have higher variance

### Step 4: Testing

The testing script automatically detects all model configurations from the checkpoint and evaluates the trained model with bootstrap statistical analysis, providing confidence intervals for ROC/PR curves and comprehensive performance metrics.

### Basic Usage
```bash
python test_lens2_auto.py \
  --model-path /path/to/best_model.pt \
  --test-data /path/to/test_list.txt \
  --data-root /path/to/data \
  --output-dir test_results
```

### Advanced Usage with All Options
```bash
python test_lens2_auto.py \
  --model-path ./results/lens_hc/fold1/phase_models/overall_best_model.pth \
  --test-data ./data/TCGA-LUNG/test.txt \
  --data-root ./data/TCGA-LUNG/simclr_files \
  --output-dir ./test_results \
  --n-bootstrap 10000 \
  --confidence-level 0.95 \
  --analyze-sparsity \
  --class-names "Normal,LUAD,LUSC"
```

### Key Parameters

**Required:**
- `--model-path`: Path to trained model checkpoint (`.pt` or `.pth` file)
- `--test-data`: Text file with test sample IDs (format: `filename\tlabel`)
- `--data-root`: Root directory containing graph data

**Optional:**
- `--n-bootstrap`: Number of bootstrap iterations for confidence intervals (default: 10000, set to 0 to skip)
- `--confidence-level`: Confidence level for bootstrap CI (default: 0.95)
- `--analyze-sparsity`: Flag to perform edge sparsity analysis
- `--output-dir`: Directory for saving test results (default: `test_results`)
- `--class-names`: Comma-separated class names for plots (e.g., "Normal,LUAD,LUSC")

**For Deployment/Comparison (Optional):**
- `--use-top-k`: Binarize edges using top-k selection (for computational budget evaluation)
- `--top-k-ratio`: Ratio of top edges to keep when using top-k (default: 0.30)

**Use this for:**
- Generating heatmap visualizations showing graded edge importance
- Understanding which edges the model considers most important
- Creating publication-quality figures with smooth color gradients
- Interpreting model decisions on individual samples

**Output:** Edge weights preserve their continuous values [0, 1], enabling smooth heatmaps.


### Step 5: Visualization


Generate heatmaps overlaying learned edge weights on whole slide images (WSIs) to visualize model attention and analyze pruning effects.

### Usage

    python LENS_heatmap.py \
      --wsi-path /path/to/slide.svs \
      --patch-info-path /path/to/c_idx.txt \
      --pruned-adj-path /path/to/pruned_adj.pt \
      --original-adj-path /path/to/original_adj.pt \
      --output-dir heatmap_results

### Key Parameters

- `--wsi-path`: Path to whole slide image (.svs file)
- `--patch-info-path`: Path to patch coordinates file (c_idx.txt)  
- `--pruned-adj-path`: Path to pruned adjacency matrix from testing
- `--original-adj-path`: Path to original adjacency matrix (optional, for comparison)
- `--output-dir`: Output directory for visualization results

### Output

The script generates three types of visualizations:

#### Heatmap Overlays
    heatmap_results/pruned_heatmap/
    ├── sample_edge_weight_heatmap.png    # WSI with JET colormap overlay
    ├── sample_combined.png               # Original and heatmap side-by-side
    └── sample_weight_distribution.png    # Edge weight histogram

#### Comparison Analysis (if original provided)
    heatmap_results/comparison/
    ├── sample_weight_comparison.png      # Original vs pruned distributions
    └── sample_importance_diff.png        # Node importance changes

#### Statistics
Edge retention statistics and weight distributions are printed to console and saved in output files.

The heatmap uses JET colormap where red indicates high edge connectivity (important regions) and blue indicates low connectivity. This visualization reveals which tissue areas the model considers most important for classification.


<img src="https://github.com/user-attachments/assets/eee7e786-9605-4c5a-b032-c5abd81998db" width="400"/>


## 📊 Results

LENS demonstrates robust discriminative power while utilizing only ~30% of the graph edges, indicating efficient extraction of relevant structural information.

<div align="center">
<img width="3000" height="1375" alt="cptac_heatmaps" src="https://github.com/user-attachments/assets/242d13fe-1b5c-4fdc-914b-cca930d2308a" />
</div>




## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


