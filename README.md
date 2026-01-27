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

python tiling.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```

### Step 2: Graph Construction
```python
# Create a graph dataset from the tiled images

python graph_construction.py  --weights "path_to_pretrained_feature_extractor" --dataset "path_to_patches" --output "../graphs"
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

### Step 4: Testing



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



## 📊 Results

LENS demonstrates robust discriminative power while utilizing only ~30% of the graph edges, indicating efficient extraction of relevant structural information.

<div align="center">
<img width="3000" height="1375" alt="cptac_heatmaps" src="https://github.com/user-attachments/assets/242d13fe-1b5c-4fdc-914b-cca930d2308a" />
</div>




## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


