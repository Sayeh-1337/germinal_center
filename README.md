# Germinal Centers

The repository contains the code used to run the analysis of the chromatin states of the cell populations in germinal
centers and selected regions of interest which is discussed in:

> [**TO BE ADDED**]()

<p align="center" width="100%">
  <b>Computational analysis pipeline</b> <br>
    <img width="66%" src="https://github.com/GVS-Lab/germinal_center/blob/bf3cfc13d338a21edf7db25b881ae6f60ea21a87/gc_overview.png">
</p>

----

# System requirements

The code has been developed and executed on a Thinkpad P1 mobile work station running Ubuntu 18.04 LTS with a Intel(R)
Core(TM) i7-9750H CPU with 2.60 GHz, 32GB RAM and a Nvidia P2000 GPU. Note that the code can also be run for machines
without a GPU or with less available RAM.

## Installation

To install the code, please clone the repository and install the required software libraries and packages.

### Quick Installation (Recommended)

This is the fastest method - creates a minimal conda environment and uses pip for package installation:

```bash
git clone https://github.com/GVS-Lab/germinal_center.git
cd germinal_center
conda env create -f environment.yml
conda activate gc
pip install -r requirements_pip.txt
pip install -e .
```

**Note:** Chromatin feature extraction (formerly from the `nmco` library) is now built-in. No external dependencies required.

### Alternative: Using requirements.txt

If you prefer to use the original `requirements.txt` file:

```bash
git clone https://github.com/GVS-Lab/germinal_center.git
cd germinal_center
conda create --name gc python=3.8
conda activate gc
pip install -r requirements.txt
```

**Note:** The `environment.yml` file creates a minimal environment with just Python and pip, then all packages are installed via pip which is much faster than conda's dependency resolver. Chromatin feature extraction is now built into the package (`src/features/`).

## Data resouces (Optional)

Intermediate results of the analysis can be obtained from
our [Google Drive here](https://drive.google.com/drive/folders/1HszNjSRFI2x25mEDQo-a_rKpemwtJZ4C?usp=sharing) but can
also be produced using the steps described below to reproduce the results of the paper. If you want to use and/or adapt
the code to run another analysis, the data is not required neither.

---

# Command-Line Interface (CLI)

The pipeline can be run using the `gc-pipeline` command-line tool, which provides a flexible and reproducible way to analyze germinal center data.

## Quick Start

```bash
# Generate a default configuration file
gc-pipeline init --name my_dataset --output configs/my_config.yaml

# Run the full pipeline
gc-pipeline pipeline --config configs/dataset1_config.yaml

# Run specific steps only
gc-pipeline pipeline --config configs/dataset1_config.yaml --steps preprocess segment
```

## CLI Commands

### `gc-pipeline init`

Generate a default configuration file:

```bash
gc-pipeline init --name dataset1 --output configs/dataset1_config.yaml
```

Options:
- `--name, -n`: Dataset name (default: dataset1)
- `--output, -o`: Output path for config file

### `gc-pipeline pipeline`

Run the full analysis pipeline using a configuration file:

```bash
gc-pipeline pipeline --config configs/dataset1_config.yaml
gc-pipeline pipeline --config configs/dataset1_config.yaml --steps preprocess segment extract
gc-pipeline pipeline --config configs/dataset1_config.yaml --output-dir custom/output/path
```

Options:
- `--config, -c`: Path to YAML configuration file (required)
- `--steps`: Specific steps to run (choices: preprocess, segment, extract, analyze)
- `--output-dir`: Override output directory from config
- `--resume`: Resume from last interrupted state (skips completed files)
- `--reset`: Reset pipeline state and start fresh
- `--status`: Show pipeline progress status and exit

#### Resume Feature

The pipeline tracks progress and can resume from an interrupted state:

```bash
# Check current progress
gc-pipeline pipeline --config configs/dataset1_config.yaml --status

# Resume from where you left off
gc-pipeline pipeline --config configs/dataset1_config.yaml --resume

# Reset and start fresh
gc-pipeline pipeline --config configs/dataset1_config.yaml --reset
```

### `gc-pipeline preprocess`

Preprocess images (extract channels and normalize):

```bash
gc-pipeline preprocess \
    --input data/images/raw/merged \
    --output data/images/processed \
    --channels dapi:1 cd3:2 aicda:3 \
    --quantiles 0.01 0.998 \
    --mask-dir data/images/masks
```

Options:
- `--input, -i`: Input directory with raw images (required)
- `--output, -o`: Output directory for processed images (required)
- `--channels`: Channels to extract as name:index pairs (default: dapi:1 cd3:2 aicda:3)
- `--quantiles`: Quantiles for normalization (default: 0.01 0.998)
- `--mask-dir`: Directory with mask images for normalization
- `--skip-normalize`: Skip quantile normalization step

### `gc-pipeline segment`

Segment nuclei using StarDist:

```bash
gc-pipeline segment \
    --input data/images/processed/dapi_scaled \
    --output-labels data/images/segmented \
    --output-rois data/images/rois \
    --prob-thresh 0.43
```

Options:
- `--input, -i`: Input directory with DAPI images (required)
- `--output-labels, -ol`: Output directory for label images (required)
- `--output-rois, -or`: Output directory for ImageJ ROIs (optional)
- `--prob-thresh`: Probability threshold for StarDist (default: 0.43)
- `--no-pretrained`: Use custom model instead of pretrained
- `--model-dir`: Directory containing custom model
- `--model-name`: Name of custom model

### `gc-pipeline extract`

Extract chrometric features:

```bash
gc-pipeline extract \
    --raw-images data/images/processed/dapi_scaled \
    --labels data/images/segmented \
    --output data/features \
    --proteins data/images/processed/cd3 data/images/processed/aicda \
    --cell-segmentation \
    --dilation-radius 10
```

Options:
- `--raw-images, -r`: Directory with raw DAPI images (required)
- `--labels, -l`: Directory with segmented label images (required)
- `--output, -o`: Output directory for features (required)
- `--proteins`: Directories with protein images for intensity measurement
- `--cell-segmentation`: Perform cell segmentation by dilation
- `--dilation-radius`: Radius for dilation in pixels (default: 10)
- `--extract-spatial`: Extract spatial coordinates (default: True)

### `gc-pipeline analyze`

Run analysis on extracted features:

```bash
gc-pipeline analyze \
    --features data/features/consolidated_features \
    --output data/analysis \
    --analysis-type cell_type_detection cell_type tcell_interaction boundary umap markers \
    --metadata data/metadata.csv \
    --random-seed 1234
```

#### Analysis Types

| Analysis Type | Alias | Description |
|---------------|-------|-------------|
| `cell_type_detection` | `detect_cells` | GMM-based cell type detection using AICDA/CD3 marker intensities |
| `cell_type` | `classification` | Random Forest classification of DZ vs LZ B-cells (10-fold CV) |
| `tcell_interaction` | `tcell` | T-cell influence zone analysis (15/30 μm radius) |
| `boundary` | `dz_lz_boundary` | DZ/LZ boundary distance and proximity analysis |
| `correlation` | - | Feature-metadata correlation with permutation tests and FDR |
| `umap` | `visualization` | UMAP embedding with HDBSCAN clustering |
| `markers` | `differential` | Welch's t-test based marker feature detection |
| `all` | - | Run all analyses |

#### Options

**Basic Options:**
- `--features, -f`: Directory with feature CSV files (required)
- `--output, -o`: Output directory for analysis results (required)
- `--analysis-type`: Types of analysis to run (see table above)
- `--metadata`: Path to metadata CSV file (for correlation analysis)
- `--correlation-threshold`: Pearson correlation threshold for feature filtering (default: 0.8)
- `--random-seed`: Random seed for reproducibility (default: 1234)

**Spatial Analysis Options:**
- `--pixel-size`: Pixel size in microns (default: 0.3225)
- `--contact-radius`: T-cell physical contact radius in microns (default: 15.0)
- `--signaling-radius`: T-cell signaling radius in microns (default: 30.0)
- `--border-threshold`: Threshold for DZ/LZ border proximity (default: 0.4)

**Visualization & Statistical Options:**
- `--no-plots`: Skip generating visualization plots
- `--n-permutations`: Number of permutations for statistical tests (default: 10000)

#### Example: Full Analysis

```bash
# Run all analyses with custom parameters
gc-pipeline analyze \
    --features data/features/consolidated_features \
    --output data/analysis \
    --analysis-type all \
    --pixel-size 0.3225 \
    --contact-radius 15.0 \
    --signaling-radius 30.0 \
    --border-threshold 0.4 \
    --n-permutations 100000 \
    --random-seed 42
```

## Configuration File Format

Configuration files are in YAML format. Example:

```yaml
dataset_name: "dataset1"
input_dir: "data/dataset1/images/raw/merged"
output_dir: "data/dataset1/processed"

preprocessing:
  channels:
    dapi: 1
    cd3: 2
    aicda: 3
  quantile_normalization:
    enabled: true
    quantiles: [0.01, 0.998]
    mask_dir: "data/dataset1/images/germinal_center_anno"

segmentation:
  prob_thresh: 0.43
  use_pretrained: true

feature_extraction:
  cell_segmentation:
    enabled: true
    dilation_radius: 10
  extract_spatial: true
  protein_measurement:
    enabled: true
    proteins:
      - name: "cd3"
        image_dir: "data/dataset1/processed/cd3"
      - name: "aicda"
        image_dir: "data/dataset1/processed/aicda"

analysis:
  # Global analysis parameters
  correlation_threshold: 0.8
  random_seed: 1234
  generate_plots: true
  n_permutations: 10000
  
  # Spatial analysis parameters
  spatial_parameters:
    pixel_size: 0.3225  # microns per pixel
    contact_radius: 15.0  # T-cell contact radius (microns)
    signaling_radius: 30.0  # T-cell signaling radius (microns)
    border_threshold: 0.4  # DZ/LZ border proximity threshold
  
  # Analysis types to run
  analyses:
    - type: "cell_type_detection"  # GMM-based marker detection
      enabled: true
    - type: "cell_type"  # DZ vs LZ classification
      enabled: true
    - type: "tcell_interaction"  # T-cell proximity analysis
      enabled: true
    - type: "boundary"  # DZ/LZ boundary analysis
      enabled: true
    - type: "umap"  # UMAP visualization
      enabled: true
    - type: "markers"  # Differential expression
      enabled: true
    - type: "correlation"  # Metadata correlation
      enabled: false
```

## Visualization Tools

### Segmentation Visualization

A standalone script is provided to visualize nuclear segmentation results:

```bash
# Interactive viewer (recommended)
python scripts/visualize_segmentation.py \
    --image-dir data/dataset1/processed/dapi_scaled \
    --label-dir data/dataset1/processed/segmented_nucleus \
    --interactive

# Save all visualizations to a folder
python scripts/visualize_segmentation.py \
    --image-dir data/dataset1/processed/dapi_scaled \
    --label-dir data/dataset1/processed/segmented_nucleus \
    --output-dir outputs/segmentation_viz

# View a single image
python scripts/visualize_segmentation.py \
    --image data/dataset1/processed/dapi_scaled/1.tif \
    --labels data/dataset1/processed/segmented_nucleus/1.tif
```

**Interactive Controls:**
- `→` / `n`: Next image
- `←` / `p`: Previous image
- `+` / `-`: Adjust overlay transparency
- `b`: Toggle cell boundaries
- `s`: Save current view
- `q`: Quit

### Analysis Visualizations

The pipeline automatically generates comprehensive visualizations in each analysis directory:

| Analysis | Generated Plots |
|----------|-----------------|
| **Cell Type Detection** | `cell_type_distribution.png` - Bar chart of DZ/LZ/T-cell counts |
| **Cell Type Classification** | `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `marker_violin_plots.png` |
| **T-cell Interaction** | `tcell_influence_distribution.png`, `spatial_tcell_influence_*.png`, `tcell_zones_overlay_*.png` |
| **Boundary Analysis** | `spatial_border_proximity_*.png`, `boundary_classification_results.csv` |
| **UMAP** | `umap_clusters.png`, `umap_by_image.png`, `umap_by_celltype.png` |

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cli --cov-report=html
```

---

# Documentation

For a detailed scientific explanation of the pipeline methodology, see:

📖 **[docs/SCIENTIFIC_METHODOLOGY.md](docs/SCIENTIFIC_METHODOLOGY.md)**

This document covers:
- Germinal center biology background
- Step-by-step pipeline explanation
- Feature extraction methodology (200+ chrometric features)
- Statistical methods (Welch's t-test, FDR correction)
- Interpretation of results

---

# Reproducing the paper results

## Using the CLI (Recommended)

### Dataset 1: B-cell population analysis

```bash
# Run full pipeline for dataset 1
gc-pipeline pipeline --config configs/dataset1_config.yaml
```

### Dataset 3: Correlation analysis

```bash
# Run full pipeline for dataset 3
gc-pipeline pipeline --config configs/dataset3_config.yaml
```

## Using Notebooks (Alternative)

### 1. Data preprocessing

The data preprocessing steps quantile-normalize the data, segment individual nuclei and cells as well as measure the
chrometric features described
in [Venkatachalapathy et al. (2020)](https://www.molbiolcell.org/doi/10.1091/mbc.E19-08-0420) for each nucleus and
quantify the associated cellular expression of the proteins stained for in the processed immunofluorescent images. To
preprocess the imaging data for the analysis of the B-cell populations in the germinal centers or the correlation
analysis of the selected microimages please use the notebooks `notebooks/dataset1/feature_generation.ipynb`
or `notebooks/dataset3/data_preprocessing.ipynb` respectively.

### 2. Analysis of the B-cell populations in germinal centers

To run the analysis regarding the different B-cell populations in the light respectively dark zone of the germinal
centers, please use the code provided in the
notebook `notebooks/dataset1/light_vs_darkzone_bcells_and_tcell_interaction.ipynb`.

### 3. Analysis of the gene expression and chromatin signature of cells

Finally, the correlation analysis of the measured dark zone gene expression signatures of the selected RoIs and the
chromatin states corresponding to those can be run using the code
in `notebooks/dataset3/chrometric_correlation_analysis.ipynb`.

---

# Project Structure

```
germinal_center/
├── cli/                          # Command-line interface
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # Configuration handling
│   ├── state.py                  # Pipeline state management (resume)
│   └── commands/                 # CLI commands
│       ├── preprocess.py         # Image preprocessing
│       ├── segment.py            # Nuclear segmentation
│       ├── extract.py            # Feature extraction
│       ├── analyze.py            # Analysis pipeline
│       └── pipeline.py           # Full pipeline runner
├── src/                          # Source modules
│   ├── analysis/                 # Analysis modules (NEW)
│   │   ├── cell_type_detection.py  # GMM-based marker detection
│   │   ├── tcell_interaction.py    # T-cell proximity analysis
│   │   ├── boundary_analysis.py    # DZ/LZ boundary analysis
│   │   ├── visualization.py        # UMAP, plots, etc.
│   │   └── statistical_tests.py    # Welch's t-test, permutation tests
│   ├── batch/                    # Batch processing
│   │   ├── cell_segmentation.py
│   │   ├── extract_features.py
│   │   └── nuclear_segmentation.py
│   ├── features/                 # Chrometric feature extraction (built-in)
│   │   ├── feature_extraction.py   # Main entry point
│   │   ├── global_morphology.py    # Shape/size features
│   │   ├── intensity_features.py   # Intensity distribution features
│   │   ├── texture_features.py     # GLCM texture features
│   │   └── curvature_features.py   # Boundary curvature features
│   └── utils/                    # Utility functions
│       ├── base.py
│       ├── cell_type_detection.py
│       ├── data_processing.py
│       ├── data_viz.py
│       ├── discrimination.py
│       └── preprocess_images.py
├── notebooks/                    # Jupyter notebooks
│   ├── dataset1/                 # Dataset 1 analysis
│   └── dataset3/                 # Dataset 3 analysis
├── configs/                      # Configuration files
│   ├── dataset1_config.yaml
│   └── dataset3_config.yaml
├── scripts/                      # Utility scripts
│   └── visualize_segmentation.py # Segmentation visualization tool
├── docs/                         # Documentation
│   └── SCIENTIFIC_METHODOLOGY.md # Detailed scientific methodology
├── tests/                        # Test suite
│   └── test_pipeline.py
├── environment.yml               # Conda environment
├── requirements.txt              # Original requirements
├── requirements_pip.txt          # Pip requirements
├── setup.py                      # Package setup
└── README.md
```

---

# How to cite

If you use any of the code or resources provided here please make sure to reference the required software libraries if
needed and also cite our work:

**TO BE ADDED**

----
