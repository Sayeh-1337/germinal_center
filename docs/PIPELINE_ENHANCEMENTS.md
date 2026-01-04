# Pipeline Enhancements: Advanced Spatial and Chromatin Analysis

This document describes the enhanced analysis modules added to the germinal center chromatin analysis pipeline. These enhancements provide deeper biological insights through advanced spatial analysis, multi-scale chromatin features, and cell cycle-aware classification.

---

## Table of Contents

1. [Overview](#overview)
2. [Spatial Graph Analysis](#spatial-graph-analysis)
3. [Multi-scale Chromatin Features](#multi-scale-chromatin-features)
4. [Cell Cycle State Inference](#cell-cycle-state-inference)
5. [Relative and Interaction Features](#relative-and-interaction-features)
6. [Usage Guide](#usage-guide)
7. [Biological Rationale](#biological-rationale)
8. [Output Files](#output-files)

---

## Overview

The enhanced pipeline addresses key limitations in the original feature extraction:

| Original Pipeline | Enhanced Pipeline |
|-------------------|-------------------|
| Distance-based spatial analysis | Graph-based network analysis |
| Single-scale GLCM textures | Multi-scale wavelet decomposition |
| No cell cycle consideration | Cell cycle state inference |
| Absolute feature values | Relative neighborhood features |

### New Modules

```
src/
├── analysis/
│   └── spatial_graph.py      # Graph-based spatial analysis
└── features/
    ├── multiscale_features.py # Wavelet & fractal analysis
    ├── cell_cycle.py          # Cell cycle inference
    └── relative_features.py   # Neighborhood-relative features
```

---

## Spatial Graph Analysis

**Module:** `src/analysis/spatial_graph.py`

### Cell Interaction Graph

Builds a spatial graph where nodes represent cells and edges represent spatial proximity.

```python
from src.analysis.spatial_graph import build_cell_interaction_graph

result = build_cell_interaction_graph(
    spatial_coords=df,           # DataFrame with centroid-0, centroid-1
    cell_types=cell_labels,      # Optional: Series with cell type labels
    k_neighbors=10,              # Number of nearest neighbors
    distance_threshold=50.0      # Maximum edge distance (pixels)
)
```

**Graph Features Computed:**

| Feature | Description | Biological Meaning |
|---------|-------------|-------------------|
| `degree_centrality` | Normalized neighbor count | Local connectivity |
| `degree` | Raw neighbor count | Crowdedness |
| `clustering_coefficient` | Local cliquishness | Microenvironment cohesion |
| `betweenness_centrality` | Bridge cell score | Cells connecting zones |
| `pagerank` | Network influence | Signaling hubs |
| `avg_neighbor_degree` | Mean neighbor connectivity | Neighborhood structure |
| `same_type_neighbor_fraction` | Homophily measure | Zonal segregation |

### Moran's I Spatial Autocorrelation

Tests whether chromatin features cluster spatially, indicating microenvironmental influence.

```python
from src.analysis.spatial_graph import compute_spatial_autocorrelation

results = compute_spatial_autocorrelation(
    features=feature_df,
    spatial_coords=coord_df,
    feature_names=['area', 'entropy', 'contrast'],
    k_neighbors=8
)

# Returns dict with I statistic, z-score, p-value per feature
```

**Interpretation:**
- **I > 0 (significant):** Feature shows spatial clustering → environment-driven
- **I ≈ 0:** Random spatial distribution → cell-intrinsic variation
- **I < 0 (significant):** Feature shows spatial dispersion

### Voronoi Tessellation Features

Computes Voronoi cell properties for each nucleus.

```python
from src.analysis.spatial_graph import compute_voronoi_features

voronoi_df = compute_voronoi_features(
    spatial_coords=coord_df,
    cell_types=cell_labels  # Optional
)
```

**Features:**

| Feature | Description | Biological Meaning |
|---------|-------------|-------------------|
| `voronoi_area` | Voronoi cell area | Inverse of local density |
| `voronoi_perimeter` | Cell boundary length | Shape complexity |
| `voronoi_n_neighbors` | Coordination number | Direct contacts |
| `voronoi_compactness` | Isoperimetric quotient | Packing regularity |
| `voronoi_same_type_neighbor_fraction` | Same-type adjacency | Zonal mixing |

**Expected Patterns:**
- **DZ cells:** Smaller Voronoi area (densely packed, proliferative)
- **LZ cells:** Larger Voronoi area (less dense, more T-cell interaction space)

### Local Cell Density

Computes cell density at multiple spatial scales.

```python
from src.analysis.spatial_graph import compute_local_cell_density

density_df = compute_local_cell_density(
    spatial_coords=coord_df,
    radii=[25, 50, 100]  # Pixels
)
```

---

## Multi-scale Chromatin Features

**Module:** `src/features/multiscale_features.py`

### Wavelet-based Texture Analysis

Decomposes chromatin texture at multiple scales using discrete wavelet transform.

```python
from src.features.multiscale_features import extract_wavelet_chromatin_features

wavelet_df = extract_wavelet_chromatin_features(
    intensity_image=nucleus_intensity,
    mask=nucleus_mask,
    wavelet='db4',    # Daubechies 4 wavelet
    levels=3          # Decomposition levels
)
```

**Features per Level:**

| Level | Scale | Features | Biological Process |
|-------|-------|----------|-------------------|
| 1 | Fine (1-2 px) | `wavelet_*_energy_l1` | Heterochromatin boundaries |
| 2 | Medium (5-10 px) | `wavelet_*_energy_l2` | TAD-like structures |
| 3 | Coarse (20-50 px) | `wavelet_*_energy_l3` | Nuclear compartments |

**Additional Features:**
- `wavelet_anisotropy_l*`: Horizontal vs vertical texture direction
- `wavelet_fine_coarse_ratio`: Texture complexity measure

### Fractal Dimension Analysis

Measures self-similarity and complexity of chromatin distribution.

```python
from src.features.multiscale_features import compute_fractal_dimension

result = compute_fractal_dimension(
    intensity_image=nucleus_intensity,
    mask=nucleus_mask,
    method='box_counting'
)
# Returns: {'fractal_dimension': 1.65, 'lacunarity': 0.23}
```

**Interpretation:**
- **Range:** 1.0 (smooth boundary) to 2.0 (space-filling)
- **DZ cells:** Higher fractal dimension (more complex, active chromatin)
- **LZ cells:** Lower fractal dimension (more ordered, condensed)

### Chromatin Domain Analysis

Segments and quantifies individual chromatin domains.

```python
from src.features.multiscale_features import analyze_chromatin_domain_sizes

domain_df = analyze_chromatin_domain_sizes(
    intensity_image=nucleus_intensity,
    mask=nucleus_mask,
    threshold_method='otsu',
    min_domain_size=10
)
```

**Features:**

| Feature | Description |
|---------|-------------|
| `n_chromatin_domains` | Number of distinct domains |
| `domain_area_mean` | Mean domain size |
| `domain_area_fraction` | Fraction of nucleus as domains |
| `domain_circularity_mean` | Domain shape regularity |
| `domain_size_power_exponent` | Size distribution scaling |

### Radial Intensity Profile

Measures intensity from nuclear center to periphery.

```python
from src.features.multiscale_features import compute_radial_intensity_profile

radial_df = compute_radial_intensity_profile(
    intensity_image=nucleus_intensity,
    mask=nucleus_mask,
    n_bins=5
)
```

**Biological Insight:**
- Heterochromatin localizes at nuclear periphery
- Active chromatin in nuclear interior
- `radial_intensity_gradient`: Positive = peripheral enrichment

---

## Cell Cycle State Inference

**Module:** `src/features/cell_cycle.py`

### Why Cell Cycle Matters

Cell cycle state is a major confounder for DZ/LZ classification:
- **DZ cells:** Actively proliferating (more S/G2 cells)
- **LZ cells:** Quiescent (more G0/G1 cells)

Cell cycle affects nuclear size, chromatin texture, and intensity distribution.

### Inference Methods

#### 1. Threshold-based (Rule-based)

```python
from src.features.cell_cycle import infer_cell_cycle_state

predictions = infer_cell_cycle_state(
    features=feature_df,
    method='threshold',
    area_col='area'
)
```

**Heuristics:**
- **G0/G1:** Small-medium area, low texture heterogeneity
- **S phase:** Medium-large area, high heterogeneity (replication foci)
- **G2/M:** Large area, intermediate heterogeneity, condensed chromatin

#### 2. Clustering-based (Data-driven)

```python
predictions = infer_cell_cycle_state(
    features=feature_df,
    method='clustering'
)
```

Uses Gaussian Mixture Model to identify 3 clusters, ordered by nuclear area.

### Cell Cycle-specific Features

```python
from src.features.cell_cycle import compute_cell_cycle_features

cc_features = compute_cell_cycle_features(
    intensity_image=nucleus_intensity,
    mask=nucleus_mask
)
```

| Feature | S-phase Indicator |
|---------|-------------------|
| `n_bright_foci` | Replication foci count |
| `bright_foci_fraction` | Foci coverage |
| `chromatin_condensation_score` | Local variance (G2/M) |
| `intensity_bimodality` | S-phase distribution |

### Adjusting for Cell Cycle

```python
from src.features.cell_cycle import compute_cell_cycle_adjusted_features

adjusted = compute_cell_cycle_adjusted_features(
    features=feature_df,
    cell_cycle_predictions=predictions
)
# Creates *_cc_adjusted columns with cell cycle effects regressed out
```

---

## Relative and Interaction Features

**Module:** `src/features/relative_features.py`

### Motivation

Absolute feature values vary with:
- Imaging conditions (exposure, gain)
- Staining batch effects
- Tissue section thickness

Relative features normalize to local context, improving robustness.

### Relative Features

```python
from src.features.relative_features import compute_relative_features

relative_df = compute_relative_features(
    features=feature_df,
    spatial_coords=coord_df,
    k_neighbors=10
)
```

**For each feature X:**
- `X_relative`: X / mean(neighbor X values)
- `X_diff_from_neighbors`: X - mean(neighbor X values)
- `X_zscore_local`: (X - mean) / std of neighbors

### Interaction Features

Captures cell-cell relationships and microenvironmental context.

```python
from src.features.relative_features import compute_interaction_features

interaction_df = compute_interaction_features(
    features=feature_df,
    spatial_coords=coord_df,
    cell_types=cell_labels,
    k_neighbors=10
)
```

**Features:**
- `dist_nearest_neighbor`: Crowdedness
- `dist_nearest_{cell_type}`: Distance to specific cell type
- `neighbor_fraction_{cell_type}`: Composition of neighborhood
- `{feature}_mean_{cell_type}_neighbors`: Mean feature of typed neighbors
- `{feature}_neighbor_variance`: Local heterogeneity

### Spatial Gradients

Estimates local spatial derivatives of features.

```python
from src.features.relative_features import compute_spatial_gradients

gradient_df = compute_spatial_gradients(
    features=feature_df,
    spatial_coords=coord_df,
    k_neighbors=10
)
```

**Features:**
- `{feature}_gradient_x/y`: Directional gradients
- `{feature}_gradient_magnitude`: Gradient strength
- `{feature}_gradient_direction`: Gradient angle

### Boundary Proximity

Computes distance to DZ/LZ boundary.

```python
from src.features.relative_features import compute_boundary_proximity

boundary_df = compute_boundary_proximity(
    spatial_coords=coord_df,
    zone_labels=dz_lz_labels
)
```

---

## Usage Guide

### Command Line Interface

**Enhanced Feature Extraction:**

```bash
# Full enhanced extraction
gc-pipeline extract-enhanced \
    --raw-images data/processed/dapi \
    --labels data/processed/segmented_nucleus \
    --output data/processed/features \
    --k-neighbors 10 \
    --wavelet-levels 3 \
    --density-radii 25 50 100

# Selective extraction (disable specific modules)
gc-pipeline extract-enhanced \
    --raw-images data/processed/dapi \
    --labels data/processed/segmented_nucleus \
    --output data/processed/features \
    --no-multiscale \
    --no-cell-cycle
```

### Python API

```python
from cli.commands.extract_enhanced import extract_enhanced_features

extract_enhanced_features(
    raw_images_dir='data/processed/dapi',
    labels_dir='data/processed/segmented_nucleus',
    output_dir='data/processed/features',
    extract_multiscale=True,
    extract_cell_cycle=True,
    extract_spatial_graph=True,
    extract_relative=True,
    k_neighbors=10,
    wavelet_levels=3,
    density_radii=[25, 50, 100]
)
```

### Individual Module Usage

```python
# Combine modules as needed
from src.analysis.spatial_graph import extract_all_spatial_graph_features
from src.features.multiscale_features import extract_all_multiscale_features
from src.features.cell_cycle import infer_cell_cycle_state
from src.features.relative_features import extract_all_relative_features

# Spatial features (image-level)
spatial_feat = extract_all_spatial_graph_features(
    spatial_coords=coord_df,
    cell_types=cell_types,
    features=base_features
)

# Multi-scale features (per-nucleus)
for prop in regionprops(labels, intensity):
    ms_feat = extract_all_multiscale_features(
        prop.intensity_image,
        prop.image
    )

# Cell cycle inference
cc_predictions = infer_cell_cycle_state(all_features)

# Relative features
rel_feat = extract_all_relative_features(
    all_features,
    coord_df,
    cell_types=cell_types,
    zone_labels=dz_lz_labels
)
```

---

## Biological Rationale

### Why These Enhancements Matter

#### 1. Spatial Graph Analysis
The germinal center is organized as a network:
- T-B cell interactions drive selection
- B-B clustering in proliferative zones
- Graph metrics capture this network topology

**Key insight:** Betweenness centrality identifies cells that bridge DZ and LZ—potential transitioning cells.

#### 2. Multi-scale Analysis
Chromatin organization operates at multiple scales:
- **Nucleosome level:** Not resolvable in standard microscopy
- **Chromatin domains (100nm-1μm):** Captured by fine wavelet levels
- **TAD-like structures (1-5μm):** Medium wavelet levels
- **Compartments (>5μm):** Coarse levels and radial profiles

**Key insight:** DZ cells show higher wavelet energy at fine scales (more open chromatin).

#### 3. Cell Cycle Awareness
DZ and LZ differ in proliferation rate:
- DZ: High proliferation → more S/G2 cells
- LZ: Low proliferation → more G0/G1 cells

**Key insight:** Adjusting for cell cycle prevents confounding in DZ/LZ classification.

#### 4. Relative Features
Microenvironment influences chromatin state:
- T-cell contact may alter B-cell chromatin
- Local cytokine gradients affect nuclear organization

**Key insight:** Relative features capture whether a cell differs from its neighbors, regardless of imaging batch effects.

---

## Output Files

### Directory Structure

```
output/
├── chrometric_features/     # Per-image feature CSVs
├── spatial_coordinates/     # Centroid coordinates
├── consolidated_features/
│   ├── nuc_features.csv        # Standard chrometric features
│   ├── spatial_coordinates.csv # All spatial data
│   └── all_features_merged.csv # Combined standard + enhanced
└── enhanced_features/
    └── enhanced_features.csv   # All enhanced features
```

### Feature Summary

| Category | Approx. # Features | File Location |
|----------|-------------------|---------------|
| Standard chrometric | ~80 | `nuc_features.csv` |
| Wavelet texture | ~20 | `enhanced_features.csv` |
| Fractal/domain | ~12 | `enhanced_features.csv` |
| Cell cycle | ~15 | `enhanced_features.csv` |
| Graph/Voronoi | ~15 | `enhanced_features.csv` |
| Relative | ~varies | `enhanced_features.csv` |

### Merged Output

The `all_features_merged.csv` file combines all features by `nuc_id`, providing a single comprehensive feature matrix for downstream analysis.

---

## Dependencies

The enhanced modules require:

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.4.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
networkx>=2.5          # For graph analysis
PyWavelets>=1.1.0      # For wavelet features
```

All dependencies are included in the existing `requirements.txt`.

---

## References

1. **Germinal Center Biology:**
   - Victora & Nussenzweig (2012). Germinal Centers. *Annu Rev Immunol*.

2. **Spatial Statistics:**
   - Moran (1950). Notes on Continuous Stochastic Phenomena. *Biometrika*.

3. **Wavelet Analysis:**
   - Mallat (1989). A Theory for Multiresolution Signal Decomposition. *IEEE Trans PAMI*.

4. **Fractal Dimension:**
   - Lopes & Betrouni (2009). Fractal and Multifractal Analysis. *Med Image Anal*.

5. **Cell Cycle Imaging:**
   - Held et al. (2010). CellCognition: time-resolved phenotype annotation. *Nat Methods*.

