# Germinal Center Analysis Pipeline: Scientific Methodology

## A Comprehensive Tutorial on Nuclear Chromatin Analysis for B-cell Zone Classification

---

## Table of Contents

1. [Introduction: The Biological Question](#1-introduction-the-biological-question)
2. [Germinal Center Biology](#2-germinal-center-biology)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Step 1: Image Preprocessing](#4-step-1-image-preprocessing)
5. [Step 2: Nuclear Segmentation](#5-step-2-nuclear-segmentation)
6. [Step 3: Feature Extraction](#6-step-3-feature-extraction)
7. [Step 4: Cell Type Detection](#7-step-4-cell-type-detection)
8. [Step 5: Classification Analysis](#8-step-5-classification-analysis)
9. [Step 6: Spatial Analysis](#9-step-6-spatial-analysis)
10. [Statistical Methods](#10-statistical-methods)
11. [Interpretation of Results](#11-interpretation-of-results)
12. [References](#12-references)

---

## 1. Introduction: The Biological Question

### 1.1 Central Hypothesis

**Can nuclear chromatin organization distinguish between functionally distinct B-cell populations within the germinal center?**

This pipeline addresses a fundamental question in immunology: whether the spatial organization of chromatin within the nucleus carries information about a cell's functional state. Specifically, we investigate whether Dark Zone (DZ) and Light Zone (LZ) B-cells—which have distinct functional roles in the immune response—exhibit measurable differences in their nuclear architecture.

### 1.2 Why This Matters

Understanding the chromatin landscape of immune cells has implications for:

- **Basic Immunology**: How does nuclear organization relate to cellular function?
- **Cancer Research**: B-cell lymphomas often arise from germinal center B-cells
- **Vaccine Development**: Optimizing antibody responses requires understanding GC dynamics
- **Diagnostic Pathology**: Potential for automated tissue classification

### 1.3 The Computational Challenge

Traditional immunofluorescence analysis relies on:
- Manual annotation (subjective, time-consuming)
- Simple intensity-based metrics (limited information)

Our approach extracts **200+ quantitative features** from nuclear morphology and chromatin texture, enabling machine learning-based classification and discovery of novel biomarkers.

---

## 2. Germinal Center Biology

### 2.1 What is a Germinal Center?

Germinal centers (GCs) are transient microanatomical structures that form in secondary lymphoid organs (lymph nodes, spleen, tonsils) during an adaptive immune response. They are the sites where B-cells undergo:

1. **Somatic Hypermutation (SHM)**: Random mutations in antibody genes
2. **Affinity Maturation**: Selection of B-cells with higher-affinity antibodies
3. **Class Switch Recombination**: Changing antibody isotype (IgM → IgG, IgA, IgE)

```
                    ┌─────────────────────────────────────┐
                    │         GERMINAL CENTER             │
                    │                                     │
                    │   ┌─────────────┐ ┌─────────────┐  │
                    │   │  DARK ZONE  │ │ LIGHT ZONE  │  │
                    │   │             │ │             │  │
                    │   │ Proliferation│ │  Selection  │  │
                    │   │     SHM     │ │   by FDC    │  │
                    │   │             │ │   T-cells   │  │
                    │   └──────┬──────┘ └──────┬──────┘  │
                    │          │               │         │
                    │          └───────────────┘         │
                    │            Cyclic re-entry         │
                    └─────────────────────────────────────┘
```

### 2.2 Dark Zone vs Light Zone

| Feature | Dark Zone (DZ) | Light Zone (LZ) |
|---------|---------------|-----------------|
| **Location** | Centroblasts near T-cell zone | Centrocytes in follicle center |
| **Function** | Proliferation, SHM | Selection, survival signals |
| **Cell State** | Highly proliferative | Quiescent, receiving signals |
| **Chromatin** | Active transcription, open chromatin | Potentially more condensed |
| **Marker** | AICDA+ (AID enzyme for SHM) | CD83+, CD86+ |

### 2.3 Key Cell Types in Our Analysis

1. **DZ B-cells (AICDA+, CD3-)**: B-cells in the dark zone expressing activation-induced cytidine deaminase
2. **LZ B-cells (AICDA-, CD3-)**: B-cells in the light zone without AICDA expression
3. **T-cells (CD3+)**: T follicular helper cells that provide selection signals

### 2.4 The AICDA Marker

**Activation-Induced Cytidine Deaminase (AID/AICDA)** is the enzyme responsible for:
- Somatic hypermutation
- Class switch recombination

Its expression marks actively mutating B-cells, predominantly in the dark zone. We use AICDA immunofluorescence as a proxy for DZ identity.

---

## 3. Pipeline Overview

### 3.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
│  Multiplexed immunofluorescence images (DAPI + CD3 + AICDA)         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                              │
│  • Channel extraction (DAPI, CD3, AICDA)                            │
│  • Quantile normalization (intensity standardization)               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: NUCLEAR SEGMENTATION                                       │
│  • StarDist 2D deep learning model                                  │
│  • Instance segmentation of individual nuclei                       │
│  • Output: Label masks (each nucleus has unique ID)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE EXTRACTION                                         │
│  A. Chrometric Features (from DAPI)                                 │
│     • Global morphology (size, shape, regularity)                   │
│     • Intensity distribution (heterochromatin patterns)             │
│     • Texture (GLCM, moments)                                       │
│     • Boundary curvature (nuclear envelope irregularity)            │
│  B. Spatial Coordinates (centroid positions)                        │
│  C. Protein Intensities (CD3, AICDA per cell)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: CELL TYPE DETECTION                                        │
│  • Gaussian Mixture Model on protein intensities                    │
│  • Classification: T-cells (CD3+), DZ B-cells (AICDA+), LZ B-cells  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: ANALYSIS                                                   │
│  • DZ vs LZ classification using chromatin features                 │
│  • T-cell interaction analysis (spatial proximity)                  │
│  • Boundary distance analysis (DZ/LZ interface)                     │
│  • Statistical testing (Welch's t-test, FDR correction)             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                             │
│  • Classification accuracy metrics                                  │
│  • Marker feature identification                                    │
│  • Spatial analysis results                                         │
│  • Visualization plots                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Segmentation | StarDist 2D | Deep learning nuclear detection |
| Feature Extraction | Custom (adapted from nmco) | Chromatin texture analysis |
| Classification | Random Forest | Machine learning classification |
| Statistics | SciPy, statsmodels | Hypothesis testing |
| Spatial Analysis | SciPy spatial, KD-trees | Distance computations |

---

## 4. Step 1: Image Preprocessing

### 4.1 Channel Extraction

**Purpose**: Separate the multiplexed immunofluorescence channels for independent analysis.

**Input**: Multi-channel TIFF images with:
- Channel 1: DAPI (DNA stain, nuclear morphology)
- Channel 2: CD3 (T-cell marker)
- Channel 3: AICDA (Dark zone B-cell marker)

**Implementation**:
```python
def extract_channels(image, channel_indices):
    """
    Extract specific channels from a multi-channel image.
    
    Channels are typically:
    - DAPI (index 0 or 1): Nuclear stain
    - CD3 (index 1 or 2): T-cell marker
    - AICDA (index 2 or 3): AID enzyme for DZ identification
    """
    channels = {}
    for name, idx in channel_indices.items():
        channels[name] = image[idx]  # or image[:,:,idx] depending on format
    return channels
```

### 4.2 Quantile Normalization

**Purpose**: Standardize intensity distributions across images to reduce batch effects.

**The Problem**: Different images may have different exposure times, staining intensities, or background levels. This creates artificial variation that can confound analysis.

**The Solution**: Quantile normalization maps intensities to a standard distribution:

```python
def quantile_normalize(image, q_low=0.01, q_high=0.998):
    """
    Normalize image intensities using percentile scaling.
    
    Scientific rationale:
    - q_low (1%): Removes dark noise/background
    - q_high (99.8%): Removes hot pixels/artifacts
    - Result: Comparable intensity scales across images
    
    Mathematical formulation:
    I_norm = (I - P_low) / (P_high - P_low)
    
    Where P_low = percentile(I, q_low*100)
          P_high = percentile(I, q_high*100)
    """
    p_low = np.percentile(image, q_low * 100)
    p_high = np.percentile(image, q_high * 100)
    
    normalized = (image - p_low) / (p_high - p_low)
    return np.clip(normalized, 0, 1)
```

**Why These Percentiles?**
- **1% lower bound**: Excludes camera noise and background
- **99.8% upper bound**: Excludes rare saturated pixels or debris
- This is more robust than min-max normalization

---

## 5. Step 2: Nuclear Segmentation

### 5.1 The Segmentation Challenge

**Problem**: Given a DAPI-stained image, identify every individual nucleus and assign it a unique label.

**Challenges**:
- Touching/overlapping nuclei
- Variable nuclear sizes
- Non-uniform staining
- Background noise

### 5.2 StarDist: Star-Convex Polygon Detection

We use **StarDist 2D**, a deep learning model specifically designed for nuclear segmentation.

**How StarDist Works**:

1. **Star-Convex Representation**: Each nucleus is represented as a star-convex polygon—a set of radial distances from the centroid to the boundary.

```
              Traditional Mask           Star-Convex Representation
              
                  ████████                      r₁  r₂
                ████████████                   ╲  │  ╱
               ██████████████                   ╲ │ ╱  r₃
              ████████████████         r₀ ───────●───────
               ██████████████                   ╱ │ ╲
                ████████████                   ╱  │  ╲
                  ████████                    r₇  r₆  r₅
```

2. **Per-Pixel Prediction**: For each pixel, the network predicts:
   - Object probability (is this pixel inside a nucleus?)
   - Radial distances to the boundary in N directions (typically 32)

3. **Non-Maximum Suppression**: Overlapping predictions are resolved by keeping the highest-probability nucleus.

**Advantages of StarDist**:
- Handles touching nuclei better than traditional methods
- Pre-trained on fluorescence microscopy data
- Fast inference (GPU-accelerated)

### 5.3 Segmentation Parameters

```python
def segment_nuclei(image, model='2D_versatile_fluo', prob_thresh=0.43, nms_thresh=0.3):
    """
    Parameters:
    - prob_thresh (0.43): Minimum probability for a pixel to be considered nuclear
      Higher = fewer detections, more confident
      Lower = more detections, more false positives
    
    - nms_thresh (0.3): Overlap threshold for non-maximum suppression
      Higher = allows more overlap between nuclei
      Lower = more aggressive merging of overlapping detections
    
    These defaults are optimized for fluorescence microscopy.
    """
```

### 5.4 Output: Label Masks

The output is a **label image** where:
- Background pixels = 0
- Each nucleus has a unique integer label (1, 2, 3, ...)

```
Original DAPI          Label Mask
┌─────────────┐       ┌─────────────┐
│ ○  ○    ○  │       │ 1  2    3  │
│   ○  ○     │  ──►  │   4  5     │
│ ○      ○   │       │ 6      7   │
└─────────────┘       └─────────────┘
```

---

## 6. Step 3: Feature Extraction

This is the core scientific innovation of the pipeline. We extract **~200 quantitative features** that describe nuclear morphology and chromatin organization.

### 6.1 Feature Categories

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CHROMETRIC FEATURES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │
│  │ GLOBAL MORPHOLOGY │  │ INTENSITY DISTRIB │  │    TEXTURE      │ │
│  │                   │  │                   │  │                 │ │
│  │ • Area            │  │ • Mean intensity  │  │ • GLCM features │ │
│  │ • Perimeter       │  │ • Std deviation   │  │ • Hu moments    │ │
│  │ • Eccentricity    │  │ • Skewness        │  │ • Zernike       │ │
│  │ • Solidity        │  │ • Kurtosis        │  │ • Haralick      │ │
│  │ • Radii stats     │  │ • Hetero/Euchro   │  │ • PDI           │ │
│  └───────────────────┘  └───────────────────┘  └─────────────────┘ │
│                                                                     │
│  ┌───────────────────┐                                             │
│  │ BOUNDARY CURVATURE│                                             │
│  │                   │                                             │
│  │ • Local curvature │                                             │
│  │ • Circumradius    │                                             │
│  │ • Irregularity    │                                             │
│  └───────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Global Morphology Features

These features describe the **size and shape** of the nucleus.

#### 6.2.1 Basic Shape Descriptors

| Feature | Description | Biological Relevance |
|---------|-------------|---------------------|
| **Area** | Number of pixels in nucleus | Cell cycle stage (larger during S/G2) |
| **Perimeter** | Boundary length | Nuclear envelope complexity |
| **Eccentricity** | Ellipse fit (0=circle, 1=line) | Cell polarization, migration |
| **Solidity** | Area / Convex hull area | Nuclear lobulation (low = lobed) |
| **Extent** | Area / Bounding box area | Nuclear shape regularity |

#### 6.2.2 Radii Statistics

The nucleus is characterized by distances from centroid to boundary:

```python
def compute_radii_features(mask):
    """
    Compute radial distance statistics from centroid to boundary.
    
    For each boundary point, calculate distance to centroid.
    Then compute statistics over all radii:
    
    - min_radius: Minimum distance (indentations)
    - max_radius: Maximum distance (protrusions)
    - mean_radius: Average size measure
    - std_radius: Shape irregularity
    - mode_radius: Most common radius
    
    Biological interpretation:
    - High std_radius → irregular nuclear envelope
    - Large max/min ratio → elongated or lobed nucleus
    """
```

#### 6.2.3 Feret Diameters (Caliper Sizes)

```
        ┌─────────────────────┐
        │     Feret Max       │
        │  ←─────────────────→│
        │    ╭───────────╮    │
        │   ╱             ╲   │ ↑
        │  │               │  │ │ Feret Min
        │   ╲             ╱   │ ↓
        │    ╰───────────╯    │
        └─────────────────────┘
```

- **Feret Max**: Maximum caliper diameter (longest axis)
- **Feret Min**: Minimum caliper diameter (shortest axis)
- **Aspect Ratio**: Feret Max / Feret Min (elongation)

### 6.3 Intensity Distribution Features

These features characterize the **chromatin distribution** within the nucleus.

#### 6.3.1 Basic Intensity Statistics

```python
def compute_intensity_features(image, mask):
    """
    Compute intensity statistics within the nuclear mask.
    
    Features:
    - int_mean: Average intensity (overall chromatin density)
    - int_std: Standard deviation (heterogeneity)
    - int_skew: Skewness (asymmetry of distribution)
    - int_kurtosis: Kurtosis (tail heaviness)
    - int_mode: Most frequent intensity value
    - int_median: Median intensity (robust center)
    """
```

#### 6.3.2 Heterochromatin/Euchromatin Analysis

Chromatin exists in two main states:
- **Heterochromatin**: Condensed, transcriptionally silent (high DAPI intensity)
- **Euchromatin**: Open, transcriptionally active (lower DAPI intensity)

```python
def hetero_euchromatin_ratio(intensities, threshold='otsu'):
    """
    Classify pixels as heterochromatin or euchromatin.
    
    Method:
    1. Apply Otsu's threshold to separate high/low intensity
    2. Count pixels in each category
    3. Compute ratio
    
    Biological interpretation:
    - High ratio → more condensed chromatin → quiescent state
    - Low ratio → more open chromatin → active transcription
    
    Hypothesis: DZ B-cells (proliferating) may have more euchromatin
    """
```

### 6.4 Texture Features

Texture analysis captures **spatial patterns** in chromatin organization.

#### 6.4.1 Gray Level Co-occurrence Matrix (GLCM)

The GLCM measures how often pairs of pixel intensities occur at specific spatial relationships.

```
For distance d=1, angle θ=0° (horizontal):

Image:                  GLCM (simplified):
┌─────────────┐         
│ 1 1 2 2 │         Intensity i →
│ 1 2 2 3 │         ┌───┬───┬───┐
│ 2 2 3 3 │         │ 2 │ 3 │ 0 │  ↓ Intensity j
│ 2 3 3 3 │         ├───┼───┼───┤
└─────────────┘         │ 1 │ 4 │ 3 │
                        ├───┼───┼───┤
                        │ 0 │ 1 │ 4 │
                        └───┴───┴───┘
```

**GLCM-derived features**:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Contrast** | Σᵢⱼ (i-j)² P(i,j) | Local intensity variation |
| **Homogeneity** | Σᵢⱼ P(i,j)/(1+|i-j|) | Uniformity of texture |
| **Energy** | Σᵢⱼ P(i,j)² | Textural uniformity |
| **Correlation** | (Σᵢⱼ ijP(i,j) - μₓμᵧ)/(σₓσᵧ) | Linear dependency of gray levels |
| **Entropy** | -Σᵢⱼ P(i,j) log P(i,j) | Randomness/complexity |

#### 6.4.2 Image Moments

Moments describe the spatial distribution of intensity:

```python
def compute_hu_moments(image, mask):
    """
    Hu's 7 invariant moments - invariant to translation, rotation, scale.
    
    M_pq = Σₓ Σᵧ x^p y^q I(x,y)
    
    Central moments (translation invariant):
    μ_pq = Σₓ Σᵧ (x-x̄)^p (y-ȳ)^q I(x,y)
    
    Normalized moments (scale invariant):
    η_pq = μ_pq / μ₀₀^((p+q)/2 + 1)
    
    Hu moments (rotation invariant):
    Combinations of η_pq designed to be rotation invariant
    """
```

#### 6.4.3 Peripheral Distribution Index (PDI)

```python
def peripheral_distribution_index(image, mask):
    """
    Measures how chromatin is distributed relative to the nuclear center.
    
    PDI = (Intensity in periphery) / (Intensity in center)
    
    High PDI → Chromatin concentrated at nuclear envelope
    Low PDI → Chromatin concentrated in nuclear center
    
    Biological relevance:
    - Heterochromatin often localizes to nuclear periphery
    - Changes in PDI can indicate chromatin reorganization
    """
```

### 6.5 Boundary Curvature Features

These features characterize the **nuclear envelope morphology**.

```python
def compute_curvature_features(boundary_coords):
    """
    Local curvature at each boundary point.
    
    Curvature κ = dθ/ds where:
    - θ is the tangent angle
    - s is arc length
    
    Features:
    - mean_curvature: Average boundary curvature
    - max_curvature: Maximum local curvature (sharp corners)
    - std_curvature: Curvature variation (irregularity)
    - positive_curvature_fraction: Fraction of convex regions
    
    Biological interpretation:
    - High curvature variation → nuclear envelope blebbing
    - May indicate mechanical stress or disease state
    """
```

### 6.6 Spatial Coordinates

```python
def extract_spatial_features(labels):
    """
    Extract centroid positions for each nucleus.
    
    Output:
    - centroid-0 (y coordinate)
    - centroid-1 (x coordinate)
    - image identifier
    - nucleus ID (nuc_id)
    
    Used for:
    - Spatial analysis (T-cell interactions)
    - Boundary distance calculations
    - Visualization
    """
```

### 6.7 Protein Intensity Measurement

```python
def measure_protein_intensity(protein_image, labels, dilation_radius=10):
    """
    Measure protein marker intensity for each cell.
    
    Method:
    1. Dilate nuclear mask to approximate cell boundary
    2. Measure mean intensity in dilated region
    
    Why dilation?
    - Nucleus ≠ whole cell
    - Cytoplasmic proteins (CD3) need larger region
    - 10 pixel dilation ≈ typical cell size
    
    Output:
    - Mean intensity per cell
    - Integrated intensity (mean × area)
    """
```

---

## 7. Step 4: Cell Type Detection

### 7.1 The Classification Problem

Given protein intensities (CD3, AICDA), assign each cell to:
- **T-cells**: CD3+
- **DZ B-cells**: AICDA+, CD3-
- **LZ B-cells**: AICDA-, CD3-

### 7.2 Gaussian Mixture Model (GMM) Thresholding

**Why GMM instead of simple thresholding?**

Intensity distributions are typically **bimodal** (positive and negative populations):

```
Frequency
    │      ╭──╮
    │     ╱    ╲         ╭───╮
    │    ╱      ╲       ╱     ╲
    │   ╱        ╲     ╱       ╲
    │──╱──────────╲───╱─────────╲──
    └──────────────────────────────► Intensity
       Negative      Positive
```

GMM fits two Gaussian distributions and finds the optimal threshold:

```python
def fit_gmm_threshold(intensities, n_components=2):
    """
    Fit Gaussian Mixture Model to find optimal threshold.
    
    Steps:
    1. Fit 2-component GMM to log-transformed intensities
    2. Identify low and high intensity components
    3. Threshold = intersection point of two Gaussians
    
    Mathematical formulation:
    P(x) = π₁ N(x|μ₁,σ₁) + π₂ N(x|μ₂,σ₂)
    
    Threshold T where: P(x=T|component1) = P(x=T|component2)
    
    Advantages:
    - Data-driven threshold
    - Adapts to staining variation
    - More robust than fixed thresholds
    """
```

### 7.3 Cell Type Assignment Logic

```python
def assign_cell_types(cd3_positive, aicda_positive):
    """
    Classification logic:
    
    if CD3+ → T-cell
    else if AICDA+ → DZ B-cell  
    else → LZ B-cell
    
    Biological rationale:
    - T-cells express CD3 (T-cell receptor complex)
    - DZ B-cells express AICDA (somatic hypermutation)
    - LZ B-cells lack AICDA (not actively mutating)
    
    Note: This is a simplification. In reality:
    - Some LZ cells may have residual AICDA
    - Cell state is a continuum, not discrete categories
    """
```

---

## 8. Step 5: Classification Analysis

### 8.1 The Machine Learning Question

**Can chromatin features alone distinguish DZ from LZ B-cells?**

This is the central scientific question. If chromatin organization differs between zones, it suggests that nuclear architecture is linked to functional state.

### 8.2 Feature Preprocessing

#### 8.2.1 Removing Low-Quality Features

```python
def clean_features(features):
    """
    Remove problematic features:
    
    1. Constant features (no variation) → no discriminative power
    2. Features with missing values → incomplete data
    3. Highly correlated features → redundant information
    
    Correlation removal:
    - Compute pairwise Pearson correlation
    - If |r| > 0.8, remove one feature
    - Keeps dataset size manageable
    - Reduces multicollinearity
    """
```

#### 8.2.2 Why Remove Correlated Features?

```
Feature A ──────────────────────────────────────────────┐
           r = 0.95 (highly correlated)                │
Feature B ──────────────────────────────────────────────┤
                                                        │ Same information,
Feature C ──────────────────────────────────────────────┤ different names
           r = 0.92                                     │
Feature D ──────────────────────────────────────────────┘

Solution: Keep only one representative feature from each correlated cluster
```

### 8.3 Random Forest Classification

**Why Random Forest?**

| Property | Advantage |
|----------|-----------|
| Non-linear | Captures complex relationships |
| Feature importance | Identifies which features matter |
| Robust to outliers | Ensemble averaging |
| Handles high dimensions | Works with 200+ features |
| No feature scaling needed | Tree-based splits |

```python
def random_forest_classification(X, y, n_estimators=100):
    """
    Random Forest Classifier
    
    Hyperparameters:
    - n_estimators=100: Number of trees
    - max_depth=None: Trees grow until pure
    - class_weight='balanced': Handles class imbalance
    
    Each tree:
    1. Bootstrap sample of training data
    2. At each split, consider √p features
    3. Split on best feature/threshold
    
    Prediction: Majority vote across all trees
    """
```

### 8.4 Cross-Validation Strategy

```python
def cross_validate(X, y, n_folds=10):
    """
    Stratified K-Fold Cross-Validation
    
    Why stratified?
    - Maintains class proportions in each fold
    - Important when classes are imbalanced
    
    Why 10 folds?
    - Good bias-variance tradeoff
    - 90% training, 10% validation per fold
    - Standard in machine learning
    
    Output:
    - Mean balanced accuracy ± std
    - Per-fold predictions
    - Feature importance
    """
```

### 8.5 Class Balancing

```python
def balance_classes(X, y):
    """
    Undersample majority class to match minority.
    
    Problem: If we have 30,000 LZ and 10,000 DZ cells,
             the classifier might just predict "LZ" always
             and achieve 75% accuracy.
    
    Solution: Randomly sample 10,000 LZ cells.
    
    Alternative: class_weight='balanced' in Random Forest
    (weights inversely proportional to class frequencies)
    """
```

### 8.6 Balanced Accuracy

```python
def balanced_accuracy(y_true, y_pred):
    """
    Balanced Accuracy = (Sensitivity + Specificity) / 2
    
    = (TP/(TP+FN) + TN/(TN+FP)) / 2
    
    Why balanced?
    - Regular accuracy is misleading with imbalanced classes
    - A classifier predicting all "majority" gets high accuracy
    - Balanced accuracy penalizes this behavior
    
    Interpretation:
    - 0.5 = random guessing
    - 0.6 = slight discriminative power
    - 0.7 = moderate discrimination
    - 0.8+ = strong discrimination
    """
```

### 8.7 Feature Importance

```python
def compute_feature_importance(forest):
    """
    Gini Importance (Mean Decrease in Impurity)
    
    For each feature:
    1. Sum the impurity decrease at all nodes using that feature
    2. Weight by the probability of reaching that node
    3. Average across all trees
    
    Interpretation:
    - Higher importance → feature is more useful for classification
    - Top features are candidate biomarkers
    - May indicate biological differences between cell types
    """
```

---

## 9. Step 6: Spatial Analysis

### 9.1 T-cell Interaction Analysis

**Biological Question**: Does proximity to T-cells affect B-cell chromatin organization?

T follicular helper (Tfh) cells provide survival signals to B-cells in the light zone. We hypothesize that B-cells near T-cells may have distinct chromatin states.

#### 9.1.1 Defining Interaction Zones

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│           T-cell Influence Zones                                    │
│                                                                     │
│                    Signaling radius (30 μm)                         │
│                 ╭─────────────────────────────╮                     │
│                ╱                               ╲                    │
│               │   Contact radius (15 μm)        │                   │
│               │     ╭─────────────────╮        │                    │
│               │    ╱                   ╲        │                   │
│               │   │                     │       │                   │
│               │   │      T-cell         │       │                   │
│               │   │        ●            │       │                   │
│               │   │                     │       │                   │
│               │    ╲                   ╱        │                   │
│               │     ╰─────────────────╯        │                    │
│               │     T-cell interactors         │                    │
│                ╲                               ╱                    │
│                 ╰─────────────────────────────╯                     │
│                 Potential interactors                               │
│                                                                     │
│     Non-T-cell interactors (beyond signaling radius)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Categories**:
- **T-cell interactors**: Within 15 μm (physical contact possible)
- **Potential interactors**: 15-30 μm (cytokine signaling range)
- **Non-interactors**: Beyond 30 μm

#### 9.1.2 Efficient Distance Computation

```python
def find_tcell_neighbors_kdtree(cells, tcells, radius):
    """
    KD-Tree for O(n log n) spatial queries.
    
    Naive approach: O(n × m) pairwise distances
    With 40,000 B-cells and 20,000 T-cells = 800 million calculations!
    
    KD-Tree approach:
    1. Build tree from T-cell coordinates: O(m log m)
    2. Query nearest neighbor for each B-cell: O(n log m)
    Total: O((n+m) log m) ≈ seconds instead of hours
    """
```

### 9.2 Boundary Distance Analysis

**Biological Question**: Are cells near the DZ/LZ boundary different from those in the zone centers?

#### 9.2.1 Defining Distance to Boundary

```python
def compute_boundary_distance(cell, dz_cells, lz_cells, k=0.02):
    """
    Two methods for boundary distance:
    
    Method 1: Spatial distance to opposite-type cells
    - Find k closest cells of opposite type
    - Compute average distance
    - Normalize by image size
    
    Method 2: Local cell type frequency
    - Find N nearest neighbors
    - Compute fraction of same-type cells
    - 0.5 = at boundary, 1.0 = deep in zone
    
    Both methods identify cells at the DZ/LZ interface.
    """
```

#### 9.2.2 Biological Rationale

Cells at the boundary may:
- Be in transition between zones
- Have mixed signaling inputs
- Show intermediate chromatin states
- Be relevant for understanding zone dynamics

---

## 10. Statistical Methods

### 10.1 Marker Feature Identification

**Goal**: Find features that significantly differ between DZ and LZ B-cells.

#### 10.1.1 Welch's t-test

```python
def welch_ttest(group1, group2):
    """
    Welch's t-test (unequal variances assumed)
    
    t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂)
    
    Degrees of freedom (Welch-Satterthwaite):
    ν = (s₁²/n₁ + s₂²/n₂)² / ((s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1))
    
    Why Welch's instead of Student's t-test?
    - Does not assume equal variances
    - More robust for unequal sample sizes
    - Generally recommended for biological data
    """
```

#### 10.1.2 Multiple Testing Correction (FDR)

**The Problem**: With 100 features tested at α=0.05, we expect 5 false positives by chance.

**Solution**: Benjamini-Hochberg False Discovery Rate (FDR) correction

```python
def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg procedure:
    
    1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
    2. Find largest k where: p(k) ≤ (k/m) × α
    3. Reject hypotheses 1, 2, ..., k
    
    Controls the False Discovery Rate:
    FDR = E[V/R] where V = false positives, R = total rejections
    
    Interpretation:
    - FDR < 0.05 means <5% of significant features are false positives
    - Less conservative than Bonferroni
    - Appropriate for discovery/screening
    """
```

### 10.2 Effect Size

```python
def cohens_d(group1, group2):
    """
    Cohen's d effect size:
    
    d = (μ₁ - μ₂) / s_pooled
    
    Where s_pooled = √((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2))
    
    Interpretation:
    |d| < 0.2: negligible
    |d| = 0.2-0.5: small
    |d| = 0.5-0.8: medium
    |d| > 0.8: large
    
    Why report effect size?
    - p-value depends on sample size
    - With 60,000 cells, tiny differences are "significant"
    - Effect size measures practical significance
    """
```

---

## 11. Interpretation of Results

### 11.1 Classification Performance

**Balanced Accuracy = 0.621 (±0.010)**

What does this mean?

```
Interpretation scale:
├── 0.50 ──────── Random guessing (no discrimination)
│
├── 0.55 ──────── Very weak discrimination
│
├── 0.60 ──────── ★ Weak but significant discrimination
│   └── Our result: 0.621
│
├── 0.70 ──────── Moderate discrimination
│
├── 0.80 ──────── Strong discrimination
│
└── 1.00 ──────── Perfect discrimination
```

**Scientific interpretation**:
- Chromatin features contain **some** information about DZ vs LZ identity
- The effect is subtle but statistically robust (low variance across folds)
- This is expected: cell type identity is primarily determined by proteins, not chromatin
- The result suggests chromatin organization is **associated with** functional state

### 11.2 Important Features

Top features indicate which aspects of chromatin organization differ between zones:

```
Example feature importance ranking:
┌──────────────────────────────────────────────────────────────────┐
│ Feature               │ Importance │ Interpretation              │
├──────────────────────────────────────────────────────────────────┤
│ entropy               │    0.08    │ Chromatin complexity        │
│ int_std               │    0.06    │ Intensity variation         │
│ glcm_contrast         │    0.05    │ Local texture contrast      │
│ het_euchro_ratio      │    0.04    │ Condensed vs open chromatin │
│ peripheral_idx        │    0.04    │ Peripheral localization     │
└──────────────────────────────────────────────────────────────────┘
```

### 11.3 Spatial Analysis Interpretation

**T-cell interaction effect**:
- If classification accuracy differs for cells near vs far from T-cells, it suggests T-cell signals affect chromatin organization.

**Boundary effect**:
- If cells at the DZ/LZ boundary have distinct chromatin, it may indicate:
  - Transitional states
  - Zone-specific microenvironments
  - Active remodeling during zone transition

---

## 12. References

### Biological Background

1. Victora, G. D., & Nussenzweig, M. C. (2012). Germinal centers. Annual Review of Immunology, 30, 429-457.

2. Mesin, L., Ersching, J., & Bhattacharya, D. (2016). Germinal center B cell dynamics. Immunity, 45(3), 471-482.

3. Muramatsu, M., et al. (2000). Class switch recombination and hypermutation require activation-induced cytidine deaminase. Cell, 102(5), 553-563.

### Computational Methods

4. Schmidt, U., et al. (2018). Cell detection with star-convex polygons. MICCAI.

5. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics.

6. Hu, M. K. (1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory.

### Statistical Methods

7. Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. Biometrika.

8. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. Journal of the Royal Statistical Society: Series B.

---

## Appendix: Running the Pipeline

### Command Line Usage

```bash
# Full pipeline
gc-pipeline pipeline --config configs/dataset1_config.yaml

# Individual steps
gc-pipeline preprocess --input-dir data/raw --output-dir data/processed
gc-pipeline segment --input-dir data/processed/dapi_scaled --output-dir data/processed/segmented
gc-pipeline extract --config configs/dataset1_config.yaml
gc-pipeline analyze --config configs/dataset1_config.yaml
```

### Configuration File Structure

```yaml
# configs/dataset1_config.yaml
dataset: dataset1
data_dir: data/dataset1

preprocessing:
  channels:
    dapi: 1
    cd3: 2
    aicda: 3
  normalize:
    q_low: 0.01
    q_high: 0.998

segmentation:
  model: 2D_versatile_fluo
  prob_thresh: 0.43
  nms_thresh: 0.3

analysis:
  correlation_threshold: 0.8
  cv_folds: 10
  random_seed: 42
  spatial_parameters:
    pixel_size: 0.3225
    contact_radius: 15.0
    signaling_radius: 30.0
```

---

*Document version: 1.0*
*Last updated: December 2025*
*Pipeline version: 0.1.0*

