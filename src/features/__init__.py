# -*- coding: utf-8 -*-
"""
Chromatin feature extraction module.
Adapted from nmco/chrometrics library with fixes for:
- Pandas 2.0+ compatibility (pd.concat instead of append)
- Scipy 1.11+ mode() compatibility
- Empty array edge cases
- Cross-platform path handling

Enhanced with:
- Multi-scale wavelet and fractal analysis
- Cell cycle state inference
- Relative and interaction features
"""

from src.features.global_morphology import measure_global_morphometrics
from src.features.intensity_features import measure_intensity_features
from src.features.texture_features import measure_texture_features
from src.features.curvature_features import measure_curvature_features
from src.features.feature_extraction import run_nuclear_chromatin_feat_ext
from src.features.multiscale_features import (
    extract_wavelet_chromatin_features,
    compute_fractal_dimension,
    analyze_chromatin_domain_sizes,
    compute_radial_intensity_profile,
    extract_all_multiscale_features
)
from src.features.cell_cycle import (
    infer_cell_cycle_state,
    compute_cell_cycle_features,
    stratify_by_cell_cycle,
    compute_cell_cycle_adjusted_features
)
from src.features.relative_features import (
    compute_relative_features,
    compute_interaction_features,
    compute_spatial_gradients,
    compute_neighborhood_composition,
    compute_boundary_proximity,
    extract_all_relative_features
)

__all__ = [
    # Core feature extraction
    'measure_global_morphometrics',
    'measure_intensity_features', 
    'measure_texture_features',
    'measure_curvature_features',
    'run_nuclear_chromatin_feat_ext',
    # Multi-scale features
    'extract_wavelet_chromatin_features',
    'compute_fractal_dimension',
    'analyze_chromatin_domain_sizes',
    'compute_radial_intensity_profile',
    'extract_all_multiscale_features',
    # Cell cycle
    'infer_cell_cycle_state',
    'compute_cell_cycle_features',
    'stratify_by_cell_cycle',
    'compute_cell_cycle_adjusted_features',
    # Relative features
    'compute_relative_features',
    'compute_interaction_features',
    'compute_spatial_gradients',
    'compute_neighborhood_composition',
    'compute_boundary_proximity',
    'extract_all_relative_features'
]

