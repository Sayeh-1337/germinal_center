"""Analysis command for feature data"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, drop_columns: List[str], index_col: str = "nuc_id"):
    """Clean data by removing specified columns and rows with missing values
    
    Args:
        data: Input DataFrame
        drop_columns: Columns to drop
        index_col: Column to use as index
        
    Returns:
        Cleaned DataFrame
    """
    # Set index
    if index_col in data.columns:
        data = data.set_index(index_col)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Drop specified columns
    cols_to_drop = [c for c in drop_columns if c in numeric_data.columns]
    numeric_data = numeric_data.drop(columns=cols_to_drop, errors='ignore')
    
    # Remove constant columns and columns with missing values
    initial_cols = len(numeric_data.columns)
    numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
    numeric_data = numeric_data.dropna(axis=1, how='any')
    removed_cols = initial_cols - len(numeric_data.columns)
    
    logger.info(f"Removed {removed_cols} constant or features with missing values. Remaining: {len(numeric_data.columns)}")
    
    # Remove rows with missing values
    initial_rows = len(numeric_data)
    numeric_data = numeric_data.dropna(axis=0, how='any')
    removed_rows = initial_rows - len(numeric_data)
    
    logger.info(f"Removed {removed_rows} samples with missing values. Remaining: {len(numeric_data)}")
    
    return numeric_data


def remove_correlated_features(data: pd.DataFrame, threshold: float = 0.8):
    """Remove highly correlated features
    
    Args:
        data: Input DataFrame with numeric features
        threshold: Correlation threshold
        
    Returns:
        DataFrame with correlated features removed
    """
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    result = data.drop(columns=to_drop)
    logger.info(f"Removed {len(to_drop)}/{len(data.columns)} features with correlation above {threshold}. Remaining: {len(result.columns)}")
    
    return result


def run_analysis(
    features_dir: str,
    output_dir: str,
    analysis_types: List[str] = None,
    metadata_path: Optional[str] = None,
    correlation_threshold: float = 0.8,
    random_seed: int = 1234,
    pixel_size: float = 0.3225,
    contact_radius: float = 15.0,
    signaling_radius: float = 30.0,
    border_threshold: float = 0.4,
    generate_plots: bool = True,
    n_permutations: int = 10000
):
    """Run analysis on extracted features
    
    Args:
        features_dir: Directory with feature CSV files
        output_dir: Output directory for analysis results
        analysis_types: Types of analysis to run
        metadata_path: Path to metadata file (for correlation analysis)
        correlation_threshold: Threshold for feature correlation filtering
        random_seed: Random seed for reproducibility
        pixel_size: Pixel size in microns (for spatial analyses)
        contact_radius: T-cell physical contact radius in microns
        signaling_radius: T-cell signaling radius in microns
        border_threshold: Threshold for DZ/LZ border proximity classification
        generate_plots: Whether to generate visualization plots
        n_permutations: Number of permutations for statistical tests
    """
    np.random.seed(random_seed)
    
    if analysis_types is None:
        analysis_types = ['cell_type']
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load feature data
    nuc_features_path = os.path.join(features_dir, "nuc_features.csv")
    if not os.path.exists(nuc_features_path):
        raise FileNotFoundError(f"Feature file not found: {nuc_features_path}")
    
    nuc_features = pd.read_csv(nuc_features_path, index_col=0)
    logger.info(f"Loaded {len(nuc_features)} samples with {len(nuc_features.columns)} features")
    
    # Load spatial coordinates if available
    spatial_path = os.path.join(features_dir, "spatial_coordinates.csv")
    spatial_coords = None
    if os.path.exists(spatial_path):
        spatial_coords = pd.read_csv(spatial_path, index_col=0)
        logger.info(f"Loaded spatial coordinates for {len(spatial_coords)} samples")
    
    # Load protein levels if available
    aicda_path = os.path.join(features_dir, "aicda_levels.csv")
    cd3_path = os.path.join(features_dir, "cd3_levels.csv")
    aicda_levels = pd.read_csv(aicda_path, index_col=0) if os.path.exists(aicda_path) else None
    cd3_levels = pd.read_csv(cd3_path, index_col=0) if os.path.exists(cd3_path) else None
    
    # Clean and preprocess data
    meta_columns = [
        "label", "label_id", "weighted_centroid-0", "weighted_centroid-1", "weighted_centroid_y", "weighted_centroid_x",
        "centroid-0", "centroid-1", "centroid_y", "centroid_x", "bbox-0", "bbox-1", "bbox-2", "bbox-3",
        "image", "orientation", "nuc_id"
    ]
    
    logger.info("Cleaning feature data...")
    cleaned_features = clean_data(nuc_features, drop_columns=meta_columns, index_col="nuc_id")
    
    logger.info("Removing correlated features...")
    filtered_features = remove_correlated_features(cleaned_features, correlation_threshold)
    
    # Save processed features
    filtered_features.to_csv(os.path.join(output_dir, "processed_features.csv"))
    
    # Run requested analyses
    results = {}
    
    # Cell type detection (must run first if other analyses need it)
    if 'cell_type_detection' in analysis_types or 'detect_cells' in analysis_types:
        logger.info("Running cell type detection...")
        nuc_features, results['cell_type_detection'] = run_cell_type_detection(
            nuc_features, aicda_levels, cd3_levels, output_dir, generate_plots
        )
    
    if 'cell_type' in analysis_types or 'classification' in analysis_types:
        logger.info("Running cell type classification analysis...")
        results['cell_type'] = run_cell_type_analysis(
            nuc_features, filtered_features, output_dir, random_seed, generate_plots
        )
    
    if 'tcell_interaction' in analysis_types or 'tcell' in analysis_types:
        logger.info("Running T-cell interaction analysis...")
        nuc_features, results['tcell_interaction'] = run_tcell_interaction_analysis(
            nuc_features, filtered_features, spatial_coords, output_dir, random_seed,
            pixel_size, contact_radius, signaling_radius, generate_plots
        )
    
    if 'boundary' in analysis_types or 'dz_lz_boundary' in analysis_types:
        logger.info("Running DZ/LZ boundary analysis...")
        results['boundary'] = run_boundary_analysis(
            nuc_features, filtered_features, spatial_coords, output_dir, 
            random_seed, border_threshold, generate_plots
        )
    
    if 'correlation' in analysis_types:
        if metadata_path is None:
            logger.warning("Correlation analysis requires metadata file. Skipping.")
        else:
            logger.info("Running correlation analysis...")
            results['correlation'] = run_correlation_analysis(
                nuc_features, filtered_features, metadata_path, output_dir,
                n_permutations, random_seed
            )
    
    if 'umap' in analysis_types or 'visualization' in analysis_types:
        logger.info("Running UMAP visualization...")
        results['umap'] = run_umap_analysis(
            nuc_features, filtered_features, output_dir, random_seed, generate_plots
        )
    
    if 'markers' in analysis_types or 'differential' in analysis_types:
        logger.info("Running marker/differential expression analysis...")
        results['markers'] = run_marker_analysis(
            nuc_features, filtered_features, output_dir
        )
    
    # Save updated features with annotations
    nuc_features.to_csv(os.path.join(output_dir, "nuc_features_annotated.csv"))
    
    # Save analysis summary
    summary = {
        'n_samples': len(nuc_features),
        'n_features_original': len(nuc_features.columns),
        'n_features_processed': len(filtered_features.columns),
        'analyses_run': analysis_types
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, "analysis_summary.csv"), index=False)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return results


def run_cell_type_detection(nuc_features, aicda_levels, cd3_levels, output_dir, generate_plots=True):
    """Run GMM-based cell type detection"""
    from src.analysis.cell_type_detection import assign_cell_types
    
    results_dir = os.path.join(output_dir, "cell_type_detection")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if aicda_levels is None or cd3_levels is None:
        logger.warning("AICDA and/or CD3 levels not available. Cannot detect cell types.")
        return nuc_features, None
    
    # Detect cell types
    nuc_features = assign_cell_types(nuc_features, aicda_levels, cd3_levels)
    
    # Save results
    cell_type_counts = nuc_features['cell_type'].value_counts()
    cell_type_counts.to_csv(os.path.join(results_dir, "cell_type_counts.csv"))
    
    # Save per-image counts
    per_image_counts = nuc_features.groupby(['image', 'cell_type']).size().unstack(fill_value=0)
    per_image_counts.to_csv(os.path.join(results_dir, "cell_type_counts_per_image.csv"))
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import plot_cell_type_distribution
            
            # Cell type distribution plot
            plot_cell_type_distribution(
                nuc_features,
                cell_type_col='cell_type',
                output_path=os.path.join(results_dir, "cell_type_distribution.png"),
                title="Cell Type Distribution"
            )
            logger.info(f"Saved cell type distribution plot")
        except Exception as e:
            logger.warning(f"Could not generate cell type plot: {e}")
    
    return nuc_features, {'cell_type_counts': cell_type_counts.to_dict()}


def run_cell_type_analysis(nuc_features, filtered_features, output_dir, random_seed, generate_plots=True):
    """Run cell type classification analysis"""
    from src.analysis.statistical_tests import run_cv_classification, find_markers
    
    results_dir = os.path.join(output_dir, "cell_type_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if cell type labels are available
    if 'cell_type' not in nuc_features.columns:
        logger.info("No cell_type column found. Generating summary statistics only.")
        
        # Generate descriptive statistics
        stats = filtered_features.describe()
        stats.to_csv(os.path.join(results_dir, "feature_statistics.csv"))
        
        return {'status': 'statistics_only', 'n_features': len(filtered_features.columns)}
    
    # Filter to B-cells for DZ vs LZ classification
    bcell_mask = nuc_features['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
    bcell_nuc_features = nuc_features[bcell_mask]
    
    if len(bcell_nuc_features) < 100:
        logger.warning(f"Too few B-cells ({len(bcell_nuc_features)}). Skipping classification.")
        return None
    
    # Get features for B-cells - use nuc_id column to match with filtered_features index
    # nuc_features uses label as index, filtered_features uses nuc_id as index
    bcell_nuc_ids = bcell_nuc_features['nuc_id'].values
    common_idx = [idx for idx in bcell_nuc_ids if idx in filtered_features.index]
    
    bcell_features = filtered_features.loc[common_idx]
    # Get labels using nuc_id column match
    bcell_nuc_features_indexed = bcell_nuc_features.set_index('nuc_id')
    bcell_labels = bcell_nuc_features_indexed.loc[common_idx, 'cell_type']
    
    logger.info(f"Running DZ vs LZ B-cell classification on {len(bcell_features)} samples...")
    
    # Run cross-validated classification
    cv_results = run_cv_classification(
        bcell_features, bcell_labels, n_folds=10, random_state=random_seed, balance=True
    )
    
    # Save results
    cv_summary = pd.DataFrame([{
        'balanced_accuracy_mean': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(bcell_features),
        'n_features': len(bcell_features.columns)
    }])
    cv_summary.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
    
    # Save feature importance
    cv_results['feature_importance'].to_csv(
        os.path.join(results_dir, "feature_importance.csv"), index=False
    )
    
    # Run marker analysis
    logger.info("Finding marker features...")
    markers = find_markers(bcell_features, bcell_labels, test='welch')
    markers.to_csv(os.path.join(results_dir, "marker_features.csv"), index=False)
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import (
                plot_confusion_matrix, plot_feature_importance, plot_roc_binary,
                plot_marker_comparison, plot_violin_with_stats, plot_cell_type_distribution
            )
            
            # Cell type distribution for B-cells
            plot_cell_type_distribution(
                bcell_nuc_features,
                cell_type_col='cell_type',
                output_path=os.path.join(results_dir, "bcell_distribution.png"),
                title="B-cell Type Distribution (DZ vs LZ)"
            )
            
            # Confusion matrix
            plot_confusion_matrix(
                cv_results['true_labels'],
                cv_results['predictions'],
                list(cv_results['classes']),
                os.path.join(results_dir, "confusion_matrix.png")
            )
            
            # Feature importance
            plot_feature_importance(
                cv_results['feature_importance']['importance'].values,
                cv_results['feature_importance']['feature'].tolist(),
                os.path.join(results_dir, "feature_importance.png"),
                n_features=20
            )
            
            # ROC curve (binary classification)
            if 'probabilities' in cv_results and cv_results['probabilities'] is not None:
                # Use DZ B-cells as positive class
                dz_probs = cv_results['probabilities']
                if dz_probs.ndim == 2:
                    # Get probability for DZ B-cells class
                    dz_class_idx = list(cv_results['classes']).index('DZ B-cells') if 'DZ B-cells' in cv_results['classes'] else 0
                    dz_probs = dz_probs[:, dz_class_idx]
                
                plot_roc_binary(
                    cv_results['true_labels'],
                    dz_probs,
                    pos_label='DZ B-cells',
                    output_path=os.path.join(results_dir, "roc_curve.png"),
                    title="ROC Curve - DZ vs LZ B-cell Classification"
                )
            
            # Marker comparison violin plots
            if len(markers) > 0:
                # Prepare data for marker plots
                marker_plot_data = bcell_features.copy()
                marker_plot_data['cell_type'] = bcell_labels.values
                
                plot_marker_comparison(
                    marker_plot_data,
                    markers,
                    group_col='cell_type',
                    n_markers=6,
                    output_path=os.path.join(results_dir, "marker_violin_plots.png")
                )
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    logger.info(f"  Balanced accuracy: {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})")
    
    return {
        'balanced_accuracy': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(bcell_features)
    }


def run_tcell_interaction_analysis(
    nuc_features, filtered_features, spatial_coords, output_dir, random_seed,
    pixel_size=0.3225, contact_radius=15.0, signaling_radius=30.0, generate_plots=True
):
    """Run T-cell interaction analysis"""
    from src.analysis.tcell_interaction import assign_tcell_influence, get_distances_to_tcells
    from src.analysis.statistical_tests import run_cv_classification, find_markers
    
    results_dir = os.path.join(output_dir, "tcell_interaction_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if spatial_coords is None:
        logger.warning("Spatial coordinates not available. Skipping T-cell interaction analysis.")
        return nuc_features, None
    
    if 'cell_type' not in nuc_features.columns:
        logger.warning("Cell type labels not available. Skipping T-cell interaction analysis.")
        return nuc_features, None
    
    # Assign T-cell influence zones
    nuc_features = assign_tcell_influence(
        nuc_features, spatial_coords,
        contact_radius=contact_radius,
        signaling_radius=signaling_radius,
        pixel_size=pixel_size
    )
    
    # Save influence counts
    influence_counts = nuc_features['tcell_influence'].value_counts()
    influence_counts.to_csv(os.path.join(results_dir, "tcell_influence_counts.csv"))
    
    # Compute distances to T-cells
    logger.info("Computing distances to T-cells...")
    tcell_distances = get_distances_to_tcells(nuc_features, spatial_coords)
    tcell_distances.to_csv(os.path.join(results_dir, "tcell_distances.csv"), index=False)
    
    # Merge distances with features
    nuc_features = nuc_features.merge(
        tcell_distances[['nuc_id', 'tcell_mean_distance', 'tcell_median_distance', 'tcell_min_distance']],
        on='nuc_id', how='left'
    )
    
    # Classification: T-cell interactors vs Non-interactors
    bcell_mask = nuc_features['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
    interaction_mask = nuc_features['tcell_influence'].isin(['T-cell interactors', 'Non-T-cell interactors'])
    analysis_mask = bcell_mask & interaction_mask
    
    analysis_features = nuc_features[analysis_mask]
    # Use nuc_id column to match with filtered_features index
    analysis_nuc_ids = analysis_features['nuc_id'].values
    common_idx = [idx for idx in analysis_nuc_ids if idx in filtered_features.index]
    
    if len(common_idx) > 100:
        X = filtered_features.loc[common_idx]
        analysis_features_indexed = analysis_features.set_index('nuc_id')
        y = analysis_features_indexed.loc[common_idx, 'tcell_influence']
        
        logger.info(f"Running T-cell interaction classification on {len(X)} B-cells...")
        cv_results = run_cv_classification(X, y, n_folds=10, random_state=random_seed, balance=True)
        
        cv_summary = pd.DataFrame([{
            'balanced_accuracy_mean': cv_results['cv_mean'],
            'balanced_accuracy_std': cv_results['cv_std'],
            'n_samples': len(X)
        }])
        cv_summary.to_csv(os.path.join(results_dir, "tcell_classification_results.csv"), index=False)
        
        cv_results['feature_importance'].to_csv(
            os.path.join(results_dir, "tcell_feature_importance.csv"), index=False
        )
        
        logger.info(f"  Balanced accuracy: {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})")
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import (
                plot_spatial_scatter, plot_tcell_influence_distribution,
                plot_tcell_interaction_zones, plot_feature_importance
            )
            
            # T-cell influence distribution by cell type
            plot_tcell_influence_distribution(
                nuc_features,
                cell_type_col='cell_type',
                influence_col='tcell_influence',
                output_path=os.path.join(results_dir, "tcell_influence_distribution.png"),
                title="T-cell Influence by Cell Type"
            )
            
            # Feature importance for T-cell classification
            if len(common_idx) > 100 and 'feature_importance' in cv_results:
                plot_feature_importance(
                    cv_results['feature_importance']['importance'].values,
                    cv_results['feature_importance']['feature'].tolist(),
                    os.path.join(results_dir, "tcell_feature_importance.png"),
                    title="Feature Importance - T-cell Interaction",
                    n_features=15
                )
            
            # Spatial distribution plots for first 3 images
            for img in nuc_features['image'].unique()[:3]:
                img_data = nuc_features[nuc_features['image'] == img].merge(
                    spatial_coords, on='nuc_id', how='left'
                )
                
                # Determine centroid column names
                if 'centroid-1' in img_data.columns:
                    x_col, y_col = 'centroid-1', 'centroid-0'
                elif 'centroid_1' in img_data.columns:
                    x_col, y_col = 'centroid_1', 'centroid_0'
                else:
                    continue
                
                # Simple spatial scatter by T-cell influence
                plot_spatial_scatter(
                    img_data,
                    x_col=x_col,
                    y_col=y_col,
                    color_col='tcell_influence',
                    output_path=os.path.join(results_dir, f"spatial_tcell_influence_img{img}.png"),
                    title=f"T-cell Influence - Image {img}"
                )
                
                # Try to load raw image for overlay visualization
                try:
                    # Look for raw image
                    data_dir = os.path.dirname(os.path.dirname(output_dir))
                    raw_img_candidates = [
                        os.path.join(data_dir, "images", "raw", "merged", f"{img}.tif"),
                        os.path.join(data_dir, "raw", "merged", f"{img}.tif"),
                        os.path.join(data_dir, "images", "merged", f"{img}.tif"),
                    ]
                    
                    for img_path in raw_img_candidates:
                        if os.path.exists(img_path):
                            plot_tcell_interaction_zones(
                                img_data,
                                image_path=img_path,
                                x_col=x_col,
                                y_col=y_col,
                                output_path=os.path.join(results_dir, f"tcell_zones_overlay_img{img}.png"),
                                title=f"T-cell Interaction Zones - Image {img}"
                            )
                            break
                except Exception as img_e:
                    logger.debug(f"Could not create overlay for image {img}: {img_e}")
                    
        except Exception as e:
            logger.warning(f"Could not generate spatial plots: {e}")
    
    return nuc_features, {'influence_counts': influence_counts.to_dict()}


def run_boundary_analysis(
    nuc_features, filtered_features, spatial_coords, output_dir, 
    random_seed, border_threshold=0.4, generate_plots=True
):
    """Run DZ/LZ boundary analysis"""
    from src.analysis.boundary_analysis import (
        get_distances_to_dz_lz_border, assign_border_proximity, analyze_boundary_differences
    )
    from src.analysis.statistical_tests import run_cv_classification
    
    results_dir = os.path.join(output_dir, "boundary_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if spatial_coords is None:
        logger.warning("Spatial coordinates not available. Skipping boundary analysis.")
        return None
    
    if 'cell_type' not in nuc_features.columns:
        logger.warning("Cell type labels not available. Skipping boundary analysis.")
        return None
    
    # Compute distances to DZ/LZ border
    logger.info("Computing distances to DZ/LZ boundary...")
    border_distances = get_distances_to_dz_lz_border(nuc_features, spatial_coords)
    
    if len(border_distances) == 0:
        logger.warning("Could not compute border distances. Check cell type labels.")
        return None
    
    # Assign proximity status
    border_distances = assign_border_proximity(border_distances, threshold=border_threshold)
    border_distances.to_csv(os.path.join(results_dir, "border_distances.csv"), index=False)
    
    # Analyze differences
    border_distances_indexed = border_distances.set_index('nuc_id')
    analysis_results = analyze_boundary_differences(
        border_distances_indexed, filtered_features
    )
    
    # Save results for each cell type
    for cell_type, diff_df in analysis_results['cell_types'].items():
        safe_name = cell_type.replace(' ', '_').replace('/', '_')
        diff_df.to_csv(
            os.path.join(results_dir, f"boundary_differences_{safe_name}.csv"),
            index=False
        )
    
    # Classification: close vs distant for LZ B-cells
    lz_border = border_distances[border_distances['cell_type'] == 'LZ B-cells']
    if len(lz_border) > 100:
        common_idx = lz_border['nuc_id'].values
        common_idx = [idx for idx in common_idx if idx in filtered_features.index]
        
        if len(common_idx) > 50:
            X = filtered_features.loc[common_idx]
            y = border_distances[border_distances['nuc_id'].isin(common_idx)].set_index('nuc_id').loc[common_idx, 'border_proximity']
            
            logger.info(f"Running LZ B-cell boundary classification on {len(X)} samples...")
            cv_results = run_cv_classification(X, y, n_folds=10, random_state=random_seed, balance=True)
            
            cv_summary = pd.DataFrame([{
                'cell_type': 'LZ B-cells',
                'balanced_accuracy_mean': cv_results['cv_mean'],
                'balanced_accuracy_std': cv_results['cv_std'],
                'n_samples': len(X)
            }])
            cv_summary.to_csv(os.path.join(results_dir, "boundary_classification_lz.csv"), index=False)
            
            logger.info(f"  LZ B-cells boundary classification: {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})")
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import plot_spatial_scatter
            
            for img in border_distances['image'].unique()[:3]:
                img_data = border_distances[border_distances['image'] == img]
                plot_spatial_scatter(
                    img_data,
                    x_col='centroid-1',
                    y_col='centroid-0',
                    color_col='border_proximity',
                    output_path=os.path.join(results_dir, f"spatial_border_proximity_img{img}.png"),
                    title=f"Border Proximity - Image {img}"
                )
        except Exception as e:
            logger.warning(f"Could not generate spatial plots: {e}")
    
    return {'n_close': (border_distances['border_proximity'] == 'close').sum(),
            'n_distant': (border_distances['border_proximity'] == 'distant').sum()}


def run_correlation_analysis(
    nuc_features, filtered_features, metadata_path, output_dir,
    n_permutations=10000, random_seed=1234
):
    """Run correlation analysis with metadata"""
    from src.analysis.statistical_tests import run_correlation_screen
    
    results_dir = os.path.join(output_dir, "correlation_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path, index_col=0)
    
    # Compute median features per image
    median_features = filtered_features.copy()
    median_features['image'] = nuc_features.loc[median_features.index, 'image']
    median_features = median_features.groupby('image').median()
    
    # Find common samples
    common = median_features.index.intersection(metadata.index)
    if len(common) < 3:
        logger.warning(f"Only {len(common)} common samples between features and metadata.")
        return None
    
    logger.info(f"Computing correlations for {len(common)} samples...")
    
    # Run correlation screen for each numeric metadata column
    all_results = []
    for meta_col in metadata.select_dtypes(include=[np.number]).columns:
        analysis_data = median_features.loc[common].copy()
        analysis_data[meta_col] = metadata.loc[common, meta_col]
        
        corr_results = run_correlation_screen(
            analysis_data, meta_col, n_permutations, random_seed
        )
        corr_results['target'] = meta_col
        all_results.append(corr_results)
    
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(os.path.join(results_dir, "correlation_results.csv"), index=False)
        
        # Get top correlations
        top_corr = results_df.nsmallest(10, 'pearson_p_adj')
        top_corr.to_csv(os.path.join(results_dir, "top_correlations.csv"), index=False)
        
        if len(top_corr) > 0:
            logger.info(f"  Top correlation: {top_corr.iloc[0]['feature']} with {top_corr.iloc[0]['target']} (r={top_corr.iloc[0]['pearson_r']:.3f})")
        
        return {'n_correlations': len(results_df), 'n_significant': (results_df['pearson_p_adj'] < 0.05).sum()}
    
    return None


def run_umap_analysis(nuc_features, filtered_features, output_dir, random_seed, generate_plots=True):
    """Run UMAP visualization and clustering"""
    from src.analysis.visualization import compute_umap_embedding, plot_umap
    
    results_dir = os.path.join(output_dir, "umap_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Computing UMAP embedding for {len(filtered_features)} samples...")
    
    # Compute UMAP
    embedding = compute_umap_embedding(filtered_features, random_state=random_seed)
    
    # Try to cluster the embedding (hdbscan may not be installed)
    cluster_labels = None
    try:
        from src.analysis.visualization import cluster_umap_embedding
        cluster_labels = cluster_umap_embedding(embedding)
    except ImportError:
        logger.warning("HDBSCAN not installed. Skipping clustering. Install with: pip install hdbscan")
    except Exception as e:
        logger.warning(f"Clustering failed: {e}. Continuing without clustering.")
    
    # Save results - use nuc_id column to match since nuc_features uses different index
    if cluster_labels is not None:
        embedding['cluster'] = cluster_labels
    nuc_features_by_nucid = nuc_features.set_index('nuc_id')
    embedding['image'] = nuc_features_by_nucid.loc[embedding.index, 'image']
    if 'cell_type' in nuc_features.columns:
        embedding['cell_type'] = nuc_features_by_nucid.loc[embedding.index, 'cell_type']
    
    embedding.to_csv(os.path.join(results_dir, "umap_embedding.csv"))
    
    # Generate plots
    if generate_plots:
        # By cluster (if clustering succeeded)
        if cluster_labels is not None:
            plot_umap(
                embedding[['umap_0', 'umap_1']],
                embedding['cluster'],
                os.path.join(results_dir, "umap_clusters.png"),
                title="UMAP - HDBSCAN Clusters"
            )
        
        # By image
        plot_umap(
            embedding[['umap_0', 'umap_1']],
            embedding['image'],
            os.path.join(results_dir, "umap_by_image.png"),
            title="UMAP - By Image"
        )
        
        # By cell type if available
        if 'cell_type' in embedding.columns:
            plot_umap(
                embedding[['umap_0', 'umap_1']],
                embedding['cell_type'],
                os.path.join(results_dir, "umap_by_celltype.png"),
                title="UMAP - By Cell Type"
            )
    
    n_clusters = 0
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    return {'n_clusters': n_clusters, 'n_samples': len(embedding)}


def run_marker_analysis(nuc_features, filtered_features, output_dir):
    """Run marker/differential expression analysis"""
    from src.analysis.statistical_tests import find_markers
    
    results_dir = os.path.join(output_dir, "marker_analysis")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # By cell type - use nuc_id column to match with filtered_features index
    if 'cell_type' in nuc_features.columns:
        nuc_ids = nuc_features['nuc_id'].values
        common_idx = [idx for idx in nuc_ids if idx in filtered_features.index]
        nuc_features_by_nucid = nuc_features.set_index('nuc_id')
        labels = nuc_features_by_nucid.loc[common_idx, 'cell_type']
        features = filtered_features.loc[common_idx]
        
        logger.info("Finding markers by cell type...")
        markers = find_markers(features, labels, test='welch')
        markers.to_csv(os.path.join(results_dir, "markers_by_celltype.csv"), index=False)
        results['by_celltype'] = len(markers[markers['adjusted_pval'] < 0.05])
    
    # By T-cell influence - use nuc_id column to match
    if 'tcell_influence' in nuc_features.columns:
        nuc_ids = nuc_features['nuc_id'].values
        common_idx = [idx for idx in nuc_ids if idx in filtered_features.index]
        nuc_features_by_nucid = nuc_features.set_index('nuc_id')
        labels = nuc_features_by_nucid.loc[common_idx, 'tcell_influence']
        features = filtered_features.loc[common_idx]
        
        logger.info("Finding markers by T-cell influence...")
        markers = find_markers(features, labels, test='welch')
        markers.to_csv(os.path.join(results_dir, "markers_by_tcell_influence.csv"), index=False)
        results['by_tcell_influence'] = len(markers[markers['adjusted_pval'] < 0.05])
    
    return results
