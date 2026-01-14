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
    n_permutations: int = 10000,
    filter_gc_inside: bool = False,
    feature_description_file: Optional[str] = None,
    raw_image_dir: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None
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
        model_config: MLOps model comparison config (optional):
            - enabled: Enable model comparison
            - models: List of models to compare
            - tune_hyperparameters: Whether to tune hyperparameters
            - use_mlflow: Whether to track with MLflow
            - mlflow_experiment_name: MLflow experiment name
        generate_plots: Whether to generate visualization plots
        n_permutations: Number of permutations for statistical tests
        filter_gc_inside: Filter to only cells inside germinal center
        feature_description_file: Path to feature description CSV for name mapping
        raw_image_dir: Directory with raw images for overlay visualizations
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
    gc_path = os.path.join(features_dir, "gc_levels.csv")
    
    aicda_levels = pd.read_csv(aicda_path, index_col=0) if os.path.exists(aicda_path) else None
    cd3_levels = pd.read_csv(cd3_path, index_col=0) if os.path.exists(cd3_path) else None
    gc_levels = pd.read_csv(gc_path, index_col=0) if os.path.exists(gc_path) else None
    
    # Filter to cells inside germinal center if requested
    if filter_gc_inside and gc_levels is not None:
        logger.info("Filtering to cells inside germinal center...")
        # Cells with gc_levels int_mean > 0 are inside the germinal center
        gc_positive_cells = gc_levels[gc_levels['int_mean'] > 0]['nuc_id'].tolist()
        
        original_count = len(nuc_features)
        nuc_features = nuc_features[nuc_features['nuc_id'].isin(gc_positive_cells)]
        
        # Also filter spatial coords and protein levels
        if spatial_coords is not None:
            spatial_coords = spatial_coords[spatial_coords['nuc_id'].isin(gc_positive_cells)]
        if aicda_levels is not None:
            aicda_levels = aicda_levels[aicda_levels['nuc_id'].isin(gc_positive_cells)]
        if cd3_levels is not None:
            cd3_levels = cd3_levels[cd3_levels['nuc_id'].isin(gc_positive_cells)]
        
        logger.info(f"  Filtered from {original_count} to {len(nuc_features)} cells inside GC")
    elif filter_gc_inside and gc_levels is None:
        logger.warning("GC filtering requested but gc_levels.csv not found. Proceeding with all cells.")
    
    # Load feature name mapping if provided
    feature_name_dict = {}
    feature_color_dict = {}
    if feature_description_file and os.path.exists(feature_description_file):
        logger.info(f"Loading feature descriptions from {feature_description_file}")
        try:
            desc_df = pd.read_csv(feature_description_file, index_col=0)
            if 'feature' in desc_df.columns and 'long_name' in desc_df.columns:
                feature_name_dict = dict(zip(desc_df['feature'], desc_df['long_name']))
                nuc_features = nuc_features.rename(columns=feature_name_dict)
                logger.info(f"  Renamed {len(feature_name_dict)} features to readable names")
            
            # Build feature color dict for category coloring in plots
            if 'category' in desc_df.columns:
                category_colors = {
                    "morphology": "tab:blue",
                    "intensity": "tab:green",
                    "boundary": "tab:red",
                    "texture": "tab:cyan",
                    "chromatin condensation": "tab:purple",
                    "moments": "tab:orange"
                }
                feature_color_dict = {
                    row['long_name']: category_colors.get(row['category'], 'tab:gray')
                    for _, row in desc_df.iterrows()
                    if pd.notna(row.get('category'))
                }
        except Exception as e:
            logger.warning(f"Could not load feature descriptions: {e}")
    
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
            nuc_features, aicda_levels, cd3_levels, gc_levels, output_dir, generate_plots
        )
    
    if 'cell_type' in analysis_types or 'classification' in analysis_types:
        logger.info("Running cell type classification analysis...")
        results['cell_type'] = run_cell_type_analysis(
            nuc_features, filtered_features, output_dir, random_seed, generate_plots,
            model_config=model_config
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
            random_seed, border_threshold, generate_plots, raw_image_dir
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
    
    # New analysis types from notebook
    if 'tcell_interaction_dz' in analysis_types:
        logger.info("Running T-cell interaction analysis for DZ B-cells...")
        results['tcell_interaction_dz'] = run_tcell_interaction_subset(
            nuc_features, filtered_features, spatial_coords, output_dir, random_seed,
            cell_type_filter='DZ B-cells', pixel_size=pixel_size,
            contact_radius=contact_radius, signaling_radius=signaling_radius,
            generate_plots=generate_plots, feature_color_dict=feature_color_dict
        )
    
    if 'tcell_interaction_lz' in analysis_types:
        logger.info("Running T-cell interaction analysis for LZ B-cells...")
        results['tcell_interaction_lz'] = run_tcell_interaction_subset(
            nuc_features, filtered_features, spatial_coords, output_dir, random_seed,
            cell_type_filter='LZ B-cells', pixel_size=pixel_size,
            contact_radius=contact_radius, signaling_radius=signaling_radius,
            generate_plots=generate_plots, feature_color_dict=feature_color_dict
        )
    
    if 'dz_prediction_probability' in analysis_types:
        logger.info("Running DZ prediction probability analysis...")
        results['dz_prediction_probability'] = run_dz_prediction_probability_analysis(
            nuc_features, filtered_features, output_dir, random_seed, generate_plots
        )
    
    if 'tcell_fraction_comparison' in analysis_types:
        logger.info("Running T-cell fraction comparison analysis...")
        results['tcell_fraction_comparison'] = run_tcell_fraction_comparison(
            nuc_features, output_dir, generate_plots
        )
    
    if 'boundary_dz' in analysis_types:
        logger.info("Running boundary analysis for DZ B-cells...")
        results['boundary_dz'] = run_boundary_subset_analysis(
            nuc_features, filtered_features, spatial_coords, output_dir,
            random_seed, border_threshold, cell_type_filter='DZ B-cells',
            generate_plots=generate_plots, feature_color_dict=feature_color_dict
        )
    
    if 'boundary_lz' in analysis_types:
        logger.info("Running boundary analysis for LZ B-cells...")
        results['boundary_lz'] = run_boundary_subset_analysis(
            nuc_features, filtered_features, spatial_coords, output_dir,
            random_seed, border_threshold, cell_type_filter='LZ B-cells',
            generate_plots=generate_plots, feature_color_dict=feature_color_dict
        )
    
    if 'enhanced_visualization' in analysis_types or 'enhanced_features' in analysis_types:
        logger.info("Running enhanced features visualization...")
        results['enhanced_visualization'] = run_enhanced_features_visualization(
            features_dir, output_dir, nuc_features
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


def run_cell_type_detection(nuc_features, aicda_levels, cd3_levels, gc_levels, output_dir, generate_plots=True):
    """Run GMM-based cell type detection"""
    from src.analysis.cell_type_detection import assign_cell_types
    
    results_dir = os.path.join(output_dir, "cell_type_detection")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if aicda_levels is None or cd3_levels is None:
        logger.warning("AICDA and/or CD3 levels not available. Cannot detect cell types.")
        return nuc_features, None
    
    # Detect cell types
    nuc_features = assign_cell_types(nuc_features, aicda_levels, cd3_levels)
    
    # Add GC status if available
    if gc_levels is not None:
        gc_positive_cells = gc_levels[gc_levels['int_mean'] > 0]['nuc_id'].tolist()
        nuc_features['gc_status'] = 'outside GC'
        nuc_features.loc[nuc_features['nuc_id'].isin(gc_positive_cells), 'gc_status'] = 'inside GC'
        
        # Save GC status counts
        gc_counts = nuc_features.groupby(['cell_type', 'gc_status']).size().unstack(fill_value=0)
        gc_counts.to_csv(os.path.join(results_dir, "cell_type_by_gc_status.csv"))
    
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
            
            # Cell type distribution by GC status if available
            if 'gc_status' in nuc_features.columns:
                plot_cell_type_distribution(
                    nuc_features,
                    cell_type_col='cell_type',
                    hue_col='gc_status',
                    output_path=os.path.join(results_dir, "cell_type_by_gc_status.png"),
                    title="Cell Type Distribution by GC Status"
                )
            
            logger.info(f"Saved cell type distribution plot")
        except Exception as e:
            logger.warning(f"Could not generate cell type plot: {e}")
    
    return nuc_features, {'cell_type_counts': cell_type_counts.to_dict()}


def run_cell_type_analysis(nuc_features, filtered_features, output_dir, random_seed, generate_plots=True, model_config=None):
    """Run cell type classification analysis with optional model comparison.
    
    Args:
        nuc_features: DataFrame with nuclear features and cell type labels
        filtered_features: Preprocessed feature DataFrame
        output_dir: Output directory
        random_seed: Random seed for reproducibility
        generate_plots: Whether to generate visualization plots
        model_config: Optional dict with model comparison settings:
            - enabled: Whether to compare multiple models
            - models: List of model names (e.g., ['random_forest', 'xgboost'])
            - tune_hyperparameters: Whether to tune hyperparameters
            - use_mlflow: Whether to track with MLflow
            - mlflow_experiment_name: MLflow experiment name
    """
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
    
    # Check if model comparison is enabled
    use_model_comparison = (
        model_config is not None and 
        model_config.get('enabled', False) and
        len(model_config.get('models', [])) > 0
    )
    
    if use_model_comparison:
        # Use model comparison framework
        from src.analysis.statistical_tests import run_model_comparison
        
        models = model_config.get('models', ['random_forest', 'xgboost'])
        use_mlflow = model_config.get('use_mlflow', False)
        mlflow_experiment = model_config.get('mlflow_experiment_name', 'cell_type_classification')
        tune_hp = model_config.get('tune_hyperparameters', False)
        
        logger.info(f"Comparing models: {', '.join(models)}")
        
        cv_results = run_model_comparison(
            bcell_features, 
            bcell_labels,
            models=models,
            n_folds=10, 
            random_state=random_seed, 
            balance=True,
            tune_hyperparameters=tune_hp,
            use_mlflow=use_mlflow,
            mlflow_experiment_name=mlflow_experiment,
            output_dir=results_dir
        )
        
        # Log best model
        logger.info(f"Best model: {cv_results['best_model']}")
        
        # Save model comparison results
        if 'comparison_results' in cv_results:
            cv_results['comparison_results'].to_csv(
                os.path.join(results_dir, "model_comparison.csv"), 
                index=False
            )
    else:
        # Use standard single-model classification
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
                plot_tcell_interaction_zones, plot_feature_importance,
                plot_tcell_distance_spatial
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
            
            # Spatial distribution plots for ALL images (like notebook)
            for img in nuc_features['image'].unique():
                # Merge nuc_features with tcell_distances for this image
                img_nuc_data = nuc_features[nuc_features['image'] == img]
                img_tcell_dist = tcell_distances[tcell_distances['image'] == img]
                
                img_data = img_nuc_data.merge(
                    img_tcell_dist[['nuc_id', 'tcell_mean_distance', 'tcell_median_distance', 'tcell_min_distance']],
                    on='nuc_id', how='left'
                )
                
                # Merge spatial coords if needed
                if 'spat_centroid_x' not in img_data.columns:
                    img_data = img_data.merge(spatial_coords, on='nuc_id', how='left')
                
                # Determine centroid column names
                if 'spat_centroid_x' in img_data.columns:
                    x_col, y_col = 'spat_centroid_x', 'spat_centroid_y'
                elif 'centroid-1' in img_data.columns:
                    x_col, y_col = 'centroid-1', 'centroid-0'
                elif 'centroid_1' in img_data.columns:
                    x_col, y_col = 'centroid_1', 'centroid_0'
                else:
                    continue
                
                # 3-panel T-cell distance spatial plot (like notebook)
                plot_tcell_distance_spatial(
                    img_data,
                    x_col=x_col,
                    y_col=y_col,
                    cell_type_col='cell_type',
                    output_path=os.path.join(results_dir, f"tcell_distance_spatial_img{img}.png"),
                    title=f"T-cell Distance Analysis - Image {img}"
                )
                
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
    
    # Add correlation analysis between chromatin features and T-cell distance
    if 'tcell_mean_distance' in nuc_features.columns:
        try:
            from src.analysis.statistical_tests import permutation_test_correlation
            from src.analysis.visualization import plot_correlation_lm
            
            # Filter to B-cells
            bcells = nuc_features[nuc_features['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])].copy()
            
            # Test correlation for key features
            key_features = ['min_intensity', 'mean_intensity', 'area']
            correlation_results = []
            
            for feature in key_features:
                if feature not in bcells.columns:
                    continue
                    
                # All B-cells
                all_bcells_data = bcells[[feature, 'tcell_mean_distance']].dropna()
                if len(all_bcells_data) > 50:
                    r, p = permutation_test_correlation(
                        all_bcells_data[feature].values,
                        all_bcells_data['tcell_mean_distance'].values,
                        n_permutations=10000
                    )
                    correlation_results.append({
                        'feature': feature,
                        'cell_type': 'all B-cells',
                        'pearson_r': r,
                        'p_value': p
                    })
                    logger.info(f"  {feature} vs tcell_mean_distance (all B-cells): r={r:.3f}, p={p:.2e}")
                
                # DZ B-cells
                dz_data = bcells[bcells['cell_type'] == 'DZ B-cells'][[feature, 'tcell_mean_distance']].dropna()
                if len(dz_data) > 50:
                    r, p = permutation_test_correlation(
                        dz_data[feature].values,
                        dz_data['tcell_mean_distance'].values,
                        n_permutations=10000
                    )
                    correlation_results.append({
                        'feature': feature,
                        'cell_type': 'DZ B-cells',
                        'pearson_r': r,
                        'p_value': p
                    })
                
                # LZ B-cells
                lz_data = bcells[bcells['cell_type'] == 'LZ B-cells'][[feature, 'tcell_mean_distance']].dropna()
                if len(lz_data) > 50:
                    r, p = permutation_test_correlation(
                        lz_data[feature].values,
                        lz_data['tcell_mean_distance'].values,
                        n_permutations=10000
                    )
                    correlation_results.append({
                        'feature': feature,
                        'cell_type': 'LZ B-cells',
                        'pearson_r': r,
                        'p_value': p
                    })
            
            # Save correlation results
            if correlation_results:
                pd.DataFrame(correlation_results).to_csv(
                    os.path.join(results_dir, "tcell_distance_correlations.csv"), index=False
                )
            
            # Generate lmplot for min_intensity vs tcell_mean_distance (like notebook)
            if generate_plots and 'min_intensity' in bcells.columns:
                plot_data = bcells[['cell_type', 'min_intensity', 'tcell_mean_distance']].copy()
                plot_data_all = bcells[['min_intensity', 'tcell_mean_distance']].copy()
                plot_data_all['cell_type'] = 'all B-cells'
                plot_data_combined = pd.concat([plot_data, plot_data_all], ignore_index=True)
                
                plot_correlation_lm(
                    plot_data_combined,
                    x_col='tcell_mean_distance',
                    y_col='min_intensity',
                    hue_col='cell_type',
                    output_path=os.path.join(results_dir, "min_intensity_vs_tcell_distance.png"),
                    title="Min Intensity vs T-cell Mean Distance"
                )
                
        except Exception as corr_e:
            logger.warning(f"Could not compute correlations: {corr_e}")
    
    return nuc_features, {'influence_counts': influence_counts.to_dict()}


def run_boundary_analysis(
    nuc_features, filtered_features, spatial_coords, output_dir, 
    random_seed, border_threshold=0.4, generate_plots=True, raw_image_dir=None
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
            from src.analysis.visualization import (
                plot_spatial_scatter, plot_boundary_analysis_per_image
            )
            
            # Generate comprehensive per-image plots (like notebook)
            for img in border_distances['image'].unique():
                img_data = border_distances[border_distances['image'] == img]
                
                # Determine centroid columns
                if 'centroid-1' in img_data.columns:
                    x_col, y_col = 'centroid-1', 'centroid-0'
                elif 'centroid_1' in img_data.columns:
                    x_col, y_col = 'centroid_1', 'centroid_0'
                else:
                    continue
                
                # Try to create 3-panel plot like notebook
                try:
                    plot_boundary_analysis_per_image(
                        img_data,
                        x_col=x_col,
                        y_col=y_col,
                        output_path=os.path.join(results_dir, f"boundary_analysis_img{img}.png"),
                        title=f"Boundary Analysis - Image {img}"
                    )
                except Exception as e:
                    # Fall back to simple scatter
                    plot_spatial_scatter(
                        img_data,
                        x_col=x_col,
                        y_col=y_col,
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


def run_tcell_interaction_subset(
    nuc_features, filtered_features, spatial_coords, output_dir, random_seed,
    cell_type_filter: str = 'DZ B-cells', pixel_size=0.3225,
    contact_radius=15.0, signaling_radius=30.0, generate_plots=True, feature_color_dict=None
):
    """Run T-cell interaction analysis for a specific B-cell subset (DZ or LZ)"""
    from src.analysis.statistical_tests import run_cv_classification, find_markers
    
    safe_name = cell_type_filter.replace(' ', '_').replace('-', '_').lower()
    results_dir = os.path.join(output_dir, f"tcell_interaction_{safe_name}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if 'cell_type' not in nuc_features.columns or 'tcell_influence' not in nuc_features.columns:
        logger.warning(f"Required columns not available for {cell_type_filter} T-cell interaction analysis.")
        return None
    
    # Filter to specified B-cell type with T-cell interaction labels
    bcell_mask = nuc_features['cell_type'] == cell_type_filter
    interaction_mask = nuc_features['tcell_influence'].isin(['T-cell interactors', 'Non-T-cell interactors'])
    analysis_mask = bcell_mask & interaction_mask
    
    analysis_features = nuc_features[analysis_mask]
    analysis_nuc_ids = analysis_features['nuc_id'].values
    common_idx = [idx for idx in analysis_nuc_ids if idx in filtered_features.index]
    
    if len(common_idx) < 100:
        logger.warning(f"Too few {cell_type_filter} for T-cell interaction analysis ({len(common_idx)} samples).")
        return None
    
    X = filtered_features.loc[common_idx]
    analysis_features_indexed = analysis_features.set_index('nuc_id')
    y = analysis_features_indexed.loc[common_idx, 'tcell_influence']
    
    logger.info(f"Running {cell_type_filter} T-cell interaction classification on {len(X)} samples...")
    cv_results = run_cv_classification(X, y, n_folds=10, random_state=random_seed, balance=True)
    
    # Save results
    cv_summary = pd.DataFrame([{
        'cell_type': cell_type_filter,
        'balanced_accuracy_mean': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(X)
    }])
    cv_summary.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
    
    cv_results['feature_importance'].to_csv(
        os.path.join(results_dir, "feature_importance.csv"), index=False
    )
    
    # Find markers
    markers = find_markers(X, y, test='welch')
    markers.to_csv(os.path.join(results_dir, "marker_features.csv"), index=False)
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import (
                plot_confusion_matrix, plot_feature_importance_colored,
                plot_roc_binary, plot_violin_with_stats
            )
            
            # Confusion matrix
            plot_confusion_matrix(
                cv_results['true_labels'],
                cv_results['predictions'],
                list(cv_results['classes']),
                os.path.join(results_dir, "confusion_matrix.png")
            )
            
            # Feature importance with category coloring
            plot_feature_importance_colored(
                cv_results['feature_importance']['importance'].values,
                cv_results['feature_importance']['feature'].tolist(),
                os.path.join(results_dir, "feature_importance.png"),
                feature_color_dict=feature_color_dict,
                title=f"{cell_type_filter} T-cell Interaction Feature Importance",
                n_features=15
            )
            
            # ROC curve
            if 'probabilities' in cv_results and cv_results['probabilities'] is not None:
                probs = cv_results['probabilities']
                if probs.ndim == 2:
                    pos_idx = list(cv_results['classes']).index('T-cell interactors') if 'T-cell interactors' in cv_results['classes'] else 0
                    probs = probs[:, pos_idx]
                
                plot_roc_binary(
                    cv_results['true_labels'],
                    probs,
                    pos_label='T-cell interactors',
                    output_path=os.path.join(results_dir, "roc_curve.png"),
                    title=f"ROC Curve - {cell_type_filter} T-cell Interaction"
                )
            
            # Violin plots for top markers with stats
            sig_markers = markers[markers['adjusted_pval'] < 0.05]
            if len(sig_markers) > 0:
                plot_data = X.copy()
                plot_data['tcell_influence'] = y.values
                
                plot_violin_with_stats(
                    plot_data,
                    sig_markers['feature'].head(6).tolist(),
                    group_col='tcell_influence',
                    output_path=os.path.join(results_dir, "marker_violin_plots.png"),
                    title=f"{cell_type_filter} - Top Differential Features"
                )
                
        except Exception as e:
            logger.warning(f"Could not generate plots for {cell_type_filter}: {e}")
    
    logger.info(f"  {cell_type_filter} T-cell interaction: Balanced accuracy {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})")
    
    return {
        'balanced_accuracy': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(X)
    }


def run_dz_prediction_probability_analysis(
    nuc_features, filtered_features, output_dir, random_seed, generate_plots=True
):
    """
    Train classifier on non-T-cell interactors only, then use DZ prediction probability
    as proxy for DZ-like phenotype to analyze relationship with T-cell interaction.
    """
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn import metrics
    
    results_dir = os.path.join(output_dir, "dz_prediction_probability")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if 'cell_type' not in nuc_features.columns or 'tcell_influence' not in nuc_features.columns:
        logger.warning("Required columns not available for DZ prediction probability analysis.")
        return None
    
    # Get B-cells only
    bcell_mask = nuc_features['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
    bcell_features = nuc_features[bcell_mask]
    
    # Separate T-cell interactors and non-interactors
    non_interactors_mask = bcell_features['tcell_influence'] == 'Non-T-cell interactors'
    interactors_mask = bcell_features['tcell_influence'] == 'T-cell interactors'
    
    non_interactors = bcell_features[non_interactors_mask]
    interactors = bcell_features[interactors_mask]
    
    if len(non_interactors) < 100 or len(interactors) < 50:
        logger.warning("Too few samples for DZ prediction probability analysis.")
        return None
    
    # Balance training set by cell type (DZ vs LZ)
    non_interactors_dz = non_interactors[non_interactors['cell_type'] == 'DZ B-cells']
    non_interactors_lz = non_interactors[non_interactors['cell_type'] == 'LZ B-cells']
    
    n_train = int(0.8 * min(len(non_interactors_dz), len(non_interactors_lz)))
    
    np.random.seed(random_seed)
    train_idx = list(np.random.choice(non_interactors_dz.index, size=n_train, replace=False))
    train_idx += list(np.random.choice(non_interactors_lz.index, size=n_train, replace=False))
    
    # Test set: remaining non-interactors + all interactors
    test_idx = [idx for idx in non_interactors.index if idx not in train_idx]
    test_idx += list(interactors.index)
    
    # Get features
    nuc_features_by_nucid = nuc_features.set_index('nuc_id')
    train_nuc_ids = nuc_features.loc[train_idx, 'nuc_id'].values
    test_nuc_ids = nuc_features.loc[test_idx, 'nuc_id'].values
    
    train_common = [idx for idx in train_nuc_ids if idx in filtered_features.index]
    test_common = [idx for idx in test_nuc_ids if idx in filtered_features.index]
    
    if len(train_common) < 50 or len(test_common) < 50:
        logger.warning("Too few samples after feature matching for DZ prediction probability analysis.")
        return None
    
    X_train = filtered_features.loc[train_common]
    X_test = filtered_features.loc[test_common]
    
    y_train = nuc_features_by_nucid.loc[train_common, 'cell_type']
    y_test = nuc_features_by_nucid.loc[test_common, 'cell_type']
    tcell_influence_test = nuc_features_by_nucid.loc[test_common, 'tcell_influence']
    
    # Train classifier
    logger.info(f"Training DZ/LZ classifier on {len(X_train)} non-T-cell interacting B-cells...")
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_seed)
    rfc.fit(X_train, y_train)
    
    # Get predictions and probabilities for test set
    predictions = rfc.predict(X_test)
    probabilities = rfc.predict_proba(X_test)
    
    # Get DZ probability (probability of being DZ B-cells)
    dz_class_idx = list(rfc.classes_).index('DZ B-cells') if 'DZ B-cells' in rfc.classes_ else 0
    dz_probs = probabilities[:, dz_class_idx]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'nuc_id': test_common,
        'true_cell_type': y_test.values,
        'predicted_cell_type': predictions,
        'dz_probability': dz_probs,
        'tcell_influence': tcell_influence_test.values
    })
    results_df.to_csv(os.path.join(results_dir, "prediction_results.csv"), index=False)
    
    # Compute mean DZ probability by cell type and T-cell influence
    grouped_means = results_df.groupby(['true_cell_type', 'tcell_influence'])['dz_probability'].agg(['mean', 'std', 'count'])
    grouped_means.to_csv(os.path.join(results_dir, "dz_probability_by_group.csv"))
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import plot_dz_probability_violin, plot_confusion_matrix
            
            # Confusion matrix on test set
            conf_mtx = metrics.confusion_matrix(y_test, predictions, labels=list(rfc.classes_))
            conf_mtx_norm = conf_mtx.astype(float) / conf_mtx.sum(axis=1, keepdims=True)
            
            plot_confusion_matrix(
                y_test.values,
                predictions,
                list(rfc.classes_),
                os.path.join(results_dir, "confusion_matrix.png")
            )
            
            # DZ probability violin plot by cell type and T-cell influence
            plot_dz_probability_violin(
                results_df,
                output_path=os.path.join(results_dir, "dz_probability_violin.png"),
                title="DZ B-cell Prediction Probability by T-cell Interaction"
            )
            
        except Exception as e:
            logger.warning(f"Could not generate plots for DZ prediction probability: {e}")
    
    logger.info(f"  DZ prediction probability analysis complete")
    
    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': (predictions == y_test.values).mean()
    }


def run_tcell_fraction_comparison(nuc_features, output_dir, generate_plots=True):
    """
    Compare the fraction of T-cell interacting B-cells between DZ and LZ.
    Reproduces the bar plots from the notebook with statistical annotations.
    """
    from scipy import stats
    
    results_dir = os.path.join(output_dir, "tcell_fraction_comparison")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if 'cell_type' not in nuc_features.columns or 'tcell_influence' not in nuc_features.columns:
        logger.warning("Required columns not available for T-cell fraction comparison.")
        return None
    
    # Get B-cells only
    bcell_mask = nuc_features['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
    bcells = nuc_features[bcell_mask]
    
    # Compute per-image statistics
    image_stats = []
    for image in bcells['image'].unique():
        img_bcells = bcells[bcells['image'] == image]
        
        dz_bcells = img_bcells[img_bcells['cell_type'] == 'DZ B-cells']
        lz_bcells = img_bcells[img_bcells['cell_type'] == 'LZ B-cells']
        
        tcell_interactors = img_bcells[img_bcells['tcell_influence'] == 'T-cell interactors']
        
        n_total = len(img_bcells)
        n_tcell_interactors = len(tcell_interactors)
        
        if n_tcell_interactors > 0:
            dz_tcell_interactors = len(dz_bcells[dz_bcells['tcell_influence'] == 'T-cell interactors'])
            lz_tcell_interactors = len(lz_bcells[lz_bcells['tcell_influence'] == 'T-cell interactors'])
            
            freq_dz_tcell_interactors = dz_tcell_interactors / n_tcell_interactors
            freq_lz_tcell_interactors = lz_tcell_interactors / n_tcell_interactors
        else:
            freq_dz_tcell_interactors = 0
            freq_lz_tcell_interactors = 0
        
        # Fraction of T-cell interactors within each zone
        n_dz = len(dz_bcells)
        n_lz = len(lz_bcells)
        
        freq_tcell_interactors_in_dz = len(dz_bcells[dz_bcells['tcell_influence'] == 'T-cell interactors']) / n_dz if n_dz > 0 else 0
        freq_tcell_interactors_in_lz = len(lz_bcells[lz_bcells['tcell_influence'] == 'T-cell interactors']) / n_lz if n_lz > 0 else 0
        
        image_stats.append({
            'image': image,
            'n_bcells': n_total,
            'n_dz_bcells': n_dz,
            'n_lz_bcells': n_lz,
            'n_tcell_interactors': n_tcell_interactors,
            'freq_dz_of_tcell_interactors': freq_dz_tcell_interactors,
            'freq_lz_of_tcell_interactors': freq_lz_tcell_interactors,
            'freq_tcell_interactors_in_dz': freq_tcell_interactors_in_dz,
            'freq_tcell_interactors_in_lz': freq_tcell_interactors_in_lz
        })
    
    stats_df = pd.DataFrame(image_stats)
    stats_df.to_csv(os.path.join(results_dir, "per_image_statistics.csv"), index=False)
    
    # Statistical tests
    dz_fractions = stats_df['freq_tcell_interactors_in_dz'].values
    lz_fractions = stats_df['freq_tcell_interactors_in_lz'].values
    
    ttest_result = stats.ttest_ind(dz_fractions, lz_fractions, equal_var=False)
    
    test_results = pd.DataFrame([{
        'comparison': 'DZ vs LZ fraction of T-cell interactors',
        't_statistic': ttest_result.statistic,
        'p_value': ttest_result.pvalue
    }])
    test_results.to_csv(os.path.join(results_dir, "statistical_tests.csv"), index=False)
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import plot_tcell_fraction_comparison
            
            plot_tcell_fraction_comparison(
                stats_df,
                output_path=os.path.join(results_dir, "tcell_fraction_comparison.png"),
                title="Fraction of T-cell Interacting B-cells"
            )
            
        except Exception as e:
            logger.warning(f"Could not generate plots for T-cell fraction comparison: {e}")
    
    logger.info(f"  T-cell fraction comparison: DZ={np.mean(dz_fractions):.3f}, LZ={np.mean(lz_fractions):.3f}, p={ttest_result.pvalue:.2e}")
    
    return {
        'mean_dz_fraction': np.mean(dz_fractions),
        'mean_lz_fraction': np.mean(lz_fractions),
        'p_value': ttest_result.pvalue
    }


def run_boundary_subset_analysis(
    nuc_features, filtered_features, spatial_coords, output_dir,
    random_seed, border_threshold, cell_type_filter, generate_plots=True, feature_color_dict=None
):
    """Run boundary analysis for a specific B-cell subset (DZ or LZ)"""
    from src.analysis.boundary_analysis import get_distances_to_dz_lz_border, assign_border_proximity
    from src.analysis.statistical_tests import run_cv_classification, find_markers
    
    safe_name = cell_type_filter.replace(' ', '_').replace('-', '_').lower()
    results_dir = os.path.join(output_dir, f"boundary_{safe_name}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if spatial_coords is None or 'cell_type' not in nuc_features.columns:
        logger.warning(f"Required data not available for {cell_type_filter} boundary analysis.")
        return None
    
    # Compute distances to DZ/LZ border
    border_distances = get_distances_to_dz_lz_border(nuc_features, spatial_coords)
    
    if len(border_distances) == 0:
        logger.warning(f"Could not compute border distances for {cell_type_filter}.")
        return None
    
    # Assign proximity status
    border_distances = assign_border_proximity(border_distances, threshold=border_threshold)
    
    # Filter to specified cell type
    cell_type_data = border_distances[border_distances['cell_type'] == cell_type_filter]
    
    if len(cell_type_data) < 100:
        logger.warning(f"Too few {cell_type_filter} for boundary analysis ({len(cell_type_data)} samples).")
        return None
    
    # Get features
    common_idx = cell_type_data['nuc_id'].values
    common_idx = [idx for idx in common_idx if idx in filtered_features.index]
    
    if len(common_idx) < 50:
        logger.warning(f"Too few matched samples for {cell_type_filter} boundary analysis.")
        return None
    
    X = filtered_features.loc[common_idx]
    y = cell_type_data.set_index('nuc_id').loc[common_idx, 'border_proximity']
    
    # Run classification
    logger.info(f"Running {cell_type_filter} boundary classification on {len(X)} samples...")
    cv_results = run_cv_classification(X, y, n_folds=10, random_state=random_seed, balance=True)
    
    # Save results
    cv_summary = pd.DataFrame([{
        'cell_type': cell_type_filter,
        'balanced_accuracy_mean': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(X)
    }])
    cv_summary.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
    
    cv_results['feature_importance'].to_csv(
        os.path.join(results_dir, "feature_importance.csv"), index=False
    )
    
    # Find markers
    markers = find_markers(X, y, test='welch')
    markers.to_csv(os.path.join(results_dir, "marker_features.csv"), index=False)
    
    # Generate plots
    if generate_plots:
        try:
            from src.analysis.visualization import (
                plot_confusion_matrix, plot_feature_importance_colored,
                plot_roc_binary, plot_violin_with_stats
            )
            
            # Confusion matrix
            plot_confusion_matrix(
                cv_results['true_labels'],
                cv_results['predictions'],
                list(cv_results['classes']),
                os.path.join(results_dir, "confusion_matrix.png")
            )
            
            # Feature importance with category coloring
            plot_feature_importance_colored(
                cv_results['feature_importance']['importance'].values,
                cv_results['feature_importance']['feature'].tolist(),
                os.path.join(results_dir, "feature_importance.png"),
                feature_color_dict=feature_color_dict,
                title=f"{cell_type_filter} Boundary Classification Feature Importance",
                n_features=15
            )
            
            # ROC curve
            if 'probabilities' in cv_results and cv_results['probabilities'] is not None:
                probs = cv_results['probabilities']
                if probs.ndim == 2:
                    pos_idx = list(cv_results['classes']).index('close') if 'close' in cv_results['classes'] else 0
                    probs = probs[:, pos_idx]
                
                plot_roc_binary(
                    cv_results['true_labels'],
                    probs,
                    pos_label='close',
                    output_path=os.path.join(results_dir, "roc_curve.png"),
                    title=f"ROC Curve - {cell_type_filter} Boundary Proximity"
                )
            
            # Violin plots for top markers
            sig_markers = markers[markers['adjusted_pval'] < 0.05]
            if len(sig_markers) > 0:
                plot_data = X.copy()
                plot_data['border_proximity'] = y.values
                
                plot_violin_with_stats(
                    plot_data,
                    sig_markers['feature'].head(6).tolist(),
                    group_col='border_proximity',
                    output_path=os.path.join(results_dir, "marker_violin_plots.png"),
                    title=f"{cell_type_filter} - Top Differential Features by Border Proximity"
                )
                
        except Exception as e:
            logger.warning(f"Could not generate plots for {cell_type_filter} boundary: {e}")
    
    logger.info(f"  {cell_type_filter} boundary classification: {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})")
    
    return {
        'balanced_accuracy': cv_results['cv_mean'],
        'balanced_accuracy_std': cv_results['cv_std'],
        'n_samples': len(X)
    }


def run_enhanced_features_visualization(
    features_dir: str,
    output_dir: str,
    nuc_features: Optional[pd.DataFrame] = None
):
    """Run enhanced features visualization analysis.
    
    Args:
        features_dir: Directory containing feature files
        output_dir: Output directory for analysis results
        nuc_features: Optional DataFrame with annotated features (for DZ/LZ labels)
        
    Returns:
        Dictionary with visualization results
    """
    try:
        from pathlib import Path
        import glob
        import importlib.util
        
        # Import visualization functions from the script
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'visualize_enhanced_features.py'
        if not script_path.exists():
            logger.warning("Enhanced features visualization script not found. Skipping.")
            return None
        
        # Import the visualization functions directly as a module
        spec = importlib.util.spec_from_file_location("visualize_enhanced_features", script_path)
        viz_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_module)
        
        # Get the functions
        load_features = viz_module.load_features
        plot_feature_distributions = viz_module.plot_feature_distributions
        plot_cell_cycle_analysis = viz_module.plot_cell_cycle_analysis
        plot_wavelet_analysis = viz_module.plot_wavelet_analysis
        plot_dz_lz_comparison = viz_module.plot_dz_lz_comparison
        plot_feature_correlations = viz_module.plot_feature_correlations
        generate_summary_statistics = viz_module.generate_summary_statistics
        
        # Find enhanced features CSV
        enhanced_features_path = None
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(features_dir, '..', 'enhanced_features', 'enhanced_features.csv'),
            os.path.join(features_dir, '..', '..', 'features_enhanced', 'enhanced_features', 'enhanced_features.csv'),
            os.path.join(output_dir, '..', 'features_enhanced', 'enhanced_features', 'enhanced_features.csv'),
        ]
        
        # Also search in the features directory
        if os.path.exists(features_dir):
            csv_files = glob.glob(os.path.join(features_dir, '**', 'enhanced_features.csv'), recursive=True)
            if csv_files:
                enhanced_features_path = csv_files[0]
        
        # Try the possible paths
        if not enhanced_features_path:
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    enhanced_features_path = abs_path
                    break
        
        if not enhanced_features_path or not os.path.exists(enhanced_features_path):
            logger.warning("Enhanced features CSV not found. Skipping enhanced features visualization.")
            logger.info(f"  Searched in: {features_dir} and related directories")
            return None
        
        logger.info(f"Found enhanced features at: {enhanced_features_path}")
        
        # Create output directory
        enhanced_output_dir = Path(output_dir) / 'enhanced_features_analysis'
        enhanced_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load features
        df = load_features(enhanced_features_path)
        
        # Define feature groups
        feature_groups = {
            'wavelet': [c for c in df.columns if 'wavelet' in c],
            'fractal': [c for c in df.columns if 'fractal' in c],
            'domain': [c for c in df.columns if 'domain' in c],
            'radial': [c for c in df.columns if 'radial' in c],
            'cell_cycle': [c for c in df.columns if 'cell_cycle' in c or 'bright_foci' in c or 'condensation' in c],
            'spatial': [c for c in df.columns if 'voronoi' in c or 'centrality' in c or 'density' in c]
        }
        
        # Determine group-by column
        group_by = 'cell_cycle_phase' if 'cell_cycle_phase' in df.columns else None
        
        # Determine DZ/LZ label column if available
        dz_label_col = None
        if nuc_features is not None:
            # Check if we have cell type or predicted labels
            if 'cell_type' in nuc_features.columns:
                # Merge cell type if possible
                if 'nuc_id' in df.columns and 'nuc_id' in nuc_features.columns:
                    df_merged = df.merge(
                        nuc_features[['nuc_id', 'cell_type']],
                        on='nuc_id',
                        how='left'
                    )
                    # Create a simplified label
                    df_merged['zone'] = df_merged['cell_type'].apply(
                        lambda x: 'DZ' if 'DZ' in str(x) else ('LZ' if 'LZ' in str(x) else None)
                    )
                    if df_merged['zone'].notna().sum() > 0:
                        df = df_merged
                        dz_label_col = 'zone'
            elif 'predicted' in nuc_features.columns:
                if 'nuc_id' in df.columns and 'nuc_id' in nuc_features.columns:
                    df = df.merge(
                        nuc_features[['nuc_id', 'predicted']],
                        on='nuc_id',
                        how='left'
                    )
                    dz_label_col = 'predicted'
        
        # Generate plots
        logger.info("  Generating feature distribution plots...")
        plot_feature_distributions(
            df, feature_groups, enhanced_output_dir,
            group_col=group_by
        )
        
        logger.info("  Generating cell cycle analysis...")
        plot_cell_cycle_analysis(df, enhanced_output_dir)
        
        logger.info("  Generating wavelet analysis...")
        plot_wavelet_analysis(df, enhanced_output_dir)
        
        if dz_label_col and dz_label_col in df.columns:
            logger.info(f"  Generating DZ/LZ comparison using '{dz_label_col}' column...")
            plot_dz_lz_comparison(df, enhanced_output_dir, dz_label_col)
        
        logger.info("  Generating correlation matrix...")
        plot_feature_correlations(df, enhanced_output_dir)
        
        logger.info("  Generating summary statistics...")
        generate_summary_statistics(df, enhanced_output_dir)
        
        logger.info(f"Enhanced features visualization complete! Figures saved to {enhanced_output_dir}")
        
        return {
            'output_dir': str(enhanced_output_dir),
            'n_features': len(df.columns),
            'n_cells': len(df),
            'figures_generated': True
        }
        
    except ImportError as e:
        logger.warning(f"Could not import enhanced features visualization functions: {e}")
        logger.warning("Skipping enhanced features visualization.")
        return None
    except Exception as e:
        logger.warning(f"Error generating enhanced features visualization: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
