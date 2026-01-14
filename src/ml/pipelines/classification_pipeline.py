# -*- coding: utf-8 -*-
"""
ZenML Classification Pipeline for Germinal Center analysis.

Provides production-grade ML pipelines with:
- Data loading and preprocessing steps
- Model training with MLflow tracking
- Model comparison across multiple algorithms
- Feature importance analysis

Note: This module requires ZenML and MLflow to be installed.
Install with: pip install zenml[mlflow] mlflow
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for ZenML availability
try:
    from zenml import step, pipeline, get_step_context
    from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    logger.warning("ZenML not installed. Pipeline functions will be regular Python functions.")
    
    # Create dummy decorators
    def step(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def pipeline(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def get_step_context():
        return None


# MLflow settings for ZenML
mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True
) if ZENML_AVAILABLE else None


@step(enable_cache=True)
def load_data_step(
    features_path: str,
    analysis_type: str = "cell_type",
    use_enhanced: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features and labels for classification.
    
    Args:
        features_path: Path to features CSV file or directory
        analysis_type: Type of analysis ("cell_type", "tcell_interaction", "boundary")
        use_enhanced: Whether to use enhanced (merged) features if available
        
    Returns:
        Tuple of (features DataFrame, labels Series, metadata DataFrame)
    """
    features_path = Path(features_path)
    
    # Determine features file
    if features_path.is_dir():
        # Look for merged features first if enhanced
        if use_enhanced:
            merged_path = features_path / "all_features_merged.csv"
            if merged_path.exists():
                features_file = merged_path
            else:
                features_file = features_path / "nuc_features.csv"
        else:
            features_file = features_path / "nuc_features.csv"
    else:
        features_file = features_path
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    # Load features
    df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(df)} samples from {features_file}")
    
    # Define metadata columns
    meta_cols = [
        'label', 'label_id', 'nuc_id', 'image',
        'weighted_centroid-0', 'weighted_centroid-1',
        'centroid-0', 'centroid-1', 'centroid_x', 'centroid_y',
        'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
        'orientation', 'cell_type', 'aicda_status', 'cd3_status',
        'gc_status', 'tcell_influence', 'border_proximity',
        'cell_cycle_phase'
    ]
    
    # Extract labels based on analysis type
    if analysis_type == "cell_type":
        # Filter to B-cells for DZ vs LZ classification
        if 'cell_type' not in df.columns:
            raise ValueError("cell_type column not found. Run cell type detection first.")
        
        bcell_mask = df['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
        df_filtered = df[bcell_mask].copy()
        labels = df_filtered['cell_type']
        
    elif analysis_type == "tcell_interaction":
        # T-cell interactors vs non-interactors
        if 'tcell_influence' not in df.columns:
            raise ValueError("tcell_influence column not found. Run T-cell analysis first.")
        
        # Filter to B-cells with interaction status
        bcell_mask = df['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
        interaction_mask = df['tcell_influence'].isin(['T-cell interactors', 'Non-T-cell interactors'])
        df_filtered = df[bcell_mask & interaction_mask].copy()
        labels = df_filtered['tcell_influence']
        
    elif analysis_type == "boundary":
        # Close vs distant to boundary
        if 'border_proximity' not in df.columns:
            raise ValueError("border_proximity column not found. Run boundary analysis first.")
        
        bcell_mask = df['cell_type'].isin(['DZ B-cells', 'LZ B-cells'])
        df_filtered = df[bcell_mask].copy()
        labels = df_filtered['border_proximity']
        
    elif analysis_type == "cell_cycle":
        # Cell cycle phase classification
        if 'cell_cycle_phase' not in df.columns:
            raise ValueError("cell_cycle_phase column not found.")
        
        df_filtered = df.copy()
        labels = df_filtered['cell_cycle_phase']
        
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Extract feature columns (numeric, non-metadata)
    feature_cols = [c for c in df_filtered.columns 
                   if c not in meta_cols and df_filtered[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    features = df_filtered[feature_cols]
    metadata = df_filtered[[c for c in meta_cols if c in df_filtered.columns]]
    
    # Remove columns with all NaN
    features = features.dropna(axis=1, how='all')
    
    # Remove rows with any NaN
    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]
    metadata = metadata[valid_mask]
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    
    return features, labels, metadata


@step(enable_cache=True)
def preprocess_data_step(
    features: pd.DataFrame,
    labels: pd.Series,
    correlation_threshold: float = 0.8,
    remove_constant: bool = True,
    scale_features: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess features for classification.
    
    Args:
        features: Raw features DataFrame
        labels: Labels Series
        correlation_threshold: Remove features with correlation above this
        remove_constant: Remove constant features
        scale_features: Whether to standardize features
        
    Returns:
        Tuple of (processed features, labels)
    """
    X = features.copy()
    y = labels.copy()
    
    # Remove constant features
    if remove_constant:
        constant_cols = X.columns[X.std() == 0]
        X = X.drop(columns=constant_cols)
        logger.info(f"Removed {len(constant_cols)} constant features")
    
    # Remove highly correlated features
    if correlation_threshold < 1.0:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
        X = X.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} correlated features (threshold: {correlation_threshold})")
    
    # Scale features
    if scale_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        logger.info("Standardized features")
    
    logger.info(f"Preprocessed: {len(X)} samples, {len(X.columns)} features")
    
    return X, y


@step(enable_cache=True)
def balance_data_step(
    features: pd.DataFrame,
    labels: pd.Series,
    method: str = "undersample",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance class distribution.
    
    Args:
        features: Features DataFrame
        labels: Labels Series
        method: Balancing method ("undersample", "oversample", "smote", "none")
        random_state: Random seed
        
    Returns:
        Tuple of (balanced features, balanced labels)
    """
    if method == "none":
        return features, labels
    
    try:
        if method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == "oversample":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=random_state)
        elif method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        X_bal, y_bal = sampler.fit_resample(features, labels)
        X_bal = pd.DataFrame(X_bal, columns=features.columns)
        y_bal = pd.Series(y_bal)
        
        logger.info(f"Balanced dataset: {len(features)} -> {len(X_bal)} samples")
        
        return X_bal, y_bal
        
    except ImportError:
        logger.warning("imbalanced-learn not installed. Using unbalanced data.")
        return features, labels


@step(enable_cache=False)
def train_model_step(
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str = "random_forest",
    n_folds: int = 10,
    tune_hyperparameters: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train and evaluate a model with cross-validation.
    
    Args:
        features: Features DataFrame
        labels: Labels Series
        model_name: Name of model to train
        n_folds: Number of CV folds
        tune_hyperparameters: Whether to tune hyperparameters
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, classification_report,
        f1_score, roc_auc_score
    )
    from src.ml.model_registry import get_model_wrapper
    
    # Get model wrapper
    wrapper = get_model_wrapper(model_name)
    
    # Get parameters
    if tune_hyperparameters:
        from sklearn.model_selection import RandomizedSearchCV
        
        model = wrapper.get_model()
        param_grid = wrapper.get_param_grid()
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=50, cv=cv,
            scoring='balanced_accuracy', n_jobs=-1, random_state=random_state
        )
        search.fit(features, labels)
        best_params = search.best_params_
        logger.info(f"Best params: {best_params}")
    else:
        best_params = wrapper.get_default_params()
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model = wrapper.get_model(**best_params)
    
    y_array = labels.values if hasattr(labels, 'values') else np.array(labels)
    y_pred = cross_val_predict(model, features, y_array, cv=cv)
    
    try:
        y_proba = cross_val_predict(model, features, y_array, cv=cv, method='predict_proba')
    except:
        y_proba = None
    
    # Compute metrics
    balanced_acc = balanced_accuracy_score(y_array, y_pred)
    f1 = f1_score(y_array, y_pred, average='weighted')
    
    # Per-fold metrics
    fold_scores = []
    for train_idx, test_idx in cv.split(features, y_array):
        m = wrapper.get_model(**best_params)
        m.fit(features.iloc[train_idx], y_array[train_idx])
        y_pred_fold = m.predict(features.iloc[test_idx])
        fold_scores.append(balanced_accuracy_score(y_array[test_idx], y_pred_fold))
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    # Fit final model
    final_model = wrapper.get_model(**best_params)
    final_model.fit(features, y_array)
    
    # Feature importance
    if wrapper.supports_feature_importance and hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame()
    
    result = {
        'model_name': model_name,
        'model': final_model,
        'cv_balanced_accuracy_mean': cv_mean,
        'cv_balanced_accuracy_std': cv_std,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'best_params': best_params,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'probabilities': y_proba,
        'true_labels': y_array,
        'classes': final_model.classes_ if hasattr(final_model, 'classes_') else np.unique(y_array)
    }
    
    logger.info(f"{model_name}: Balanced accuracy = {cv_mean:.3f} ± {cv_std:.3f}")
    
    # Log to MLflow if in ZenML context
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_params(best_params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("balanced_accuracy_mean", cv_mean)
            mlflow.log_metric("balanced_accuracy_std", cv_std)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(final_model, "model")
    except:
        pass
    
    return result


@step(enable_cache=False)
def compare_models_step(
    features: pd.DataFrame,
    labels: pd.Series,
    models: List[str] = None,
    n_folds: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """Compare multiple models.
    
    Args:
        features: Features DataFrame
        labels: Labels Series
        models: List of model names
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    if models is None:
        models = ['random_forest', 'xgboost']
    
    from src.ml.model_comparison import ModelComparison
    
    comparer = ModelComparison(
        models=models,
        n_folds=n_folds,
        random_state=random_state,
        balance=False,  # Already balanced
        tune_hyperparameters=False,
        use_mlflow=True
    )
    
    comparison_df = comparer.compare_models(features, labels)
    
    return {
        'comparison_df': comparison_df,
        'best_model_name': comparer.best_model_name,
        'best_model': comparer.get_best_model(),
        'best_result': comparer.get_best_result(),
        'all_results': comparer.results
    }


@pipeline(enable_cache=True)
def classification_pipeline(
    features_path: str,
    analysis_type: str = "cell_type",
    model_name: str = "random_forest",
    correlation_threshold: float = 0.8,
    balance_method: str = "undersample",
    n_folds: int = 10,
    tune_hyperparameters: bool = False,
    random_state: int = 42,
    use_enhanced: bool = True
):
    """End-to-end classification pipeline.
    
    Args:
        features_path: Path to features file or directory
        analysis_type: Type of classification
        model_name: Model to use
        correlation_threshold: Feature correlation threshold
        balance_method: Class balancing method
        n_folds: Number of CV folds
        tune_hyperparameters: Whether to tune hyperparameters
        random_state: Random seed
        use_enhanced: Whether to use enhanced features
    """
    # Load data
    features, labels, metadata = load_data_step(
        features_path=features_path,
        analysis_type=analysis_type,
        use_enhanced=use_enhanced
    )
    
    # Preprocess
    features_processed, labels_processed = preprocess_data_step(
        features=features,
        labels=labels,
        correlation_threshold=correlation_threshold
    )
    
    # Balance
    features_balanced, labels_balanced = balance_data_step(
        features=features_processed,
        labels=labels_processed,
        method=balance_method,
        random_state=random_state
    )
    
    # Train
    result = train_model_step(
        features=features_balanced,
        labels=labels_balanced,
        model_name=model_name,
        n_folds=n_folds,
        tune_hyperparameters=tune_hyperparameters,
        random_state=random_state
    )
    
    return result


@pipeline(enable_cache=True)
def model_comparison_pipeline(
    features_path: str,
    analysis_type: str = "cell_type",
    models: List[str] = None,
    correlation_threshold: float = 0.8,
    balance_method: str = "undersample",
    n_folds: int = 10,
    random_state: int = 42,
    use_enhanced: bool = True
):
    """Pipeline to compare multiple models.
    
    Args:
        features_path: Path to features file or directory
        analysis_type: Type of classification
        models: List of models to compare
        correlation_threshold: Feature correlation threshold
        balance_method: Class balancing method
        n_folds: Number of CV folds
        random_state: Random seed
        use_enhanced: Whether to use enhanced features
    """
    if models is None:
        models = ['random_forest', 'xgboost']
    
    # Load data
    features, labels, metadata = load_data_step(
        features_path=features_path,
        analysis_type=analysis_type,
        use_enhanced=use_enhanced
    )
    
    # Preprocess
    features_processed, labels_processed = preprocess_data_step(
        features=features,
        labels=labels,
        correlation_threshold=correlation_threshold
    )
    
    # Balance
    features_balanced, labels_balanced = balance_data_step(
        features=features_processed,
        labels=labels_processed,
        method=balance_method,
        random_state=random_state
    )
    
    # Compare models
    result = compare_models_step(
        features=features_balanced,
        labels=labels_balanced,
        models=models,
        n_folds=n_folds,
        random_state=random_state
    )
    
    return result


# Convenience function to run pipeline without ZenML
def run_classification(
    features_path: str,
    analysis_type: str = "cell_type",
    model_name: str = "random_forest",
    models: Optional[List[str]] = None,
    correlation_threshold: float = 0.8,
    balance_method: str = "undersample",
    n_folds: int = 10,
    tune_hyperparameters: bool = False,
    random_state: int = 42,
    use_enhanced: bool = True,
    use_mlflow: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run classification without ZenML orchestration.
    
    This is a convenience function that runs the classification steps
    directly without ZenML, useful for quick experiments.
    
    Args:
        features_path: Path to features file or directory
        analysis_type: Type of classification
        model_name: Model to use (if models is None)
        models: List of models to compare (overrides model_name)
        correlation_threshold: Feature correlation threshold
        balance_method: Class balancing method
        n_folds: Number of CV folds
        tune_hyperparameters: Whether to tune hyperparameters
        random_state: Random seed
        use_enhanced: Whether to use enhanced features
        use_mlflow: Whether to track with MLflow
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    # Load data
    features, labels, metadata = load_data_step(
        features_path=features_path,
        analysis_type=analysis_type,
        use_enhanced=use_enhanced
    )
    
    # Preprocess
    features_processed, labels_processed = preprocess_data_step(
        features=features,
        labels=labels,
        correlation_threshold=correlation_threshold
    )
    
    # Balance
    features_balanced, labels_balanced = balance_data_step(
        features=features_processed,
        labels=labels_processed,
        method=balance_method,
        random_state=random_state
    )
    
    # Train or compare
    if models and len(models) > 1:
        from src.ml.model_comparison import ModelComparison
        
        comparer = ModelComparison(
            models=models,
            n_folds=n_folds,
            random_state=random_state,
            balance=False,
            tune_hyperparameters=tune_hyperparameters,
            use_mlflow=use_mlflow
        )
        
        comparison_df = comparer.compare_models(
            features_balanced, 
            labels_balanced,
            output_dir=output_dir
        )
        
        return {
            'comparison_df': comparison_df,
            'best_model_name': comparer.best_model_name,
            'best_model': comparer.get_best_model(),
            'best_result': comparer.get_best_result(),
            'all_results': comparer.results
        }
    else:
        result = train_model_step(
            features=features_balanced,
            labels=labels_balanced,
            model_name=model_name or 'random_forest',
            n_folds=n_folds,
            tune_hyperparameters=tune_hyperparameters,
            random_state=random_state
        )
        
        return result
