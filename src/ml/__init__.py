# -*- coding: utf-8 -*-
"""
Machine Learning module for Germinal Center analysis.

Provides:
- Model registry with multiple classifier implementations
- Model comparison framework
- MLflow experiment tracking (optional)
- ZenML pipeline integration (optional)

Usage:
    # Compare models
    from src.ml import ModelComparison
    comparer = ModelComparison(models=['random_forest', 'xgboost'])
    results = comparer.compare_models(X, y)
    
    # Get specific model
    from src.ml import get_model_wrapper
    rf = get_model_wrapper('random_forest').get_model()
    
    # MLflow tracking (if installed)
    from src.ml import MLflowExperimentTracker
    tracker = MLflowExperimentTracker(experiment_name='my_exp')
"""

from src.ml.model_registry import (
    get_model_wrapper,
    get_model,
    list_available_models,
    MODEL_REGISTRY,
    BaseClassifier,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
    LightGBMClassifierWrapper,
    LogisticRegressionWrapper,
    SVMClassifierWrapper
)
from src.ml.model_comparison import ModelComparison
from src.ml.mlflow_tracker import MLflowExperimentTracker, get_mlflow_tracker

__all__ = [
    # Model registry
    'get_model_wrapper',
    'get_model',
    'list_available_models',
    'MODEL_REGISTRY',
    'BaseClassifier',
    'RandomForestClassifierWrapper',
    'XGBoostClassifierWrapper',
    'LightGBMClassifierWrapper',
    'LogisticRegressionWrapper',
    'SVMClassifierWrapper',
    # Model comparison
    'ModelComparison',
    # MLflow
    'MLflowExperimentTracker',
    'get_mlflow_tracker'
]
