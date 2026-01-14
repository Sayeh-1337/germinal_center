# -*- coding: utf-8 -*-
"""
ZenML pipeline definitions for Germinal Center analysis.
"""

# Import pipelines only if ZenML is available
try:
    from src.ml.pipelines.classification_pipeline import (
        classification_pipeline,
        model_comparison_pipeline,
        load_data_step,
        preprocess_data_step,
        train_model_step,
        compare_models_step
    )
    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    
    # Provide dummy implementations for when ZenML is not installed
    def classification_pipeline(*args, **kwargs):
        raise ImportError("ZenML is not installed. Install with: pip install zenml[mlflow]")
    
    def model_comparison_pipeline(*args, **kwargs):
        raise ImportError("ZenML is not installed. Install with: pip install zenml[mlflow]")

__all__ = [
    'classification_pipeline',
    'model_comparison_pipeline',
    'ZENML_AVAILABLE'
]
