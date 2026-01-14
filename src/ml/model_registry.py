# -*- coding: utf-8 -*-
"""
Model Registry for Germinal Center classification.

Provides a unified interface for multiple classifier implementations
with default hyperparameters and search grids.

Supported models:
- Random Forest (sklearn)
- XGBoost
- LightGBM
- Logistic Regression
- Support Vector Machine
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Abstract base class for all classifiers.
    
    All classifier wrappers must implement:
    - get_model(): Returns initialized model instance
    - get_default_params(): Returns default hyperparameters
    - get_param_grid(): Returns hyperparameter search grid
    """
    
    @abstractmethod
    def get_model(self, **kwargs) -> Any:
        """Return initialized model instance with given parameters.
        
        Args:
            **kwargs: Override default hyperparameters
            
        Returns:
            Initialized model instance
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for the model.
        
        Returns:
            Dictionary of parameter names to values
        """
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Return hyperparameter search grid for tuning.
        
        Returns:
            Dictionary of parameter names to lists of values to try
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def supports_feature_importance(self) -> bool:
        """Whether the model supports feature importance extraction."""
        pass


class RandomForestClassifierWrapper(BaseClassifier):
    """Random Forest classifier wrapper using sklearn."""
    
    @property
    def name(self) -> str:
        return "random_forest"
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_model(self, **kwargs) -> Any:
        """Get Random Forest classifier."""
        from sklearn.ensemble import RandomForestClassifier
        
        defaults = self.get_default_params()
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters optimized for chromatin feature classification."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Hyperparameter grid for RandomizedSearchCV."""
        return {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }


class XGBoostClassifierWrapper(BaseClassifier):
    """XGBoost classifier wrapper."""
    
    @property
    def name(self) -> str:
        return "xgboost"
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_model(self, **kwargs) -> Any:
        """Get XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        
        defaults = self.get_default_params()
        defaults.update(kwargs)
        return xgb.XGBClassifier(**defaults)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for XGBoost."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'verbosity': 0
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Hyperparameter grid for XGBoost."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }


class LightGBMClassifierWrapper(BaseClassifier):
    """LightGBM classifier wrapper."""
    
    @property
    def name(self) -> str:
        return "lightgbm"
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_model(self, **kwargs) -> Any:
        """Get LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        
        defaults = self.get_default_params()
        defaults.update(kwargs)
        return lgb.LGBMClassifier(**defaults)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for LightGBM."""
        return {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Hyperparameter grid for LightGBM."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'num_leaves': [31, 50, 100],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }


class LogisticRegressionWrapper(BaseClassifier):
    """Logistic Regression classifier wrapper."""
    
    @property
    def name(self) -> str:
        return "logistic_regression"
    
    @property
    def supports_feature_importance(self) -> bool:
        return True  # Uses coefficients
    
    def get_model(self, **kwargs) -> Any:
        """Get Logistic Regression classifier."""
        from sklearn.linear_model import LogisticRegression
        
        defaults = self.get_default_params()
        defaults.update(kwargs)
        return LogisticRegression(**defaults)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for Logistic Regression."""
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Hyperparameter grid for Logistic Regression."""
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }


class SVMClassifierWrapper(BaseClassifier):
    """Support Vector Machine classifier wrapper."""
    
    @property
    def name(self) -> str:
        return "svm"
    
    @property
    def supports_feature_importance(self) -> bool:
        return False  # SVM doesn't have built-in feature importance
    
    def get_model(self, **kwargs) -> Any:
        """Get SVM classifier."""
        from sklearn.svm import SVC
        
        defaults = self.get_default_params()
        defaults.update(kwargs)
        return SVC(**defaults)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for SVM."""
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,  # Enable probability estimates
            'random_state': 42
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Hyperparameter grid for SVM."""
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }


# Model Registry - maps model names to wrapper instances
MODEL_REGISTRY: Dict[str, BaseClassifier] = {
    # Random Forest aliases
    'random_forest': RandomForestClassifierWrapper(),
    'rf': RandomForestClassifierWrapper(),
    'randomforest': RandomForestClassifierWrapper(),
    
    # XGBoost aliases
    'xgboost': XGBoostClassifierWrapper(),
    'xgb': XGBoostClassifierWrapper(),
    
    # LightGBM aliases
    'lightgbm': LightGBMClassifierWrapper(),
    'lgbm': LightGBMClassifierWrapper(),
    'lgb': LightGBMClassifierWrapper(),
    
    # Logistic Regression aliases
    'logistic_regression': LogisticRegressionWrapper(),
    'logreg': LogisticRegressionWrapper(),
    'lr': LogisticRegressionWrapper(),
    
    # SVM aliases
    'svm': SVMClassifierWrapper(),
    'svc': SVMClassifierWrapper(),
    'support_vector_machine': SVMClassifierWrapper()
}


def get_model_wrapper(model_name: str) -> BaseClassifier:
    """Get model wrapper from registry by name.
    
    Args:
        model_name: Name or alias of the model
        
    Returns:
        BaseClassifier wrapper instance
        
    Raises:
        ValueError: If model name is not in registry
    """
    model_name = model_name.lower().strip()
    
    if model_name not in MODEL_REGISTRY:
        available = sorted(set(w.name for w in MODEL_REGISTRY.values()))
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    return MODEL_REGISTRY[model_name]


def list_available_models() -> List[str]:
    """List all available model names (without aliases).
    
    Returns:
        List of unique model names
    """
    return sorted(set(wrapper.name for wrapper in MODEL_REGISTRY.values()))


def get_model(model_name: str, **kwargs) -> Any:
    """Convenience function to get initialized model directly.
    
    Args:
        model_name: Name or alias of the model
        **kwargs: Override default hyperparameters
        
    Returns:
        Initialized model instance
    """
    wrapper = get_model_wrapper(model_name)
    return wrapper.get_model(**kwargs)
