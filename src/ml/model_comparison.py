# -*- coding: utf-8 -*-
"""
Model Comparison Framework for Germinal Center classification.

Provides production-grade model comparison with:
- Cross-validation evaluation
- Multiple metric computation
- Feature importance extraction
- Optional hyperparameter tuning
- MLflow integration (optional)
- Model persistence
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class ModelComparison:
    """Production-grade model comparison framework.
    
    Compares multiple models using cross-validation and comprehensive metrics.
    Supports optional MLflow tracking and hyperparameter tuning.
    
    Example:
        >>> comparer = ModelComparison(
        ...     models=['random_forest', 'xgboost'],
        ...     n_folds=10,
        ...     tune_hyperparameters=False
        ... )
        >>> results_df = comparer.compare_models(X, y, output_dir='results/')
        >>> print(f"Best model: {comparer.best_model_name}")
    """
    
    def __init__(
        self,
        models: List[str] = None,
        n_folds: int = 10,
        random_state: int = 42,
        balance: bool = True,
        tune_hyperparameters: bool = False,
        n_iter: int = 50,
        use_mlflow: bool = False,
        mlflow_experiment_name: str = "model_comparison"
    ):
        """Initialize model comparison.
        
        Args:
            models: List of model names to compare (default: ['random_forest', 'xgboost'])
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            balance: Whether to balance classes using undersampling
            tune_hyperparameters: Whether to perform hyperparameter tuning
            n_iter: Number of iterations for random search (if tuning)
            use_mlflow: Whether to log to MLflow
            mlflow_experiment_name: Name of MLflow experiment
        """
        if models is None:
            models = ['random_forest', 'xgboost']
        
        self.models = models
        self.n_folds = n_folds
        self.random_state = random_state
        self.balance = balance
        self.tune_hyperparameters = tune_hyperparameters
        self.n_iter = n_iter
        self.use_mlflow = use_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        
        # Results storage
        self.results: Dict[str, Dict] = {}
        self.fitted_models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.comparison_df: Optional[pd.DataFrame] = None
        
        # Feature names (set during comparison)
        self.feature_names: Optional[List[str]] = None
        
        # Label encoder (for string to numeric conversion)
        self.label_encoder: Optional[Any] = None
        self.label_classes: Optional[np.ndarray] = None
        
    def compare_models(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Labels (Series or array)
            output_dir: Directory to save results (optional)
            
        Returns:
            DataFrame with comparison results sorted by best metric
        """
        from src.ml.model_registry import get_model_wrapper
        
        logger.info(f"Comparing {len(self.models)} models: {', '.join(self.models)}")
        logger.info(f"Dataset: {len(X)} samples, {len(X.columns)} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Encode string labels to numeric (required for XGBoost and some other models)
        from sklearn.preprocessing import LabelEncoder
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Check if labels are strings/objects and need encoding
        if len(y_array) > 0 and (y_array.dtype == object or (len(y_array) > 0 and isinstance(y_array[0], str))):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_array)
            self.label_classes = self.label_encoder.classes_
            logger.info(f"Encoded labels: {dict(zip(self.label_classes, range(len(self.label_classes))))}")
        else:
            self.label_encoder = None
            self.label_classes = np.unique(y_array)
            y_encoded = y_array
        
        # Convert back to same type as input
        if hasattr(y, 'values'):
            y = pd.Series(y_encoded, index=y.index if hasattr(y, 'index') else None)
        else:
            y = y_encoded
        
        # Balance dataset if requested
        if self.balance:
            X, y = self._balance_dataset(X, y)
        
        # Initialize MLflow if requested
        if self.use_mlflow:
            self._init_mlflow()
        
        # Compare each model
        comparison_results = []
        
        for model_name in self.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                result = self._evaluate_model(model_name, X, y, output_dir)
                result['model_name'] = model_name
                comparison_results.append(result)
                self.results[model_name] = result
                
            except ImportError as e:
                logger.warning(f"Skipping {model_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if not comparison_results:
            raise RuntimeError("All models failed evaluation")
        
        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame(comparison_results)
        
        # Sort by balanced accuracy
        self.comparison_df = self.comparison_df.sort_values(
            'cv_balanced_accuracy_mean', 
            ascending=False
        ).reset_index(drop=True)
        
        # Set best model
        self.best_model_name = self.comparison_df.iloc[0]['model_name']
        
        # Log summary
        self._log_summary()
        
        # Save results
        if output_dir:
            self._save_results(output_dir)
        
        return self.comparison_df
    
    def _evaluate_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single model with cross-validation."""
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import (
            balanced_accuracy_score, confusion_matrix, classification_report,
            roc_auc_score, f1_score, precision_score, recall_score
        )
        from src.ml.model_registry import get_model_wrapper
        
        # Get model wrapper
        wrapper = get_model_wrapper(model_name)
        
        # Hyperparameter tuning if requested
        if self.tune_hyperparameters:
            best_params = self._tune_hyperparameters(wrapper, X, y)
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            best_params = wrapper.get_default_params()
        
        # Cross-validation setup
        cv = StratifiedKFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Get model
        model = wrapper.get_model(**best_params)
        
        # Get cross-validated predictions
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        y_pred = cross_val_predict(model, X, y_array, cv=cv)
        
        # Get probabilities (if supported)
        try:
            y_proba = cross_val_predict(model, X, y_array, cv=cv, method='predict_proba')
        except:
            y_proba = None
        
        # Compute per-fold metrics
        fold_metrics = self._compute_fold_metrics(
            wrapper, best_params, X, y_array, cv
        )
        
        # Aggregate metrics
        metrics_df = pd.DataFrame(fold_metrics)
        
        # Fit final model for feature importance
        final_model = wrapper.get_model(**best_params)
        final_model.fit(X, y_array)
        self.fitted_models[model_name] = final_model
        
        # Get feature importance
        feature_importance = self._get_feature_importance(
            final_model, 
            X.columns,
            wrapper.supports_feature_importance
        )
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            self._log_to_mlflow(
                model_name, best_params, metrics_df, 
                feature_importance, final_model, output_dir
            )
        
        # Compile results
        result = {
            'cv_balanced_accuracy_mean': metrics_df['balanced_accuracy'].mean(),
            'cv_balanced_accuracy_std': metrics_df['balanced_accuracy'].std(),
            'cv_f1_mean': metrics_df['f1'].mean(),
            'cv_f1_std': metrics_df['f1'].std(),
            'cv_precision_mean': metrics_df['precision'].mean(),
            'cv_precision_std': metrics_df['precision'].std(),
            'cv_recall_mean': metrics_df['recall'].mean(),
            'cv_recall_std': metrics_df['recall'].std(),
            'cv_roc_auc_mean': metrics_df['roc_auc'].mean() if 'roc_auc' in metrics_df else np.nan,
            'cv_roc_auc_std': metrics_df['roc_auc'].std() if 'roc_auc' in metrics_df else np.nan,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'best_params': best_params,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_array,
            'classes': self.label_classes if self.label_encoder is not None else (
                final_model.classes_ if hasattr(final_model, 'classes_') else np.unique(y_array)
            ),
            'fold_metrics': fold_metrics
        }
        
        logger.info(
            f"  Balanced accuracy: {result['cv_balanced_accuracy_mean']:.3f} "
            f"± {result['cv_balanced_accuracy_std']:.3f}"
        )
        
        return result
    
    def _compute_fold_metrics(
        self,
        wrapper,
        params: Dict,
        X: pd.DataFrame,
        y: np.ndarray,
        cv
    ) -> List[Dict]:
        """Compute metrics for each fold."""
        from sklearn.metrics import (
            balanced_accuracy_score, f1_score, precision_score, 
            recall_score, roc_auc_score
        )
        
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Get fold data
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Train model
            model = wrapper.get_model(**params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            metrics = {
                'fold': fold_idx,
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # ROC-AUC
            try:
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_test, y_proba, multi_class='ovr', average='weighted'
                    )
            except:
                metrics['roc_auc'] = np.nan
            
            fold_metrics.append(metrics)
        
        return fold_metrics
    
    def _tune_hyperparameters(
        self,
        wrapper,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune hyperparameters using RandomizedSearchCV."""
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        
        logger.info("Tuning hyperparameters...")
        
        model = wrapper.get_model()
        param_grid = wrapper.get_param_grid()
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=self.n_iter,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        
        logger.info(f"Best CV score: {search.best_score_:.3f}")
        
        return search.best_params_
    
    def _balance_dataset(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Balance dataset using RandomUnderSampler."""
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(random_state=self.random_state)
            X_bal, y_bal = rus.fit_resample(X, y)
            
            logger.info(f"Balanced dataset: {len(X)} -> {len(X_bal)} samples")
            
            return pd.DataFrame(X_bal, columns=X.columns), np.array(y_bal)
            
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed. Using unbalanced data. "
                "Install with: pip install imbalanced-learn"
            )
            y_array = y.values if hasattr(y, 'values') else np.array(y)
            return X, y_array
    
    def _get_feature_importance(
        self,
        model: Any,
        feature_names: pd.Index,
        supports_importance: bool
    ) -> pd.DataFrame:
        """Extract feature importance from model."""
        if not supports_importance:
            return pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models - use absolute coefficients
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info(f"MLflow experiment: {self.mlflow_experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed. Disabling tracking.")
            self.use_mlflow = False
    
    def _log_to_mlflow(
        self,
        model_name: str,
        params: Dict,
        metrics_df: pd.DataFrame,
        feature_importance: pd.DataFrame,
        model: Any,
        output_dir: Optional[str]
    ):
        """Log results to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn
            
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_params(params)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_folds", self.n_folds)
                
                # Log metrics
                mlflow.log_metric("balanced_accuracy_mean", metrics_df['balanced_accuracy'].mean())
                mlflow.log_metric("balanced_accuracy_std", metrics_df['balanced_accuracy'].std())
                mlflow.log_metric("f1_mean", metrics_df['f1'].mean())
                mlflow.log_metric("precision_mean", metrics_df['precision'].mean())
                mlflow.log_metric("recall_mean", metrics_df['recall'].mean())
                
                if 'roc_auc' in metrics_df:
                    mlflow.log_metric("roc_auc_mean", metrics_df['roc_auc'].mean())
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log feature importance
                if not feature_importance.empty:
                    top_features = feature_importance.head(20).to_dict('records')
                    mlflow.log_dict({"top_features": top_features}, "feature_importance.json")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def _log_summary(self):
        """Log comparison summary."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        for _, row in self.comparison_df.iterrows():
            logger.info(
                f"{row['model_name']:20s} | "
                f"Bal.Acc: {row['cv_balanced_accuracy_mean']:.3f} ± {row['cv_balanced_accuracy_std']:.3f} | "
                f"F1: {row['cv_f1_mean']:.3f} | "
                f"ROC-AUC: {row['cv_roc_auc_mean']:.3f}"
            )
        
        logger.info(f"\nBest model: {self.best_model_name}")
    
    def _save_results(self, output_dir: str):
        """Save all results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison summary
        self.comparison_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Save per-model results
        for model_name, result in self.results.items():
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Feature importance
            if result.get('feature_importance') is not None and not result['feature_importance'].empty:
                result['feature_importance'].to_csv(
                    model_dir / "feature_importance.csv", 
                    index=False
                )
            
            # Best parameters
            import json
            with open(model_dir / "best_params.json", 'w') as f:
                # Convert numpy types for JSON serialization
                params = {k: int(v) if isinstance(v, np.integer) else 
                         float(v) if isinstance(v, np.floating) else v 
                         for k, v in result['best_params'].items()}
                json.dump(params, f, indent=2)
            
            # Save model
            model_path = model_dir / "model.joblib"
            joblib.dump(self.fitted_models[model_name], model_path)
        
        logger.info(f"Results saved to: {output_path}")
    
    def get_best_model(self) -> Any:
        """Get the best fitted model.
        
        Returns:
            Best fitted model instance
        """
        if self.best_model_name is None:
            raise RuntimeError("No models have been compared yet")
        
        return self.fitted_models[self.best_model_name]
    
    def get_best_result(self) -> Dict:
        """Get results for the best model.
        
        Returns:
            Dictionary with best model results
        """
        if self.best_model_name is None:
            raise RuntimeError("No models have been compared yet")
        
        return self.results[self.best_model_name]
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using a fitted model.
        
        Args:
            X: Features to predict
            model_name: Model to use (default: best model)
            
        Returns:
            Predictions array
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not found in fitted models")
        
        return self.fitted_models[model_name].predict(X)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities using a fitted model.
        
        Args:
            X: Features to predict
            model_name: Model to use (default: best model)
            
        Returns:
            Probability array
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not found in fitted models")
        
        return self.fitted_models[model_name].predict_proba(X)
