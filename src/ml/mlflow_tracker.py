# -*- coding: utf-8 -*-
"""
MLflow Experiment Tracker for Germinal Center analysis.

Provides a wrapper around MLflow for:
- Experiment tracking
- Model logging and registry
- Artifact management
- Run comparison
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
import tempfile
import json

logger = logging.getLogger(__name__)


class MLflowExperimentTracker:
    """Wrapper for MLflow experiment tracking.
    
    Simplifies MLflow integration for the germinal center analysis pipeline.
    
    Example:
        >>> tracker = MLflowExperimentTracker(
        ...     experiment_name="dz_lz_classification",
        ...     tracking_uri="file:./mlruns"
        ... )
        >>> with tracker.start_run(run_name="random_forest_v1"):
        ...     tracker.log_params({"n_estimators": 100})
        ...     tracker.log_metrics({"accuracy": 0.85})
        ...     tracker.log_model(model, "classifier")
    """
    
    def __init__(
        self,
        experiment_name: str = "germinal_center_analysis",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: file:./mlruns)
            artifact_location: Location for artifacts (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or f"file:{Path.cwd()}/mlruns"
        self.artifact_location = artifact_location
        
        self._mlflow = None
        self._active_run = None
        self._initialized = False
        
        # Try to initialize MLflow
        self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize MLflow connection."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
            
            mlflow.set_experiment(self.experiment_name)
            
            self._initialized = True
            logger.info(f"MLflow initialized: {self.experiment_name}")
            logger.info(f"Tracking URI: {self.tracking_uri}")
            
        except ImportError:
            logger.warning(
                "MLflow is not installed. Install with: pip install mlflow"
            )
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if MLflow is available and initialized."""
        return self._initialized
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            nested: Whether this is a nested run
            
        Returns:
            MLflow run context (or dummy context if not available)
        """
        if not self._initialized:
            return _DummyContext()
        
        self._active_run = self._mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        return self._active_run
    
    def end_run(self):
        """End the current MLflow run."""
        if self._initialized and self._active_run:
            self._mlflow.end_run()
            self._active_run = None
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if self._initialized:
            self._mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters.
        
        Args:
            params: Dictionary of parameter names to values
        """
        if self._initialized:
            # Handle numpy types
            clean_params = {}
            for k, v in params.items():
                try:
                    import numpy as np
                    if isinstance(v, np.integer):
                        v = int(v)
                    elif isinstance(v, np.floating):
                        v = float(v)
                except ImportError:
                    pass
                clean_params[k] = v
            
            self._mlflow.log_params(clean_params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step number (for tracking over time)
        """
        if self._initialized:
            self._mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step number
        """
        if self._initialized:
            self._mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """Log a model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within run artifacts
            registered_model_name: Name for model registry (optional)
            **kwargs: Additional arguments for mlflow.sklearn.log_model
        """
        if not self._initialized:
            return
        
        try:
            import mlflow.sklearn
            
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
            
            logger.debug(f"Logged model to {artifact_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact.
        
        Args:
            local_path: Path to local file
            artifact_path: Destination path in artifacts
        """
        if self._initialized:
            self._mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory of files as artifacts.
        
        Args:
            local_dir: Path to local directory
            artifact_path: Destination path in artifacts
        """
        if self._initialized:
            self._mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact (e.g., "config.json")
        """
        if not self._initialized:
            return
        
        # Write to temp file and log
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        ) as f:
            json.dump(dictionary, f, indent=2, default=str)
            temp_path = f.name
        
        self._mlflow.log_artifact(temp_path, artifact_file.rsplit('/', 1)[0] if '/' in artifact_file else None)
        Path(temp_path).unlink()
    
    def log_dataframe(
        self,
        df,
        artifact_path: str,
        filename: str = "data.csv"
    ):
        """Log a pandas DataFrame as a CSV artifact.
        
        Args:
            df: DataFrame to log
            artifact_path: Destination path in artifacts
            filename: Filename for the CSV
        """
        if not self._initialized:
            return
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False
        ) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        self._mlflow.log_artifact(temp_path, artifact_path)
        Path(temp_path).unlink()
    
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the artifact
        """
        if not self._initialized:
            return
        
        with tempfile.NamedTemporaryFile(
            suffix='.png', 
            delete=False
        ) as f:
            figure.savefig(f.name, dpi=150, bbox_inches='tight')
            temp_path = f.name
        
        self._mlflow.log_artifact(temp_path)
        Path(temp_path).unlink()
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run.
        
        Args:
            key: Tag name
            value: Tag value
        """
        if self._initialized:
            self._mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on the current run.
        
        Args:
            tags: Dictionary of tag names to values
        """
        if self._initialized:
            self._mlflow.set_tags(tags)
    
    def get_run(self, run_id: str):
        """Get a run by ID.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            MLflow Run object or None
        """
        if not self._initialized:
            return None
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        return client.get_run(run_id)
    
    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List:
        """Search for runs in the experiment.
        
        Args:
            filter_string: Filter expression (e.g., "metrics.accuracy > 0.8")
            order_by: List of columns to order by
            max_results: Maximum number of results
            
        Returns:
            List of MLflow Run objects
        """
        if not self._initialized:
            return []
        
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            return []
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["metrics.balanced_accuracy_mean DESC"],
            max_results=max_results
        )
        
        return runs
    
    def get_best_run(
        self,
        metric: str = "balanced_accuracy_mean",
        ascending: bool = False
    ):
        """Get the best run based on a metric.
        
        Args:
            metric: Metric name to sort by
            ascending: If True, lower is better
            
        Returns:
            Best MLflow Run object or None
        """
        order = "ASC" if ascending else "DESC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        return runs[0] if runs else None
    
    def load_model(self, run_id: str, artifact_path: str = "model"):
        """Load a model from a run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact
            
        Returns:
            Loaded model
        """
        if not self._initialized:
            raise RuntimeError("MLflow not initialized")
        
        import mlflow.sklearn
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(model_uri)
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ):
        """Register a model from a run to the model registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact
            
        Returns:
            ModelVersion object
        """
        if not self._initialized:
            raise RuntimeError("MLflow not initialized")
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return self._mlflow.register_model(model_uri, model_name)


class _DummyContext:
    """Dummy context manager for when MLflow is not available."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def get_mlflow_tracker(
    experiment_name: str = "germinal_center_analysis",
    tracking_uri: Optional[str] = None
) -> MLflowExperimentTracker:
    """Get or create an MLflow tracker instance.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        
    Returns:
        MLflowExperimentTracker instance
    """
    return MLflowExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
