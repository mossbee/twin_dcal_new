import os
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """Base class for experiment tracking."""
    
    @abstractmethod
    def init_experiment(self, project_name: str, experiment_name: str, config: Dict):
        """Initialize experiment."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        pass
    
    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish experiment."""
        pass


class MLFlowTracker(BaseTracker):
    """MLFlow tracker implementation."""
    
    def __init__(self, tracking_uri: str = None):
        """
        Args:
            tracking_uri: MLFlow tracking URI
        """
        try:
            import mlflow
            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            raise ImportError("MLFlow not installed. Install with: pip install mlflow")
        
        self.experiment_id = None
        self.run = None
    
    def init_experiment(self, project_name: str, experiment_name: str, config: Dict):
        """Initialize MLFlow experiment."""
        # Set or create experiment
        try:
            experiment = self.mlflow.get_experiment_by_name(project_name)
            if experiment is None:
                experiment_id = self.mlflow.create_experiment(project_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = self.mlflow.create_experiment(project_name)
        
        # Start run
        self.run = self.mlflow.start_run(
            experiment_id=experiment_id,
            run_name=experiment_name
        )
        
        # Log config
        self.log_params(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow."""
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            self.mlflow.log_param(key, value)
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact to MLFlow."""
        self.mlflow.log_artifact(artifact_path, artifact_name)
    
    def finish(self):
        """End MLFlow run."""
        if self.run:
            self.mlflow.end_run()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


class WandBTracker(BaseTracker):
    """Weights & Biases tracker implementation."""
    
    def __init__(self, entity: str = None):
        """
        Args:
            entity: WandB entity name
        """
        try:
            import wandb
            self.wandb = wandb
            self.entity = entity
        except ImportError:
            raise ImportError("WandB not installed. Install with: pip install wandb")
        
        self.run = None
    
    def init_experiment(self, project_name: str, experiment_name: str, config: Dict):
        """Initialize WandB experiment."""
        self.run = self.wandb.init(
            project=project_name,
            name=experiment_name,
            entity=self.entity,
            config=config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB."""
        self.wandb.log(metrics, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to WandB (done in init)."""
        if self.run:
            self.wandb.config.update(params)
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact to WandB."""
        artifact = self.wandb.Artifact(
            name=artifact_name or os.path.basename(artifact_path),
            type='model'
        )
        artifact.add_file(artifact_path)
        self.wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish WandB run."""
        if self.run:
            self.wandb.finish()


class NoTracker(BaseTracker):
    """No-op tracker for local development."""
    
    def __init__(self):
        self.metrics = []
        self.params = {}
    
    def init_experiment(self, project_name: str, experiment_name: str, config: Dict):
        """Initialize no-op experiment."""
        print(f"Starting experiment: {project_name}/{experiment_name}")
        self.params.update(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics locally."""
        log_entry = {'step': step, **metrics}
        self.metrics.append(log_entry)
        print(f"Step {step}: {metrics}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters locally."""
        self.params.update(params)
        print(f"Parameters: {params}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact locally."""
        print(f"Artifact saved: {artifact_path}")
    
    def finish(self):
        """Finish local tracking."""
        print("Experiment finished")
        # Optionally save metrics to file
        if self.metrics:
            with open('metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)


class ExperimentTracker:
    """Main experiment tracker that wraps different tracking backends."""
    
    def __init__(self, method: str = 'none', **kwargs):
        """
        Initialize tracker.
        
        Args:
            method: Tracking method ('mlflow', 'wandb', 'none')
            **kwargs: Additional arguments for specific trackers
        """
        self.method = method
        
        if method == 'mlflow':
            self.tracker = MLFlowTracker(tracking_uri=kwargs.get('mlflow_uri'))
        elif method == 'wandb':
            self.tracker = WandBTracker(entity=kwargs.get('entity'))
        elif method == 'none':
            self.tracker = NoTracker()
        else:
            raise ValueError(f"Unknown tracking method: {method}")
    
    def init_experiment(self, project_name: str, experiment_name: str, config: Dict):
        """Initialize experiment."""
        return self.tracker.init_experiment(project_name, experiment_name, config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        return self.tracker.log_metrics(metrics, step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        return self.tracker.log_params(params)
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact."""
        return self.tracker.log_artifact(artifact_path, artifact_name)
    
    def finish(self):
        """Finish experiment."""
        return self.tracker.finish()


def create_tracker(config: Dict) -> ExperimentTracker:
    """Create tracker from config."""
    tracking_config = config.get('tracking', {})
    method = tracking_config.get('method', 'none')
    
    kwargs = {}
    if method == 'mlflow':
        kwargs['mlflow_uri'] = tracking_config.get('mlflow_uri', 'http://localhost:5000')
    elif method == 'wandb':
        kwargs['entity'] = tracking_config.get('entity')
        # Set API key from environment
        if 'WANDB_API_KEY' in os.environ:
            os.environ['WANDB_API_KEY'] = os.environ['WANDB_API_KEY']
    
    return ExperimentTracker(method=method, **kwargs)
