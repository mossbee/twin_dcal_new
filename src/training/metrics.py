import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from typing import Tuple, List, Dict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

warnings.filterwarnings('ignore')


class EvaluationMetrics:
    """Evaluation metrics for face verification."""
    
    @staticmethod
    def compute_distance(embeddings1: torch.Tensor, embeddings2: torch.Tensor, metric: str = 'cosine') -> torch.Tensor:
        """
        Compute distance between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [N, D]
            metric: Distance metric ('cosine', 'euclidean')
        
        Returns:
            distances: Distance values [N]
        """
        if metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
            cosine_sim = torch.sum(embeddings1_norm * embeddings2_norm, dim=1)
            distances = 1.0 - cosine_sim
        elif metric == 'euclidean':
            # Euclidean distance
            distances = torch.sqrt(torch.sum((embeddings1 - embeddings2) ** 2, dim=1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distances
    
    @staticmethod
    def compute_eer(distances: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER).
        
        Args:
            distances: Distance values
            labels: Binary labels (1 for same person, 0 for different)
        
        Returns:
            eer: Equal Error Rate
            threshold: Threshold at EER
        """
        # Convert labels: 1 for same person (positive), 0 for different person (negative)
        # For ROC curve, we need similarity scores, so we use negative distances
        scores = -distances
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find EER point where FPR = 1 - TPR (or FNR = FPR)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = -thresholds[eer_idx]  # Convert back to distance
        
        return float(eer), float(eer_threshold)
    
    @staticmethod
    def compute_tar_at_far(distances: np.ndarray, labels: np.ndarray, far_targets: List[float]) -> Dict[str, float]:
        """
        Compute True Accept Rate (TAR) at given False Accept Rate (FAR) values.
        
        Args:
            distances: Distance values
            labels: Binary labels (1 for same person, 0 for different)
            far_targets: Target FAR values
        
        Returns:
            tar_at_far: Dictionary with TAR values at each FAR
        """
        # Convert labels and compute ROC curve
        scores = -distances
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        tar_at_far = {}
        for far_target in far_targets:
            # Find the threshold where FPR is closest to target FAR
            idx = np.argmin(np.abs(fpr - far_target))
            tar = tpr[idx]
            tar_at_far[f'TAR@FAR={far_target}'] = float(tar)
        
        return tar_at_far
    
    @staticmethod
    def compute_auc(distances: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Area Under Curve (AUC) of ROC.
        
        Args:
            distances: Distance values
            labels: Binary labels (1 for same person, 0 for different)
        
        Returns:
            auc_score: AUC value
        """
        scores = -distances
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        return float(auc_score)
    
    @staticmethod
    def compute_accuracy_at_threshold(distances: np.ndarray, labels: np.ndarray, threshold: float) -> float:
        """
        Compute accuracy at a given threshold.
        
        Args:
            distances: Distance values
            labels: Binary labels (1 for same person, 0 for different)
            threshold: Distance threshold
        
        Returns:
            accuracy: Accuracy value
        """
        predictions = (distances < threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        return float(accuracy)
    
    @staticmethod
    def find_best_threshold(distances: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Find the threshold that maximizes accuracy.
        
        Args:
            distances: Distance values
            labels: Binary labels (1 for same person, 0 for different)
        
        Returns:
            best_threshold: Threshold that maximizes accuracy
            best_accuracy: Maximum accuracy
        """
        # Try a range of thresholds
        thresholds = np.linspace(distances.min(), distances.max(), 1000)
        accuracies = []
        
        for threshold in thresholds:
            acc = EvaluationMetrics.compute_accuracy_at_threshold(distances, labels, threshold)
            accuracies.append(acc)
        
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        return float(best_threshold), float(best_accuracy)


class VerificationEvaluator:
    """Evaluator for face verification task."""
    
    def __init__(self, distance_metric: str = 'cosine', far_targets: List[float] = None):
        """
        Args:
            distance_metric: Distance metric to use ('cosine', 'euclidean')
            far_targets: Target FAR values for TAR@FAR computation
        """
        self.distance_metric = distance_metric
        self.far_targets = far_targets or [0.001, 0.01, 0.1]
        self.metrics = EvaluationMetrics()
    
    def evaluate_pairs(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate verification performance on pairs.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [N, D]
            labels: Binary labels (1 for same person, 0 for different) [N]
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        # Compute distances
        distances = self.metrics.compute_distance(embeddings1, embeddings2, self.distance_metric)
        
        # Convert to numpy
        distances_np = distances.cpu().numpy()
        labels_np = labels.cpu().numpy().astype(int)
        
        # Compute metrics
        results = {}
        
        # EER
        eer, eer_threshold = self.metrics.compute_eer(distances_np, labels_np)
        results['EER'] = eer
        results['EER_threshold'] = eer_threshold
        
        # TAR@FAR
        tar_at_far = self.metrics.compute_tar_at_far(distances_np, labels_np, self.far_targets)
        results.update(tar_at_far)
        
        # AUC
        auc_score = self.metrics.compute_auc(distances_np, labels_np)
        results['AUC'] = auc_score
        
        # Best accuracy
        best_threshold, best_accuracy = self.metrics.find_best_threshold(distances_np, labels_np)
        results['best_threshold'] = best_threshold
        results['best_accuracy'] = best_accuracy
        
        # Accuracy at EER threshold
        eer_accuracy = self.metrics.compute_accuracy_at_threshold(distances_np, labels_np, eer_threshold)
        results['accuracy_at_eer'] = eer_accuracy
        
        return results
    
    def evaluate_twin_verification(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate twin verification performance.
        
        Args:
            model: Trained model
            dataloader: Data loader with twin pairs
            device: Device to run evaluation on
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        model.eval()
        
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract image pairs and labels
                # This assumes the dataloader provides pairs of images
                images1 = batch['image1'].to(device)
                images2 = batch['image2'].to(device)
                labels = batch['label'].to(device)  # 1 for same person, 0 for different
                
                # Get embeddings
                embeddings1 = model.forward_features(images1)
                embeddings2 = model.forward_features(images2)
                
                all_embeddings1.append(embeddings1)
                all_embeddings2.append(embeddings2)
                all_labels.append(labels)
        
        # Concatenate all results
        all_embeddings1 = torch.cat(all_embeddings1, dim=0)
        all_embeddings2 = torch.cat(all_embeddings2, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Evaluate
        results = self.evaluate_pairs(all_embeddings1, all_embeddings2, all_labels)
        
        return results


class ExtendedVerificationEvaluator(VerificationEvaluator):
    """Extended evaluator with additional metrics and visualizations."""
    
    def __init__(self, distance_metric='cosine', far_targets=None):
        super().__init__(distance_metric, far_targets)
        self.detailed_results = {}
    
    def evaluate_with_analysis(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor,
        save_plots: bool = True,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation with detailed analysis.
        
        Args:
            embeddings: Feature embeddings [N, feature_dim]
            labels: Person labels [N]
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots
        
        Returns:
            metrics: Comprehensive evaluation metrics
        """
        # Basic metrics
        basic_metrics = self.evaluate_verification(embeddings, labels)
        
        # Get similarity matrix and labels
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        pair_labels = self._create_pair_labels(labels)
        
        # Additional metrics
        extended_metrics = self._compute_extended_metrics(similarity_matrix, pair_labels)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **extended_metrics}
        
        # Store detailed results
        self.detailed_results = {
            'similarity_matrix': similarity_matrix.cpu().numpy(),
            'pair_labels': pair_labels.cpu().numpy(),
            'embeddings': embeddings.cpu().numpy(),
            'labels': labels.cpu().numpy()
        }
        
        # Generate visualizations
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            self.generate_visualizations(output_dir)
        
        return all_metrics
    
    def _compute_extended_metrics(
        self, 
        similarity_matrix: torch.Tensor, 
        pair_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute extended evaluation metrics."""
        
        # Convert to numpy for some calculations
        similarities = similarity_matrix.cpu().numpy().flatten()
        labels = pair_labels.cpu().numpy().flatten()
        
        # Remove diagonal (self-similarities)
        mask = ~np.eye(len(similarity_matrix), dtype=bool)
        similarities = similarity_matrix[mask].cpu().numpy()
        labels = pair_labels[mask].cpu().numpy()
        
        # Precision-Recall metrics
        precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
        avg_precision = average_precision_score(labels, similarities)
        
        # Find best F1 score
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_f1_threshold = pr_thresholds[best_f1_idx]
        
        # Rank-based metrics
        rank_metrics = self._compute_rank_metrics(similarities, labels)
        
        # Distribution analysis
        pos_similarities = similarities[labels == 1]
        neg_similarities = similarities[labels == 0]
        
        separation_metrics = {
            'positive_mean': float(np.mean(pos_similarities)),
            'positive_std': float(np.std(pos_similarities)),
            'negative_mean': float(np.mean(neg_similarities)),
            'negative_std': float(np.std(neg_similarities)),
            'separation_score': float(
                (np.mean(pos_similarities) - np.mean(neg_similarities)) / 
                (np.std(pos_similarities) + np.std(neg_similarities) + 1e-8)
            )
        }
        
        return {
            'average_precision': float(avg_precision),
            'best_f1_score': float(best_f1),
            'best_f1_threshold': float(best_f1_threshold),
            **rank_metrics,
            **separation_metrics
        }
    
    def _compute_rank_metrics(self, similarities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute rank-based metrics (CMC, mAP)."""
        
        # For each positive pair, find its rank among all pairs with the same query
        # This is a simplified version - in practice, you'd need gallery/query splits
        
        # Compute mean reciprocal rank
        positive_indices = np.where(labels == 1)[0]
        reciprocal_ranks = []
        
        for pos_idx in positive_indices:
            pos_sim = similarities[pos_idx]
            # Find rank (how many similarities are higher)
            rank = np.sum(similarities >= pos_sim)
            reciprocal_ranks.append(1.0 / rank)
        
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        
        # Compute Rank-1 accuracy (simplified)
        rank1_correct = 0
        for pos_idx in positive_indices:
            pos_sim = similarities[pos_idx]
            if np.sum(similarities > pos_sim) == 0:  # This is the highest similarity
                rank1_correct += 1
        
        rank1_accuracy = float(rank1_correct / len(positive_indices)) if positive_indices.size > 0 else 0.0
        
        return {
            'mean_reciprocal_rank': mrr,
            'rank1_accuracy': rank1_accuracy
        }
    
    def generate_visualizations(self, output_dir: str):
        """Generate comprehensive visualization plots."""
        
        similarities = self.detailed_results['similarity_matrix']
        pair_labels = self.detailed_results['pair_labels']
        embeddings = self.detailed_results['embeddings']
        labels = self.detailed_results['labels']
        
        # 1. ROC Curve
        self._plot_roc_curve(similarities, pair_labels, output_dir)
        
        # 2. Precision-Recall Curve
        self._plot_pr_curve(similarities, pair_labels, output_dir)
        
        # 3. Similarity Distribution
        self._plot_similarity_distribution(similarities, pair_labels, output_dir)
        
        # 4. Embedding Visualization (t-SNE/UMAP)
        self._plot_embedding_visualization(embeddings, labels, output_dir)
        
        # 5. Confusion Matrix at optimal threshold
        self._plot_confusion_matrix(similarities, pair_labels, output_dir)
        
        # 6. Twin pair analysis
        self._plot_twin_analysis(similarities, pair_labels, output_dir)
    
    def _plot_roc_curve(self, similarities, pair_labels, output_dir):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        # Flatten and remove diagonal
        mask = ~np.eye(len(similarities), dtype=bool)
        sims_flat = similarities[mask]
        labels_flat = pair_labels[mask]
        
        fpr, tpr, _ = roc_curve(labels_flat, sims_flat)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, similarities, pair_labels, output_dir):
        """Plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        mask = ~np.eye(len(similarities), dtype=bool)
        sims_flat = similarities[mask]
        labels_flat = pair_labels[mask]
        
        precision, recall, _ = precision_recall_curve(labels_flat, sims_flat)
        avg_precision = average_precision_score(labels_flat, sims_flat)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pr_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_distribution(self, similarities, pair_labels, output_dir):
        """Plot similarity score distributions."""
        mask = ~np.eye(len(similarities), dtype=bool)
        sims_flat = similarities[mask]
        labels_flat = pair_labels[mask]
        
        pos_sims = sims_flat[labels_flat == 1]
        neg_sims = sims_flat[labels_flat == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(neg_sims, bins=50, alpha=0.7, label='Different Persons', color='red', density=True)
        plt.hist(pos_sims, bins=50, alpha=0.7, label='Same Person', color='green', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Similarity Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/similarity_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_embedding_visualization(self, embeddings, labels, output_dir):
        """Plot embedding visualization using t-SNE."""
        try:
            from sklearn.manifold import TSNE
            
            # Sample subset for visualization if too large
            n_samples = min(1000, len(embeddings))
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = labels[indices]
            
            # Compute t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(sample_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=sample_labels, cmap='tab20', alpha=0.7)
            plt.colorbar(scatter)
            plt.title('t-SNE Visualization of Embeddings')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/embedding_tsne.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("scikit-learn not available, skipping t-SNE visualization")
    
    def _plot_confusion_matrix(self, similarities, pair_labels, output_dir):
        """Plot confusion matrix at optimal threshold."""
        from sklearn.metrics import roc_curve, confusion_matrix
        
        mask = ~np.eye(len(similarities), dtype=bool)
        sims_flat = similarities[mask]
        labels_flat = pair_labels[mask]
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels_flat, sims_flat)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Create predictions
        predictions = (sims_flat >= optimal_threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(labels_flat, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Different', 'Same'],
                   yticklabels=['Different', 'Same'])
        plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_twin_analysis(self, similarities, pair_labels, output_dir):
        """Plot analysis specific to twin face verification."""
        # This would analyze twin vs non-twin performance
        # For now, just plot the similarity matrix heatmap
        
        plt.figure(figsize=(10, 8))
        
        # Sample subset for visualization
        n_show = min(100, len(similarities))
        sample_sims = similarities[:n_show, :n_show]
        
        sns.heatmap(sample_sims, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Similarity'})
        plt.title('Similarity Matrix Heatmap')
        plt.xlabel('Person Index')
        plt.ylabel('Person Index')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/similarity_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()


class PerformanceProfiler:
    """Profile model performance and computational efficiency."""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
    
    def profile_model_inference(
        self, 
        model: torch.nn.Module, 
        input_shape: Tuple[int, ...],
        device: torch.device,
        n_runs: int = 100
    ) -> Dict[str, float]:
        """
        Profile model inference performance.
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
            device: Device to run on
            n_runs: Number of runs for averaging
        
        Returns:
            profile_results: Performance metrics
        """
        model.eval()
        model.to(device)
        
        # Warmup
        dummy_input = torch.randn(input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Time inference
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / n_runs
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        else:
            memory_allocated = memory_reserved = 0
        
        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': 1.0 / avg_inference_time,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2  # Assuming float32
        }
        
        return results
    
    def compare_models(
        self, 
        models: Dict[str, torch.nn.Module],
        input_shape: Tuple[int, ...],
        device: torch.device
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance of multiple models."""
        comparison_results = {}
        
        for name, model in models.items():
            print(f"Profiling {name}...")
            results = self.profile_model_inference(model, input_shape, device)
            comparison_results[name] = results
        
        return comparison_results
