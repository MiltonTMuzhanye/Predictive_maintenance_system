"""
Model evaluation utilities for predictive maintenance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, 
                             classification_report, confusion_matrix)
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance comprehensively."""
    
    def __init__(self):
        plt.style.use('fivethirtyeight')
        sns.set_palette("Set2")
    
    def generate_report(self, y_true: pd.Series, y_pred: pd.Series, 
                       y_pred_proba: np.ndarray = None,
                       model_name: str = "Model") -> Dict:
        """Generate comprehensive evaluation report."""
        
        report = {
            'model': model_name,
            'metrics': self._calculate_metrics(y_true, y_pred, y_pred_proba),
            'classification_report': classification_report(y_true, y_pred, 
                                                          output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            report['roc_curve'] = self._calculate_roc_curve(y_true, y_pred_proba)
            report['pr_curve'] = self._calculate_pr_curve(y_true, y_pred_proba)
        
        return report
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                          y_pred_proba: np.ndarray = None) -> Dict:
        """Calculate evaluation metrics."""
        from sklearn.metrics import (accuracy_score, precision_score, 
                                     recall_score, f1_score, roc_auc_score)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Calculate ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': float(optimal_threshold)
        }
    
    def _calculate_pr_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Calculate Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find optimal threshold (max F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': float(thresholds[optimal_idx]) if len(thresholds) > optimal_idx else 0.5
        }
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series,
                             model_name: str = "Model"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Failure', 'Failure'],
                   yticklabels=['No Failure', 'Failure'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, results: Dict[str, Dict], save_path: str = None):
        """Plot ROC curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, model_results in results.items():
            if 'roc_curve' in model_results:
                fpr = model_results['roc_curve']['fpr']
                tpr = model_results['roc_curve']['tpr']
                auc = model_results['metrics'].get('roc_auc', 0)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, results: Dict[str, Dict], save_path: str = None):
        """Plot Precision-Recall curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, model_results in results.items():
            if 'pr_curve' in model_results:
                precision = model_results['pr_curve']['precision']
                recall = model_results['pr_curve']['recall']
                plt.plot(recall, precision, label=model_name)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """Compare model performance across metrics."""
        metrics_df = pd.DataFrame({
            model_name: model_results['metrics'] 
            for model_name, model_results in results.items()
        }).T
        
        metrics_df = metrics_df[['precision', 'recall', 'f1', 'roc_auc']]
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', colormap='Set2')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()