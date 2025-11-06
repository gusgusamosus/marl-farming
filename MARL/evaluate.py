# evaluate.py
import os
import argparse
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from cnn_encoder import CNNEncoder
from policy import ActorCritic


class ModelEvaluator:
    def __init__(self, checkpoint_path, dataset_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load encoder
        print("Loading CNN Encoder...")
        self.encoder = CNNEncoder().to(self.device)
        self.encoder.eval()
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.class_names = self.dataset.classes
        
        # Load policy
        print(f"Loading policy from {checkpoint_path}...")
        self.policy = ActorCritic(obs_dim=512, action_dim=self.num_classes).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        
        print(f"✓ Model loaded successfully (Episode {checkpoint['episode']})")
        print(f"✓ Dataset: {len(self.dataset)} images, {self.num_classes} classes")
    
    def predict(self, idx):
        """Predict single image"""
        img, true_label = self.dataset[idx]
        
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(self.device)
            embedding = self.encoder(img_tensor).squeeze(0)
            action, log_prob = self.policy.act(embedding)
            confidence = torch.exp(log_prob).item()
        
        return action, confidence, true_label
    
    def evaluate_all(self, sample_size=None):
        """Evaluate on entire dataset or sample"""
        print("\n" + "="*60)
        print("Running Evaluation...")
        print("="*60)
        
        dataset_size = len(self.dataset)
        if sample_size and sample_size < dataset_size:
            indices = np.random.choice(dataset_size, sample_size, replace=False)
            print(f"Evaluating on random sample of {sample_size} images")
        else:
            indices = range(dataset_size)
            print(f"Evaluating on all {dataset_size} images")
        
        predictions = []
        confidences = []
        true_labels = []
        
        for idx in tqdm(indices, desc="Evaluating"):
            pred, conf, true = self.predict(idx)
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(true)
        
        return np.array(predictions), np.array(confidences), np.array(true_labels)
    
    def compute_metrics(self, predictions, true_labels):
        """Compute classification metrics"""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
        
        return metrics
    
    def plot_confusion_matrix(self, predictions, true_labels, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[name.replace('Tomato_', '') for name in self.class_names],
                    yticklabels=[name.replace('Tomato_', '') for name in self.class_names])
        plt.title('Confusion Matrix - Disease Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
        
        return cm
    
    def plot_per_class_metrics(self, metrics, save_path='per_class_metrics.png'):
        """Plot per-class performance metrics"""
        classes = list(metrics['per_class'].keys())
        precisions = [metrics['per_class'][c]['precision'] for c in classes]
        recalls = [metrics['per_class'][c]['recall'] for c in classes]
        f1_scores = [metrics['per_class'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
        ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Disease Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('Tomato_', '') for name in classes], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics plot saved to {save_path}")
        plt.close()
    
    def plot_confidence_distribution(self, confidences, predictions, true_labels, save_path='confidence_dist.png'):
        """Plot confidence score distribution for correct/incorrect predictions"""
        correct_mask = predictions == true_labels
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
        ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2.boxplot([correct_confidences, incorrect_confidences],
                    labels=['Correct', 'Incorrect'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence Comparison', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence distribution plot saved to {save_path}")
        plt.close()
    
    def save_results(self, metrics, predictions, true_labels, confidences, output_dir='evaluation_results'):
        """Save all evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Metrics saved to {metrics_path}")
        
        # Save detailed predictions as CSV
        csv_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'True Label', 'Predicted Label', 'Confidence', 'Correct'])
            for i, (true, pred, conf) in enumerate(zip(true_labels, predictions, confidences)):
                writer.writerow([
                    i,
                    self.class_names[true],
                    self.class_names[pred],
                    f"{conf:.4f}",
                    true == pred
                ])
        print(f"✓ Detailed predictions saved to {csv_path}")
        
        # Save classification report
        report_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DISEASE CLASSIFICATION - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(predictions)}\n")
            f.write(f"Number of Classes: {self.num_classes}\n\n")
            f.write("="*60 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy:  {metrics['overall']['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['overall']['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['overall']['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['overall']['f1_score']:.4f}\n\n")
            f.write("="*60 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(classification_report(true_labels, predictions, 
                                         target_names=self.class_names,
                                         digits=4))
        print(f"✓ Classification report saved to {report_path}")
        
        return output_dir
    
    def print_summary(self, metrics):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"\n{'Metric':<20} {'Score':<10}")
        print("-"*30)
        print(f"{'Accuracy':<20} {metrics['overall']['accuracy']:.4f}")
        print(f"{'Precision':<20} {metrics['overall']['precision']:.4f}")
        print(f"{'Recall':<20} {metrics['overall']['recall']:.4f}")
        print(f"{'F1-Score':<20} {metrics['overall']['f1_score']:.4f}")
        print("\n" + "="*60)
        print("TOP 3 PERFORMING CLASSES")
        print("="*60)
        
        # Sort by F1 score
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        for i, (class_name, class_metrics) in enumerate(sorted_classes[:3], 1):
            print(f"\n{i}. {class_name.replace('Tomato_', '')}")
            print(f"   F1-Score: {class_metrics['f1_score']:.4f}")
            print(f"   Precision: {class_metrics['precision']:.4f}")
            print(f"   Recall: {class_metrics['recall']:.4f}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO disease classification model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_10000.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='TomatoDataset',
                       help='Path to evaluation dataset')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.dataset, args.device)
    
    # Run evaluation
    predictions, confidences, true_labels = evaluator.evaluate_all(args.sample_size)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, true_labels)
    
    # Generate visualizations
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.plot_confusion_matrix(predictions, true_labels,
                                    os.path.join(output_dir, 'confusion_matrix.png'))
    evaluator.plot_per_class_metrics(metrics,
                                     os.path.join(output_dir, 'per_class_metrics.png'))
    evaluator.plot_confidence_distribution(confidences, predictions, true_labels,
                                          os.path.join(output_dir, 'confidence_distribution.png'))
    
    # Save results
    evaluator.save_results(metrics, predictions, true_labels, confidences, output_dir)
    
    # Print summary
    evaluator.print_summary(metrics)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
