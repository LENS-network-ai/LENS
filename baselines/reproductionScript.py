"""
Main script to reproduce baseline comparison results.
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


from dataset import GraphDataset

# Import baseline models
from baselines.GTP import GTP
from baselines.gcn import GCN
from baselines.graphLsurv import GraphLSurv
from baselines.patchGCN import PatchGCN, DeepGraphConv
from baselines.FLOPcalcul import ComputationalProfiler


def collate_baseline(batch):
    """Collate function for baseline models."""
    return {
        'image': batch[0]['image'],
        'adj_s': batch[0]['adj_s'], 
        'label': batch[0]['label'],
        'id': batch[0]['id']
    }


class BaselineTrainer:
    """Trainer class for baseline models."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
    def train_epoch(self, dataloader, epoch=0):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, sample in enumerate(dataloader):
            # Move sample to device
            for key in ['image', 'adj_s']:
                if key in sample:
                    sample[key] = sample[key].to(self.device)
            labels = torch.tensor([sample['label']], dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(sample)
            loss = self.criterion(logits.unsqueeze(0), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == labels[0]).sum().item()
            total += 1
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for sample in dataloader:
                # Move sample to device
                for key in ['image', 'adj_s']:
                    if key in sample:
                        sample[key] = sample[key].to(self.device)
                labels = torch.tensor([sample['label']], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(sample)
                loss = self.criterion(logits.unsqueeze(0), labels)
                
                # Track metrics
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == labels[0]).sum().item()
                total += 1
                
                # For AUROC calculation
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels[0].cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Calculate AUROC for 3-class classification
        if len(np.unique(all_labels)) > 2:
            try:
                all_probs = np.vstack(all_probs)
                auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            except ValueError:
                auroc = 0.0
        else:
            auroc = 0.0
            
        return accuracy, avg_loss, auroc


def train_baseline_model(model_name, model, train_loader, val_loader, epochs=100, 
                        device='cuda', output_dir='./results'):
    """Train a baseline model."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"{'='*60}")
    
    trainer = BaselineTrainer(model, device=device, learning_rate=0.001, weight_decay=1e-4)
    
    best_val_acc = 0.0
    best_val_auroc = 0.0
    best_epoch = 0
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_acc, train_loss = trainer.train_epoch(train_loader, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        # Validation
        val_acc, val_loss, val_auroc = trainer.evaluate(val_loader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"   Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f'best_{model_name.lower()}_model.pt')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
                'val_auroc': best_val_auroc,
            }, model_path)
            
            print(f"   New Best Model! Accuracy: {best_val_acc:.4f}, AUROC: {best_val_auroc:.4f}")
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def measure_computational_complexity(models_dict, device='cuda', features_path=None, adj_path=None):
    """Measure GFLOPs for all baseline models using real WSI data."""
    profiler = ComputationalProfiler(device=device)
    
    print(f"\n{'='*60}")
    print("MEASURING COMPUTATIONAL COMPLEXITY")
    print(f"{'='*60}")
    
    results = profiler.measure_all_baselines(models_dict, features_path=features_path, adj_path=adj_path)
    
    return results


def create_models():
    """Create all baseline models."""
    models = {
        'GraphLSurv': GraphLSurv(input_dim=1024, hidden_dim=256, num_classes=3),
        'GTP': GTP(input_dim=1024, hidden_dim=64, num_classes=3, pool_size=100),
        'GCN': GCN(input_dim=1024, hidden_dim=256, num_classes=3),
        'PatchGCN': PatchGCN(input_dim=1024, hidden_dim=128, num_classes=3),
        'DeepGraphConv': DeepGraphConv(input_dim=1024, hidden_dim=256, num_classes=3)
    }
    return models


def load_dataset(data_root, train_ids_file, val_ids_file, site='LUAD'):
    """Load dataset."""
    
    # Load train/val splits
    with open(train_ids_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    with open(val_ids_file, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    # Create datasets
    train_dataset = GraphDataset(
        root=data_root,
        ids=train_ids,
        site=site,
        use_refined_adj=False
    )
    
    val_dataset = GraphDataset(
        root=data_root,
        ids=val_ids,
        site=site,
        use_refined_adj=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_baseline)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_baseline)
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Reproduce baseline comparison results')
    parser.add_argument('--model', choices=['graphlsurv', 'gtp', 'gcn', 'patchgcn', 'deepgraphconv', 'all'], 
                       default='all', help='Model to train/evaluate')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--train_ids', type=str, required=True,
                       help='Path to train IDs file')
    parser.add_argument('--val_ids', type=str, required=True,
                       help='Path to validation IDs file')
    parser.add_argument('--site', type=str, default='LUAD',
                       help='Dataset site')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./baseline_results',
                       help='Output directory for results')
    parser.add_argument('--compute_flops_only', action='store_true',
                       help='Only compute FLOPs without training')
    parser.add_argument('--features_path', type=str, default=None,
                       help='Path to real features.pt file for FLOP measurement')
    parser.add_argument('--adj_path', type=str, default=None,
                       help='Path to real adj_s.pt file for FLOP measurement')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    models = create_models()
    
    if args.compute_flops_only:
        # Only measure computational complexity
        if args.features_path and args.adj_path:
            print(f"Using real WSI data:")
            print(f"  Features: {args.features_path}")
            print(f"  Adjacency: {args.adj_path}")
        
        flop_results = measure_computational_complexity(
            models, device, 
            features_path=args.features_path, 
            adj_path=args.adj_path
        )
        
        # Save results
        with open(os.path.join(args.output_dir, 'flop_results.json'), 'w') as f:
            json.dump(flop_results, f, indent=2)
        
        print("\nFLOP measurement completed!")
        return
    
    # Load dataset
    train_loader, val_loader = load_dataset(
        args.data_root, args.train_ids, args.val_ids, args.site
    )
    
    # Train models
    all_results = {}
    
    if args.model == 'all':
        model_names = list(models.keys())
    else:
        model_name_map = {
            'graphlsurv': 'GraphLSurv',
            'gtp': 'GTP', 
            'gcn': 'GCN',
            'patchgcn': 'PatchGCN',
            'deepgraphconv': 'DeepGraphConv'
        }
        model_names = [model_name_map[args.model]]
    
    for model_name in model_names:
        model = models[model_name]
        
        # Train model
        results = train_baseline_model(
            model_name, model, train_loader, val_loader,
            epochs=args.epochs, device=device, output_dir=args.output_dir
        )
        
        all_results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Best Validation Accuracy: {results['best_val_acc']:.4f}")
        print(f"  Best Validation AUROC: {results['best_val_auroc']:.4f}")
        print(f"  Best Epoch: {results['best_epoch']}")
    
    # Measure computational complexity
    flop_results = measure_computational_complexity(models, device)
    
    # Combine results
    final_results = {
        'training_results': all_results,
        'computational_results': flop_results
    }
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("FINAL BASELINE COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Accuracy (%)':<12} {'AUROC (%)':<10} {'GFLOPs':<8}")
    print("-"*80)
    
    for model_name in model_names:
        if model_name in all_results and model_name in flop_results:
            acc = all_results[model_name]['best_val_acc'] * 100
            auroc = all_results[model_name]['best_val_auroc'] * 100  
            gflops = flop_results[model_name] if flop_results[model_name] else 0
            
            print(f"{model_name:<15} {acc:<12.1f} {auroc:<10.1f} {gflops:<8.1f}")
    
    print("="*80)
    
    # Save final results
    with open(os.path.join(args.output_dir, 'baseline_comparison_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Baseline comparison completed!")


if __name__ == "__main__":
    main()
