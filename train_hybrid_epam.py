import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import torchvision.utils as vutils

# Import the EigenCAM model
from hybrid_EPAM import ImprovedHybridModel
from utils_HelperFunction import extract_eigencam_maps, save_attention_visualizations


class DeadAssayDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None, class_mapping=None):
        """
        Args:
            img_dir (string): Directory with all the images
            csv_file (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            class_mapping (dict, optional): Fixed class mapping 
        """
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        
        if class_mapping is None:
            
            unique_diagnostics = sorted(self.annotations['diagnostic'].unique())
            self.class_to_idx = {diagnostic: idx for idx, diagnostic in enumerate(unique_diagnostics)}
        else:
            self.class_to_idx = class_mapping
            
        self.idx_to_class = {idx: diagnostic for diagnostic, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        print(f"Loaded dataset with {len(self.annotations)} images and {self.num_classes} classes")
        print(f"Class mapping: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, f"{img_name}.tif")
        
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = os.path.join(self.img_dir, f"{img_name}{ext}")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
                    
        image = Image.open(img_path).convert('RGB')
        

        diagnostic = self.annotations.iloc[idx]['diagnostic']
        label = self.class_to_idx[diagnostic]
        
        if self.transform:
            image = self.transform(image)
        return image, label, img_name
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = self.annotations['diagnostic'].value_counts().to_dict()
        total_samples = len(self.annotations)
        
        weights = {self.class_to_idx[cls]: total_samples / (len(class_counts) * count) 
                  for cls, count in class_counts.items() if cls in self.class_to_idx}
        
        return torch.tensor([weights.get(i, 1.0) for i in range(self.num_classes)])

# Enhanced data loading function with stronger augmentations
def prepare_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances
    train_img_dir = os.path.join(data_dir, 'Cell_death6', 'train', 'imgs')
    train_csv = os.path.join(data_dir, 'Cell_death6', 'train', 'Cell_Death_parsed_train.csv')
    
    test_img_dir = os.path.join(data_dir, 'Cell_death', 'Test_dataset', 'imgs2')
    test_csv = os.path.join(data_dir, 'Cell_death', 'Test_dataset', 'Cell_Death_parsed_test2.csv')
    
    # First, determine all possible classes from both datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    all_classes = sorted(set(train_df['diagnostic'].unique()) | set(test_df['diagnostic'].unique()))
    
    # Create fixed class mapping (consistent across datasets)
    fixed_class_mapping = {cls: idx for idx, cls in enumerate(all_classes)}
    print(f"Using consistent class mapping: {fixed_class_mapping}")
    
    # Create full training dataset with fixed mapping
    full_train_dataset = DeadAssayDataset(
        img_dir=train_img_dir,
        csv_file=train_csv,
        transform=train_transform,
        class_mapping=fixed_class_mapping
    )
    
    # Split into train and validation sets (80:20)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Create train/validation split with seed for reproducibility
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with test transforms
    # This is important - we need to apply validation dataset with test transforms
    val_dataset_with_test_transform = DeadAssayDataset(
        img_dir=train_img_dir,
        csv_file=train_csv,
        transform=test_transform,
        class_mapping=fixed_class_mapping
    )
    
    # Get indices from train/val split
    val_indices = list(range(len(full_train_dataset)))[train_size:]
    
    # Create a smaller visualization set from validation data (for EigenCAM analysis)
    vis_indices = val_indices[:min(50, len(val_indices))]  # Take first 50 validation samples for visualization
    
    # Create test dataset with the same fixed mapping
    test_dataset = DeadAssayDataset(
        img_dir=test_img_dir,
        csv_file=test_csv,
        transform=test_transform,
        class_mapping=fixed_class_mapping
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        [val_dataset_with_test_transform[i] for i in val_indices], 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Special loader just for visualization with smaller batch size
    vis_loader = DataLoader(
        [val_dataset_with_test_transform[i] for i in vis_indices],
        batch_size=8,  # Smaller batch size for visualization
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class weights for handling imbalanced data
    class_weights = full_train_dataset.get_class_weights()
    
    dataset_info = {
        'num_classes': full_train_dataset.num_classes,
        'class_to_idx': full_train_dataset.class_to_idx,
        'idx_to_class': full_train_dataset.idx_to_class,
        'class_weights': class_weights
    }
    
    return train_loader, val_loader, vis_loader, test_loader, dataset_info

# Function to save EigenCAM visualizations during training
def save_eigencam_batch_visualization(model, data_batch, epoch, output_dir, dataset_info):
    """Generate and save EigenCAM visualizations for a batch of images"""
    device = next(model.parameters()).device
    images, labels, img_names = data_batch
    images = images.to(device)
    labels = labels.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions and attention maps
    with torch.no_grad():
        outputs, feature_maps, attention_maps = model(images, return_attention=True)
        _, preds = torch.max(outputs, 1)
    
    # Convert labels to class names
    idx_to_class = dataset_info['idx_to_class']
    label_names = [idx_to_class[label.item()] for label in labels]
    pred_names = [idx_to_class[pred.item()] for pred in preds]
    
    # Create a figure with subplots for each image in the batch
    batch_size = len(images)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    # If batch_size is 1, wrap axes in a list for consistent indexing
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image: {img_names[i]}\nTrue: {label_names[i]}")
        axes[i, 0].axis('off')
        
        # Feature visualization (stage 4 features)
        if 'stage4' in feature_maps:
            feat_map = feature_maps['stage4'][i].mean(dim=0).cpu().numpy()
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            axes[i, 1].imshow(feat_map, cmap='viridis')
            axes[i, 1].set_title(f"Stage 4 Features")
            axes[i, 1].axis('off')
        
        # EigenCAM attention visualization
        if 'final' in attention_maps:
            att_map = attention_maps['final'][i, 0].cpu().numpy()
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            
            # Create a heatmap overlay on the original image
            axes[i, 2].imshow(img)
            heatmap = axes[i, 2].imshow(att_map, cmap='jet', alpha=0.6)
            axes[i, 2].set_title(f"EigenCAM Attention\nPred: {pred_names[i]}")
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"eigencam_epoch_{epoch}.png"), dpi=150)
    plt.close(fig)
    
    return

# Enhanced training function with EigenCAM visualization
def train_hybrid_model_with_eigencam(model, train_loader, val_loader, vis_loader, dataset_info, 
                                     num_epochs=30, device=None, output_dir="results"):
    """
    Train the hybrid model with EigenCAM visualization and comprehensive metrics tracking
    
    Args:
        model: The HybridConvNeXtMobileViT model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        vis_loader: DataLoader for visualization samples
        dataset_info: Information about the dataset
        num_epochs: Number of training epochs
        device: Device to use for training
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Initialize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}")
    model = model.to(device)
    
    # Use weighted loss for imbalanced dataset
    class_weights = dataset_info['class_weights'].to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Initialize tracking variables
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets, _) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_targets = []
        all_preds = []
        all_probs = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, targets, _ in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for metrics calculation
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # For multi-class ROC, store probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss / (val_pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Generate EigenCAM visualizations every few epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print("Generating EigenCAM visualizations...")
            # Get a batch from the visualization loader
            try:
                vis_batch = next(iter(vis_loader))
                save_eigencam_batch_visualization(
                    model=model,
                    data_batch=vis_batch,
                    epoch=epoch,
                    output_dir=os.path.join(output_dir, "visualizations"),
                    dataset_info=dataset_info
                )
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f'Epoch: {epoch+1}/{num_epochs} ({elapsed_time:.1f}s) | Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'class_mapping': dataset_info['class_to_idx']
            }, os.path.join(output_dir, "checkpoints", f'model_epoch_{epoch+1}.pth'))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'class_mapping': dataset_info['class_to_idx']
            }, os.path.join(output_dir, "checkpoints", 'best_hybrid_model.pth'))
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    return model, train_losses, val_losses, train_accs, val_accs

# Enhanced evaluation function with EigenCAM visualization
def evaluate_model_with_eigencam(model, test_loader, dataset_info, device=None, output_dir="results"):
    """
    Evaluate the model on the test set with EigenCAM visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Create output directory for attention maps
    attention_dir = os.path.join(output_dir, "test_attention_maps")
    os.makedirs(attention_dir, exist_ok=True)
    
    # Sample images for visualization
    vis_images = []
    vis_preds = []
    vis_targets = []
    vis_names = []
    vis_attention_maps = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_names) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions and attention maps
            outputs, feature_maps, attention_maps = model(inputs, return_attention=True)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Save a few samples for visualization
            if batch_idx < 5:  # Save first 5 batches
                for i in range(min(4, len(inputs))):  # Save up to 4 samples per batch
                    vis_images.append(inputs[i].cpu())
                    vis_preds.append(preds[i].item())
                    vis_targets.append(targets[i].item())
                    vis_names.append(img_names[i])
                    
                    if 'final' in attention_maps:
                        vis_attention_maps.append(attention_maps['final'][i].cpu())
    
    # Concatenate all probabilities
    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    
    # Calculate accuracy
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    accuracy = 100 * np.mean(all_preds == all_targets)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    class_names = [dataset_info['idx_to_class'][i] for i in range(len(dataset_info['idx_to_class']))]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Print classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=3)
    print("\nClassification Report:")
    print(report)
    
    # Save the report to a text file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate ROC curves for each class (one-vs-all)
    plt.figure(figsize=(12, 10))
    
    # Calculate ROC curves
    for i in range(len(class_names)):
        if i < all_probs.shape[1]:  # Ensure the class exists in predictions
            fpr, tpr, _ = roc_curve(all_targets == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-All)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    
    # Visualize sample images with attention maps
    if vis_images:
        idx_to_class = dataset_info['idx_to_class']
        fig, axes = plt.subplots(len(vis_images), 2, figsize=(10, 4 * len(vis_images)))
        
        for i, (img, pred, target, name, att_map) in enumerate(zip(vis_images, vis_preds, vis_targets, vis_names, vis_attention_maps)):
            # Original image
            img_np = img.permute(1, 2, 0).numpy()
            # Denormalize
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            axes[i, 0].imshow(img_np)
            pred_class = idx_to_class[pred]
            true_class = idx_to_class[target]
            axes[i, 0].set_title(f"Image: {name}\nTrue: {true_class}, Pred: {pred_class}")
            axes[i, 0].axis('off')
            
            # Attention map overlay
            att_np = att_map[0].numpy()
            att_np = (att_np - att_np.min()) / (att_np.max() - att_np.min() + 1e-8)
            
            axes[i, 1].imshow(img_np)
            axes[i, 1].imshow(att_np, cmap='jet', alpha=0.6)
            axes[i, 1].set_title(f"EigenCAM Attention")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_attention_maps.png'))
    
    # Return evaluation metrics
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    return results

# Main function for model training and evaluation
def main():
    # Set up basic configurations
    config = {
        'model_name': 'HybridConvNeXtMobileViT_EigenCAM',
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.0001,
        'weight_decay': 0.05,
        'img_size': 224,
        'data_dir': "/home/ipcv/Harshal/Death-Cell-Assay/data",
        'output_dir': f"results/HybridEigenCAM_{datetime.now().strftime('%Y%m%d_%H%M')}"
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Prepare data with visualization loader
    train_loader, val_loader, vis_loader, test_loader, dataset_info = prepare_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Initialize model with EigenCAM support
    model = ImprovedHybridModel(
        num_classes=dataset_info['num_classes'],
        use_eigen_attention=True  # Enable EigenCAM attention
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    print("Starting model training...")
    model, train_losses, val_losses, train_accs, val_accs = train_hybrid_model_with_eigencam(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vis_loader=vis_loader,
        dataset_info=dataset_info,
        num_epochs=config['num_epochs'],
        device=device,
        output_dir=config['output_dir']
    )
    
    print("Training completed!")
    
    # Load best model for evaluation
    best_model_path = os.path.join(config['output_dir'], "checkpoints", "best_hybrid_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate model on test dataset
    print("Evaluating model on test dataset...")
    test_results = evaluate_model_with_eigencam(
        model=model,
        test_loader=test_loader,
        dataset_info=dataset_info,
        device=device,
        output_dir=config['output_dir']
    )
    
    # Extract specific attention maps for further analysis
    print("Generating detailed EigenCAM visualizations...")
    try:
        # Create directory for EigenCAM maps
        eigencam_dir = os.path.join(config['output_dir'], "eigencam_analysis")
        os.makedirs(eigencam_dir, exist_ok=True)
        
        # Extract and save EigenCAM visualizations for test data
        extract_eigencam_maps(
            model=model,
            dataloader=test_loader,
            idx_to_class=dataset_info['idx_to_class'],
            output_dir=eigencam_dir,
            device=device,
            num_samples=50  # Analyze first 50 test samples
        )
        
        # Save attention visualizations
        save_attention_visualizations(
            model=model,
            dataloader=test_loader,
            idx_to_class=dataset_info['idx_to_class'],
            output_dir=eigencam_dir,
            device=device,
            num_samples=20  # Visualize first 20 test samples
        )
    except Exception as e:
        print(f"Error during EigenCAM analysis: {e}")
    
    print(f"All results saved to {config['output_dir']}")
    print("Done!")

if __name__ == "__main__":
    main()