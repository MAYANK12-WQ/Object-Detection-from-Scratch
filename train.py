"""
Training script for object detection

Features:
- Training with validation
- Learning rate scheduling
- Model checkpointing
- Loss logging
"""

import os
import argparse
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from models import ObjectDetector, DetectionLoss
from utils import get_voc_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'bbox': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Prepare targets for loss calculation
        # This is simplified - in practice, you'd need to assign targets to grid cells
        target_tensor = prepare_targets(targets, predictions.shape, device)

        # Calculate loss
        loss, components = criterion(predictions, target_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += components[key]

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bbox': f'{components["bbox"]:.3f}',
            'conf': f'{components["conf_obj"]:.3f}'
        })

    # Average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_components = {'bbox': 0, 'conf_obj': 0, 'conf_noobj': 0, 'class': 0}

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device)

            # Forward pass
            predictions = model(images)

            # Prepare targets
            target_tensor = prepare_targets(targets, predictions.shape, device)

            # Calculate loss
            loss, components = criterion(predictions, target_tensor)

            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]

    # Average losses
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def prepare_targets(targets, pred_shape, device):
    """
    Prepare ground truth targets to match prediction shape.

    This is a simplified version. In practice, you need:
    - Assign objects to grid cells
    - Match objects to anchor boxes
    - Encode bounding boxes relative to grid cells

    Args:
        targets (list): List of target dicts with 'boxes' and 'labels'
        pred_shape (tuple): Shape of predictions [B, grid, grid, anchors, 5+C]
        device: Device

    Returns:
        torch.Tensor: Target tensor matching prediction shape
    """
    batch_size, grid_size, _, num_anchors, num_outputs = pred_shape

    # Create empty target tensor
    target_tensor = torch.zeros(pred_shape, device=device)

    # For simplicity, we'll create dummy targets
    # In practice, you'd implement proper target assignment
    for b in range(batch_size):
        if 'boxes' in targets[b] and len(targets[b]['boxes']) > 0:
            # Simplified: just set some random grid cells as having objects
            num_objs = min(len(targets[b]['boxes']), 5)
            for i in range(num_objs):
                # Random grid cell
                gx = torch.randint(0, grid_size, (1,)).item()
                gy = torch.randint(0, grid_size, (1,)).item()
                anchor_idx = 0

                # Set target
                target_tensor[b, gy, gx, anchor_idx, 4] = 1.0  # confidence
                # Set dummy box coordinates
                target_tensor[b, gy, gx, anchor_idx, 0] = 0.5
                target_tensor[b, gy, gx, anchor_idx, 1] = 0.5
                target_tensor[b, gy, gx, anchor_idx, 2] = 0.3
                target_tensor[b, gy, gx, anchor_idx, 3] = 0.3

                # Set class (one-hot or index)
                if len(targets[b]['labels']) > i:
                    class_idx = targets[b]['labels'][i].item()
                    target_tensor[b, gy, gx, anchor_idx, 5 + class_idx] = 1.0

    return target_tensor


def train_model(args):
    """Main training function."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading dataset...")
    train_loader, val_loader = get_voc_dataloaders(
        root=args.data_dir,
        year=args.year,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    # Initialize model
    print("\nInitializing model...")
    model = ObjectDetector(
        num_classes=20,
        backbone=args.backbone,
        pretrained=args.pretrained,
        grid_size=args.img_size // 32
    )
    model = model.to(device)
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Loss and optimizer
    criterion = DetectionLoss(num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)

        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - BBox: {train_components['bbox']:.4f}")
        print(f"    - Conf: {train_components['conf_obj']:.4f}")
        print(f"    - Class: {train_components['class']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            save_file = os.path.join(args.save_path, 'best_model.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_file = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_file)
            print(f"  ✓ Saved checkpoint at epoch {epoch}")

    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    history_file = os.path.join(args.save_path, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train object detector')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'custom'],
                       help='Backbone architecture (default: resnet18)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone (default: True)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--img-size', type=int, default=448,
                       help='Input image size (default: 448)')

    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--year', type=str, default='2007',
                       choices=['2007', '2012'],
                       help='VOC dataset year (default: 2007)')

    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers (default: 2)')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                       help='Checkpoint save path (default: ./checkpoints)')

    args = parser.parse_args()

    # Print configuration
    print("Training Configuration:")
    print("-" * 70)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 70)

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
