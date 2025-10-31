"""
Loss Functions for Object Detection

Implements multi-task loss combining:
- Localization loss (bounding box regression)
- Confidence loss (objectness score)
- Classification loss (class prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bbox_utils import bbox_iou, xywh_to_xyxy


class DetectionLoss(nn.Module):
    """
    Multi-task loss for object detection.

    Combines:
    - L_bbox: Smooth L1 loss for bounding box regression
    - L_conf: Binary cross-entropy for objectness
    - L_class: Cross-entropy for classification

    Args:
        num_classes (int): Number of object classes
        lambda_coord (float): Weight for bbox loss
        lambda_noobj (float): Weight for no-object confidence loss
        lambda_obj (float): Weight for object confidence loss
        lambda_class (float): Weight for classification loss
    """

    def __init__(self, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5,
                 lambda_obj=1.0, lambda_class=1.0):
        super(DetectionLoss, self).__init__()

        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Calculate detection loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch, grid, grid, anchors, 5+C]
            targets (torch.Tensor): Ground truth [batch, grid, grid, anchors, 5+C]
                                   Format: [x, y, w, h, confidence, class_id/one_hot]

        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        device = predictions.device
        batch_size = predictions.size(0)

        # Extract prediction components
        pred_boxes = predictions[..., :4]      # [x, y, w, h]
        pred_conf = predictions[..., 4]        # confidence
        pred_class = predictions[..., 5:]      # class probabilities

        # Extract target components
        target_boxes = targets[..., :4]
        target_conf = targets[..., 4]
        target_class = targets[..., 5:] if targets.size(-1) > 5 else None

        # Create masks for objectness
        obj_mask = target_conf > 0  # Cells that contain objects
        noobj_mask = target_conf == 0  # Cells that don't contain objects

        # ===== Bounding Box Loss =====
        if obj_mask.sum() > 0:
            # Only calculate for cells that contain objects
            loss_x = self.mse_loss(pred_boxes[..., 0][obj_mask],
                                  target_boxes[..., 0][obj_mask])
            loss_y = self.mse_loss(pred_boxes[..., 1][obj_mask],
                                  target_boxes[..., 1][obj_mask])

            # Use square root for w, h (YOLO style)
            loss_w = self.mse_loss(torch.sqrt(pred_boxes[..., 2][obj_mask] + 1e-8),
                                  torch.sqrt(target_boxes[..., 2][obj_mask] + 1e-8))
            loss_h = self.mse_loss(torch.sqrt(pred_boxes[..., 3][obj_mask] + 1e-8),
                                  torch.sqrt(target_boxes[..., 3][obj_mask] + 1e-8))

            loss_bbox = loss_x + loss_y + loss_w + loss_h
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        # ===== Confidence Loss =====
        # Object confidence loss
        if obj_mask.sum() > 0:
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask],
                                         target_conf[obj_mask])
        else:
            loss_conf_obj = torch.tensor(0.0, device=device)

        # No-object confidence loss
        if noobj_mask.sum() > 0:
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask],
                                           target_conf[noobj_mask])
        else:
            loss_conf_noobj = torch.tensor(0.0, device=device)

        # ===== Classification Loss =====
        if target_class is not None and obj_mask.sum() > 0:
            # If target is one-hot encoded
            if target_class.size(-1) == self.num_classes:
                loss_class = F.binary_cross_entropy(
                    pred_class[obj_mask],
                    target_class[obj_mask],
                    reduction='sum'
                )
            # If target is class index
            else:
                target_class_idx = target_class[obj_mask].long()
                loss_class = self.ce_loss(
                    pred_class[obj_mask],
                    target_class_idx
                )
        else:
            loss_class = torch.tensor(0.0, device=device)

        # ===== Total Loss =====
        total_loss = (
            self.lambda_coord * loss_bbox +
            self.lambda_obj * loss_conf_obj +
            self.lambda_noobj * loss_conf_noobj +
            self.lambda_class * loss_class
        )

        # Normalize by batch size
        num_obj = obj_mask.sum().float()
        if num_obj > 0:
            total_loss = total_loss / batch_size

        # Loss components for logging
        loss_components = {
            'total': total_loss.item(),
            'bbox': (self.lambda_coord * loss_bbox / batch_size).item() if num_obj > 0 else 0,
            'conf_obj': (self.lambda_obj * loss_conf_obj / batch_size).item() if num_obj > 0 else 0,
            'conf_noobj': (self.lambda_noobj * loss_conf_noobj / batch_size).item(),
            'class': (self.lambda_class * loss_class / batch_size).item() if num_obj > 0 else 0,
        }

        return total_loss, loss_components


class IoULoss(nn.Module):
    """
    IoU-based loss for bounding box regression.

    More geometrically meaningful than MSE loss.
    """

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        Calculate IoU loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes [N, 4]
            target_boxes (torch.Tensor): Target boxes [N, 4]

        Returns:
            torch.Tensor: IoU loss
        """
        # Calculate IoU
        iou = bbox_iou(pred_boxes, target_boxes, x1y1x2y2=False)

        # IoU loss = 1 - IoU
        loss = 1 - iou.diagonal()

        return loss.mean()


if __name__ == "__main__":
    # Test detection loss
    batch_size = 2
    grid_size = 14
    num_anchors = 5
    num_classes = 20

    predictions = torch.rand(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
    targets = torch.rand(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)

    criterion = DetectionLoss(num_classes=num_classes)
    loss, components = criterion(predictions, targets)

    print("Loss Test:")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Components: {components}")
