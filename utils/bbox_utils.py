"""
Bounding Box Utilities

Functions for:
- IoU calculation
- Non-Maximum Suppression (NMS)
- Box coordinate transformations
- Box encoding/decoding
"""

import torch
import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (torch.Tensor): First box [N, 4] or [4]
        box2 (torch.Tensor): Second box [M, 4] or [4]
        x1y1x2y2 (bool): If True, boxes are in (x1, y1, x2, y2) format,
                        else (x_center, y_center, width, height)
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: IoU values [N, M] or scalar
    """
    # Convert to x1y1x2y2 format if needed
    if not x1y1x2y2:
        # Convert from xywh to x1y1x2y2
        box1 = xywh_to_xyxy(box1)
        box2 = xywh_to_xyxy(box2)

    # Ensure 2D tensors
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0:1], box2[:, 1:2], box2[:, 2:3], box2[:, 3:4]

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1.T)
    inter_y1 = torch.max(b1_y1, b2_y1.T)
    inter_x2 = torch.min(b1_x2, b2_x2.T)
    inter_y2 = torch.min(b1_y2, b2_y2.T)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area.T - inter_area + eps

    # IoU
    iou = inter_area / union_area

    return iou


def xywh_to_xyxy(boxes):
    """
    Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2).

    Args:
        boxes (torch.Tensor): Boxes in xywh format [..., 4]

    Returns:
        torch.Tensor: Boxes in xyxy format [..., 4]
    """
    x_center, y_center, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height).

    Args:
        boxes (torch.Tensor): Boxes in xyxy format [..., 4]

    Returns:
        torch.Tensor: Boxes in xywh format [..., 4]
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x_center, y_center, w, h], dim=-1)


def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.5, max_detections=300):
    """
    Perform Non-Maximum Suppression (NMS) on predictions.

    Args:
        predictions (torch.Tensor): Predictions [batch_size, num_boxes, 5 + num_classes]
                                   Format: [x, y, w, h, conf, class_probs...]
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        max_detections (int): Maximum number of detections per image

    Returns:
        list: List of detections for each image [x1, y1, x2, y2, conf, class_id]
    """
    batch_size = predictions.shape[0]
    output = []

    for i in range(batch_size):
        pred = predictions[i]

        # Filter by confidence threshold
        conf = pred[:, 4]
        mask = conf > conf_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            output.append(torch.empty((0, 6)))
            continue

        # Get class predictions
        class_probs = pred[:, 5:]
        class_conf, class_pred = class_probs.max(dim=1, keepdim=True)

        # Multiply objectness with class probability
        conf = pred[:, 4:5] * class_conf

        # Convert boxes from xywh to xyxy
        boxes = xywh_to_xyxy(pred[:, :4])

        # Concatenate [x1, y1, x2, y2, conf, class]
        detections = torch.cat([boxes, conf, class_pred.float()], dim=1)

        # NMS per class
        unique_classes = detections[:, 5].unique()
        final_detections = []

        for cls in unique_classes:
            # Get detections of this class
            cls_mask = detections[:, 5] == cls
            cls_detections = detections[cls_mask]

            # Sort by confidence
            conf_sort_idx = cls_detections[:, 4].argsort(descending=True)
            cls_detections = cls_detections[conf_sort_idx]

            # NMS
            keep_boxes = []
            while cls_detections.shape[0] > 0:
                # Keep box with highest confidence
                keep_boxes.append(cls_detections[0])

                if cls_detections.shape[0] == 1:
                    break

                # Calculate IoU with remaining boxes
                ious = bbox_iou(cls_detections[0:1, :4], cls_detections[1:, :4], x1y1x2y2=True)

                # Remove boxes with IoU > threshold
                iou_mask = ious[0] < iou_threshold
                cls_detections = cls_detections[1:][iou_mask]

            if keep_boxes:
                final_detections.append(torch.stack(keep_boxes))

        if final_detections:
            detections = torch.cat(final_detections, dim=0)
            # Limit to max detections
            if detections.shape[0] > max_detections:
                detections = detections[:max_detections]
        else:
            detections = torch.empty((0, 6))

        output.append(detections)

    return output


def clip_boxes(boxes, img_size):
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor): Boxes in xyxy format [..., 4]
        img_size (tuple): Image size (height, width)

    Returns:
        torch.Tensor: Clipped boxes
    """
    h, w = img_size
    boxes[..., 0] = torch.clamp(boxes[..., 0], 0, w)  # x1
    boxes[..., 1] = torch.clamp(boxes[..., 1], 0, h)  # y1
    boxes[..., 2] = torch.clamp(boxes[..., 2], 0, w)  # x2
    boxes[..., 3] = torch.clamp(boxes[..., 3], 0, h)  # y2
    return boxes


def scale_boxes(boxes, from_size, to_size):
    """
    Scale bounding boxes from one image size to another.

    Args:
        boxes (torch.Tensor): Boxes [..., 4]
        from_size (tuple): Original size (height, width)
        to_size (tuple): Target size (height, width)

    Returns:
        torch.Tensor: Scaled boxes
    """
    from_h, from_w = from_size
    to_h, to_w = to_size

    boxes[..., 0] = boxes[..., 0] * (to_w / from_w)
    boxes[..., 1] = boxes[..., 1] * (to_h / from_h)
    boxes[..., 2] = boxes[..., 2] * (to_w / from_w)
    boxes[..., 3] = boxes[..., 3] * (to_h / from_h)

    return boxes


if __name__ == "__main__":
    # Test IoU calculation
    box1 = torch.tensor([[0, 0, 10, 10]])
    box2 = torch.tensor([[5, 5, 15, 15]])
    iou = bbox_iou(box1, box2, x1y1x2y2=True)
    print(f"IoU test: {iou.item():.4f}")

    # Test NMS
    predictions = torch.rand(1, 100, 25)  # batch=1, boxes=100, 5+20 classes
    predictions[:, :, 4] = torch.rand(100) * 0.8  # Random confidences
    output = non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.5)
    print(f"NMS output: {len(output[0])} detections")
