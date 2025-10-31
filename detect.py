"""
Object detection inference script

Detect objects in images using trained model.
"""

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import ObjectDetector
from utils import non_max_suppression, VOC_CLASSES


# Color map for visualization
COLORS = np.random.randint(0, 255, size=(len(VOC_CLASSES), 3), dtype=np.uint8)


def load_model(model_path, device='cuda'):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device)

    model = ObjectDetector(num_classes=20, backbone='resnet18')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")

    return model


def preprocess_image(image_path, img_size=448):
    """Preprocess image for inference."""
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (width, height)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, orig_size


def postprocess_predictions(predictions, conf_threshold=0.5, iou_threshold=0.5):
    """
    Post-process model predictions.

    Args:
        predictions (torch.Tensor): Model output [B, grid, grid, anchors, 5+C]
        conf_threshold (float): Confidence threshold
        iou_threshold (float): NMS IoU threshold

    Returns:
        list: Detections for each image
    """
    batch_size, grid_size, _, num_anchors, num_outputs = predictions.shape

    # Reshape predictions to [batch, num_boxes, 5+C]
    predictions = predictions.view(batch_size, -1, num_outputs)

    # Apply NMS
    detections = non_max_suppression(
        predictions,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    return detections


def draw_detections(image_path, detections, output_path, img_size=448):
    """
    Draw bounding boxes on image.

    Args:
        image_path (str): Path to input image
        detections (torch.Tensor): Detections [N, 6] - [x1, y1, x2, y2, conf, class]
        output_path (str): Path to save output
        img_size (int): Model input size
    """
    # Load image
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    # Scale factor
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    # Draw each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()

        # Scale coordinates back to original image size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Get class info
        class_id = int(cls)
        class_name = VOC_CLASSES[class_id]
        color = COLORS[class_id].tolist()

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f'{class_name}: {conf:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)

        cv2.rectangle(image, (x1, label_y - label_size[1] - 10),
                     (x1 + label_size[0], label_y), color, -1)
        cv2.putText(image, label, (x1, label_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save output
    cv2.imwrite(output_path, image)
    print(f"Detection result saved to {output_path}")

    return image


def detect_image(model, image_path, device, conf_threshold=0.5,
                iou_threshold=0.5, img_size=448, output_path=None):
    """
    Perform detection on a single image.

    Args:
        model: Trained model
        image_path (str): Path to input image
        device: Device to use
        conf_threshold (float): Confidence threshold
        iou_threshold (float): NMS IoU threshold
        img_size (int): Model input size
        output_path (str): Path to save output (optional)

    Returns:
        torch.Tensor: Detections
    """
    # Preprocess
    image_tensor, orig_size = preprocess_image(image_path, img_size)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Post-process
    detections = postprocess_predictions(
        predictions,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    detections = detections[0]  # Get first image

    # Print detections
    print(f"\nDetected {len(detections)} objects:")
    print("-" * 60)
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        class_name = VOC_CLASSES[int(cls)]
        print(f"{i+1}. {class_name}: {conf:.3f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print("-" * 60)

    # Visualize
    if output_path:
        draw_detections(image_path, detections, output_path, img_size)

    return detections


def main():
    parser = argparse.ArgumentParser(description='Object detection inference')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save output (default: input_path + _detected.jpg)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=448,
                       help='Input image size (default: 448)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)

    # Set output path
    if args.output_path is None:
        base_name = args.image_path.rsplit('.', 1)[0]
        args.output_path = f"{base_name}_detected.jpg"

    # Detect objects
    print(f"\nProcessing image: {args.image_path}")
    detections = detect_image(
        model, args.image_path, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        img_size=args.img_size,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
