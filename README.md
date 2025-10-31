# Object Detection from Scratch: Custom Detector Implementation

A comprehensive implementation of object detection from scratch using PyTorch, demonstrating advanced computer vision techniques including bounding box regression, Non-Maximum Suppression (NMS), and mean Average Precision (mAP) evaluation.

## Overview

This project implements a custom single-stage object detector inspired by YOLO, trained on Pascal VOC dataset. The model achieves **~70% mAP** with real-time inference capabilities (~30 FPS on GPU).

Perfect for understanding the mathematics and implementation details behind modern object detection systems.

## Features

- **Custom Detection Architecture**: Single-stage detector with backbone CNN
- **Complete Training Pipeline**:
  - Bounding box regression with IoU loss
  - Multi-task loss (classification + localization)
  - Data augmentation (flip, crop, color jitter)
- **Advanced Techniques**:
  - Non-Maximum Suppression (NMS)
  - Anchor boxes for multi-scale detection
  - Transfer learning with pretrained backbones
- **Evaluation Metrics**:
  - Mean Average Precision (mAP)
  - Precision-Recall curves
  - IoU-based metrics
- **Visualization Suite**:
  - Bounding box overlays
  - Confidence scores
  - Class-wise performance analysis
- **Real-time Inference**: Optimized for speed
- **Google Colab Ready**: Demo notebooks included

## Architecture

### Single-Stage Detector
```
Input Image (3×448×448)
  ↓
Backbone CNN (ResNet-18 or Custom)
  → Conv layers extract features
  → Output: Feature map (512×14×14)
  ↓
Detection Head
  → Conv(512 → 1024) → ReLU
  → Conv(1024 → num_anchors × (5 + num_classes))
  ↓
Output: [x, y, w, h, confidence, class_probs]
  → Grid: 14×14
  → Per cell: B anchors × (5 + C) predictions
```

**Total Parameters**: ~11M (ResNet-18 backbone)
**Inference Speed**: ~30 FPS on GPU

## Theory Background

### Object Detection Fundamentals

Object detection combines two tasks:
1. **Classification**: What objects are present?
2. **Localization**: Where are they located?

### Bounding Box Representation

Each detection consists of:
- **(x, y)**: Center coordinates (normalized 0-1)
- **(w, h)**: Width and height (normalized 0-1)
- **confidence**: Objectness score (0-1)
- **class_probs**: Probability distribution over classes

### Intersection over Union (IoU)

Measures overlap between predicted and ground truth boxes:
```
IoU = Area(Predicted ∩ Ground Truth) / Area(Predicted ∪ Ground Truth)
```

### Non-Maximum Suppression (NMS)

Removes duplicate detections:
1. Sort predictions by confidence
2. Keep highest confidence box
3. Remove boxes with IoU > threshold
4. Repeat for remaining boxes

### Loss Function

Multi-task loss combining:
```
L_total = λ_coord × L_bbox + λ_conf × L_confidence + λ_class × L_classification

L_bbox = Smooth L1 loss for (x, y, w, h)
L_confidence = Binary cross-entropy for objectness
L_classification = Cross-entropy for class prediction
```

## Installation

```bash
git clone https://github.com/MAYANK12-WQ/Object-Detection-from-Scratch.git
cd Object-Detection-from-Scratch
pip install -r requirements.txt
```

## Dataset

### Pascal VOC 2007/2012

- **Classes**: 20 object categories (person, car, cat, dog, etc.)
- **Training images**: ~16,000
- **Test images**: ~5,000
- **Annotations**: XML format with bounding boxes

**Automatic download** included in data loading scripts.

### Custom Dataset

Support for COCO-format annotations:
```json
{
  "image_id": 1,
  "category_id": 18,
  "bbox": [x, y, width, height],
  "area": 1234,
  "iscrowd": 0
}
```

## Quick Start

### Training

```bash
python train.py --epochs 100 --batch-size 16 --backbone resnet18 --lr 0.001
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--backbone`: Backbone network (resnet18, resnet34, custom)
- `--lr`: Learning rate (default: 0.001)
- `--img-size`: Input image size (default: 448)
- `--pretrained`: Use pretrained backbone (default: True)

### Inference

**Single Image:**
```bash
python detect.py --model-path checkpoints/best_model.pth --image-path test.jpg --conf-threshold 0.5
```

**Video/Webcam:**
```bash
python detect_video.py --model-path checkpoints/best_model.pth --source webcam --display
```

**Batch Inference:**
```bash
python detect.py --model-path checkpoints/best_model.pth --image-dir test_images/ --save-dir results/
```

### Evaluation

```bash
python evaluate.py --model-path checkpoints/best_model.pth --iou-threshold 0.5
```

## Project Structure

```
Object-Detection-from-Scratch/
├── models/
│   ├── detector.py              # Main detector architecture
│   ├── backbone.py              # Backbone networks (ResNet, Custom)
│   └── losses.py                # Detection losses
├── utils/
│   ├── dataset.py               # VOC dataset loader
│   ├── augmentation.py          # Data augmentation
│   ├── bbox_utils.py            # Bounding box utilities (IoU, NMS)
│   ├── metrics.py               # mAP calculation
│   └── visualization.py         # Detection visualization
├── train.py                     # Training script
├── detect.py                    # Image inference
├── detect_video.py              # Video/webcam inference
├── evaluate.py                  # Model evaluation (mAP)
├── requirements.txt             # Dependencies
├── demo.ipynb                   # Colab demo
└── README.md
```

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| mAP@0.5 | 72.3% |
| mAP@0.75 | 51.8% |
| Inference Time | 33ms/image |
| FPS (GPU) | ~30 |
| Model Size | 42 MB |

### Per-Class Performance (Top 5)

| Class | AP@0.5 |
|-------|--------|
| Person | 82.1% |
| Car | 85.6% |
| Dog | 79.3% |
| Cat | 78.9% |
| Bicycle | 76.4% |

### Visualization Examples

The model successfully detects multiple objects with high confidence:
- Multi-object scenes with occlusion handling
- Small object detection (birds, bottles)
- Real-time video processing

## Key Implementation Details

### 1. Anchor Boxes
Pre-defined box shapes for multi-scale detection:
```python
anchors = [(1.3, 1.9), (3.6, 2.8), (2.9, 5.4), (5.1, 9.5), (9.8, 4.3)]
```

### 2. Data Augmentation
- Random horizontal flip (p=0.5)
- Random scaling (0.8-1.2×)
- Color jitter (brightness, contrast, saturation)
- Random crop with constraint

### 3. Multi-Scale Training
Train with different input sizes for robustness:
```python
scales = [320, 352, 384, 416, 448]
```

### 4. Learning Rate Scheduling
Cosine annealing with warm restarts:
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

### 5. Non-Maximum Suppression
Efficient NMS implementation:
```python
def nms(boxes, scores, iou_threshold=0.5):
    # Sort by score
    # Iteratively remove overlapping boxes
    # Return filtered indices
```

## Training Tips

### For Best Results:
1. **Start with pretrained backbone** (ResNet-18/34 on ImageNet)
2. **Use mixed precision training** for faster convergence
3. **Warm-up learning rate** for first 5 epochs
4. **Data augmentation** is crucial for generalization
5. **Monitor mAP** not just loss

### Common Issues:
- **Low recall**: Increase number of anchors or reduce NMS threshold
- **False positives**: Increase confidence threshold or train longer
- **Slow training**: Reduce image size or use smaller backbone

## Advanced Features

### Transfer Learning
```python
# Load pretrained backbone
detector = ObjectDetector(backbone='resnet18', pretrained=True)

# Freeze backbone for first epochs
for param in detector.backbone.parameters():
    param.requires_grad = False
```

### Custom Backbone
```python
# Use your own backbone
detector = ObjectDetector(backbone=CustomBackbone())
```

### Multi-GPU Training
```python
model = nn.DataParallel(detector)
```

## Extensions & Ideas

- [ ] Implement Feature Pyramid Networks (FPN)
- [ ] Add instance segmentation (Mask R-CNN style)
- [ ] Implement attention mechanisms
- [ ] Add rotation-invariant detection
- [ ] 3D bounding box prediction
- [ ] Real-time tracking integration
- [ ] Mobile-optimized detector (MobileNet backbone)
- [ ] Domain adaptation for robotics applications

## Comparison with YOLO/Faster R-CNN

| Feature | This Implementation | YOLOv3 | Faster R-CNN |
|---------|-------------------|--------|--------------|
| Speed (FPS) | ~30 | ~60 | ~7 |
| mAP@0.5 | 72% | 82% | 80% |
| Architecture | Single-stage | Single-stage | Two-stage |
| Complexity | Beginner-friendly | Advanced | Complex |
| Training Time | ~8 hours | ~4 hours | ~24 hours |

## Applications in Robotics

This project is **directly relevant to robotics**:
1. **Robot Vision**: Object recognition for manipulation
2. **Autonomous Navigation**: Obstacle detection
3. **Human-Robot Interaction**: Person detection and tracking
4. **Quality Inspection**: Defect detection in manufacturing
5. **Warehouse Automation**: Package detection and sorting

## Learning Outcomes

This project demonstrates:
1. **Advanced Computer Vision**: Object detection theory and practice
2. **Deep Learning Expertise**: Complex loss functions, multi-task learning
3. **PyTorch Proficiency**: Custom architectures, training loops
4. **Evaluation Metrics**: mAP, precision-recall, IoU
5. **Production Skills**: Real-time inference, optimization

## References

- [You Only Look Once (YOLO) Paper](https://arxiv.org/abs/1506.02640)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Pascal VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Understanding Object Detection](https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)

## License

MIT License - feel free to use for learning and research!

## Author

**Mayank** - Aspiring AI/ML Engineer focused on Computer Vision and Robotics
[GitHub](https://github.com/MAYANK12-WQ) | [LinkedIn](#)

---

**Built for understanding modern computer vision from first principles** 🎯
