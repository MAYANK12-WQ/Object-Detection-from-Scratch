"""
Object Detector Architecture

Single-stage detector inspired by YOLO architecture.
Uses a backbone CNN for feature extraction and detection head for predictions.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ObjectDetector(nn.Module):
    """
    Single-stage object detector.

    Args:
        num_classes (int): Number of object classes
        backbone (str): Backbone architecture ('resnet18', 'resnet34', 'custom')
        pretrained (bool): Use pretrained backbone weights
        num_anchors (int): Number of anchor boxes per grid cell
        grid_size (int): Output grid size (image_size // 32)
    """

    def __init__(self, num_classes=20, backbone='resnet18', pretrained=True,
                 num_anchors=5, grid_size=14):
        super(ObjectDetector, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        # Backbone network for feature extraction
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 512
        else:
            self.backbone = self._create_custom_backbone()
            backbone_out_channels = 512

        # Detection head
        # Output: num_anchors × (5 + num_classes) per grid cell
        # 5 = [x, y, w, h, confidence]
        self.detection_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(1024, num_anchors * (5 + num_classes), kernel_size=1)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images [batch_size, 3, H, W]

        Returns:
            torch.Tensor: Predictions [batch_size, grid_size, grid_size, num_anchors, 5 + num_classes]
        """
        batch_size = x.size(0)

        # Extract features
        features = self.backbone(x)  # [batch, 512, grid_size, grid_size]

        # Detection predictions
        detections = self.detection_head(features)  # [batch, anchors*(5+C), grid, grid]

        # Reshape output
        detections = detections.permute(0, 2, 3, 1).contiguous()
        detections = detections.view(batch_size, self.grid_size, self.grid_size,
                                    self.num_anchors, 5 + self.num_classes)

        # Apply activations
        # Sigmoid for x, y, confidence
        detections[..., 0:2] = torch.sigmoid(detections[..., 0:2])  # x, y
        detections[..., 4] = torch.sigmoid(detections[..., 4])      # confidence

        # Exponential for w, h
        detections[..., 2:4] = torch.exp(detections[..., 2:4])      # w, h

        # Softmax for class probabilities
        detections[..., 5:] = torch.softmax(detections[..., 5:], dim=-1)

        return detections

    def _create_custom_backbone(self):
        """Create a custom lightweight backbone."""
        layers = []

        # Block 1: 3 → 64
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        # Block 2: 64 → 128
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        # Block 3: 128 → 256
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        # Block 4: 256 → 512
        layers.extend([
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test the detector model."""
    model = ObjectDetector(num_classes=20, backbone='resnet18', pretrained=False)

    # Test input
    x = torch.randn(2, 3, 448, 448)

    # Forward pass
    output = model(x)

    print("Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [batch, grid, grid, anchors, 5+classes]")
    print(f"Total parameters: {model.get_num_parameters():,}")


if __name__ == "__main__":
    test_model()
