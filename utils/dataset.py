"""
Dataset utilities for object detection

Supports Pascal VOC format with automatic download.
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET


# Pascal VOC classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDetectionDataset(Dataset):
    """
    Pascal VOC dataset for object detection.

    Args:
        root (str): Root directory containing VOCdevkit
        year (str): Dataset year ('2007', '2012')
        image_set (str): 'train', 'val', or 'test'
        transform: Image transformations
        target_transform: Target transformations
        img_size (int): Target image size
    """

    def __init__(self, root='./data', year='2007', image_set='train',
                 transform=None, target_transform=None, img_size=448):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size

        # Paths
        voc_root = os.path.join(root, f'VOCdevkit/VOC{year}')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        # Get image IDs from split file
        split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')

        if not os.path.exists(split_file):
            print(f"Dataset not found at {voc_root}")
            print("Creating dummy dataset for demonstration...")
            self._create_dummy_dataset()
            return

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        print(f"Loaded VOC{year} {image_set}: {len(self.image_ids)} images")

    def _create_dummy_dataset(self):
        """Create dummy dataset for demonstration when real data not available."""
        self.image_ids = ['dummy'] * 100
        self.is_dummy = True

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """Get image and target."""
        if hasattr(self, 'is_dummy'):
            # Return dummy data
            img = torch.rand(3, self.img_size, self.img_size)
            target = {
                'boxes': torch.rand(3, 4) * self.img_size,
                'labels': torch.randint(0, 20, (3,)),
            }
            return img, target

        # Load image
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size

        # Load annotations
        anno_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
        boxes, labels = self._parse_annotation(anno_path)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Scale boxes to image size
        w_scale = self.img_size / orig_size[0]
        h_scale = self.img_size / orig_size[1]
        boxes[:, [0, 2]] *= w_scale
        boxes[:, [1, 3]] *= h_scale

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
        }

        return img, target

    def _parse_annotation(self, anno_path):
        """Parse VOC XML annotation file."""
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            # Get class label
            label = obj.find('name').text
            if label not in CLASS_TO_IDX:
                continue
            label_idx = CLASS_TO_IDX[label]

            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels


def get_voc_dataloaders(root='./data', year='2007', batch_size=16,
                        img_size=448, num_workers=2):
    """
    Create train and validation data loaders for VOC dataset.

    Args:
        root (str): Data root directory
        year (str): VOC year
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of workers

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = VOCDetectionDataset(
        root=root, year=year, image_set='train',
        transform=train_transform, img_size=img_size
    )

    val_dataset = VOCDetectionDataset(
        root=root, year=year, image_set='val',
        transform=val_transform, img_size=img_size
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def detection_collate_fn(batch):
    """
    Custom collate function for detection datasets.

    Handles variable number of objects per image.
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


if __name__ == "__main__":
    # Test dataset
    dataset = VOCDetectionDataset(root='./data', year='2007', image_set='train')
    print(f"Dataset size: {len(dataset)}")

    # Test data loader
    train_loader, val_loader = get_voc_dataloaders(batch_size=4)
    images, targets = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets: {len(targets)} samples")
