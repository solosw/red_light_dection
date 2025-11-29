"""
数据集加载和预处理
包含数据增强和平衡采样
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def get_transforms():
    """获取训练和验证的数据变换"""

    # 训练数据变换（包含数据增强）
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 验证数据变换（只做标准化）
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_data_loaders(train_dir, test_dir, batch_size=32, num_workers=2):
    """创建带权重采样的数据加载器"""

    train_transform, val_transform = get_transforms()

    # 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # 计算类别权重（处理数据不平衡）
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_dataset.targets])
    sample_weights = torch.from_numpy(sample_weights).float()

    # 创建权重采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def get_class_names():
    """获取类别名称"""
    return ["red","green", "yellow"]


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader, train_size, test_size = get_data_loaders(
        "traffic_light_images/training", "traffic_light_images/test", batch_size=32
    )

    print(f"训练集大小: {train_size}")
    print(f"测试集大小: {test_size}")
    print(f"类别: {get_class_names()}")

    # 测试一个batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: 图像形状 {images.shape}, 标签形状 {labels.shape}")
        break
