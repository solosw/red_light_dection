"""
红绿灯分类模型训练脚本
使用Focal Loss处理数据不平衡，结合数据增强提升性能
"""

import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_class_names, get_data_loaders
from models import FocalLoss, TrafficLightClassifier


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练中")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_dir = "traffic_light_images/training"
    test_dir = "traffic_light_images/test"

    batch_size = 64
    num_epochs = 5
    learning_rate = 3e-4
    weight_decay = 1e-4

    train_loader, test_loader, train_size, test_size = get_data_loaders(
        train_dir, test_dir, batch_size=batch_size
    )
    print(f"训练集大小: {train_size}, 测试集大小: {test_size}")

    model = TrafficLightClassifier(num_classes=3, pretrained=True)
    model = model.to(device)

    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    writer = SummaryWriter("runs/traffic_light_classifier")

    best_acc = 0
    best_loss = float("inf")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"[{epoch_time:.1f}s] "
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "acc": val_acc,
                },
                "checkpoints/best_model.pth",
            )
            print(f"  保存最佳模型: Acc={best_acc:.2f}%")

        if early_stopping(val_loss, model):
            print(f"\n早停触发！在第{epoch + 1}个epoch停止训练")
            print(f"最佳验证准确率: {best_acc:.2f}%")
            break

    print("\n训练完成！")
    print("=" * 60)
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"最佳验证损失: {best_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/traffic_light_classifier.pth")
    print("模型已保存到: models/traffic_light_classifier.pth")

    print("\n训练曲线:")
    print(f"最终训练损失: {train_losses[-1]:.4f}, 训练准确率: {train_accs[-1]:.2f}%")
    print(f"最终验证损失: {val_losses[-1]:.4f}, 验证准确率: {val_accs[-1]:.2f}%")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    main()
