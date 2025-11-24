"""
轻量级红绿灯分类模型
使用MobileNetV3_small作为骨干网络，结合注意力机制提升准确率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TrafficLightClassifier(nn.Module):
    """高效的红绿灯分类器"""

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # 使用预训练的MobileNetV3_small作为骨干网络
        if pretrained:
            self.backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
            )
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # 获取骨干网络的特征维度（classifier是Sequential，取最后一层的in_features）
        in_features = 576

        # 移除原始分类头
        self.backbone.classifier = nn.Identity()

        # 添加自定义分类头
        self.classifier = nn.Sequential(
            # 第一层：Dropout + Linear + ReLU
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            # 第二层：Dropout + Linear + ReLU
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 第三层：Dropout + Linear（输出层）
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class AttentionModule(nn.Module):
    """可选的注意力模块（用于进一步提升准确率）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡问题"""

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    # 测试模型
    model = TrafficLightClassifier(num_classes=3, pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
