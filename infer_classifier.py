"""
红绿灯分类模型推理脚本
支持单张图片预测和批量预测
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset import get_class_names
from models import TrafficLightClassifier


class TrafficLightPredictor:
    """红绿灯分类预测器"""

    def __init__(self, model_path="models/traffic_light_classifier.pth", device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.class_names = get_class_names()

        self.model = TrafficLightClassifier(num_classes=3, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=False))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict_single(self, image_path):
        """预测单张图片"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence = confidence.item() * 100

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: prob.item() * 100
                for i, prob in enumerate(probabilities[0])
            },
        }

    def predict_batch(self, image_paths):
        """批量预测图片"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result["path"] = image_path
                results.append(result)
            except Exception as e:
                print(f"预测失败 {image_path}: {e}")
                results.append(
                    {
                        "path": image_path,
                        "class": "error",
                        "confidence": 0,
                        "error": str(e),
                    }
                )

        return results

    def predict_from_array(self, image_array):
        """从numpy数组预测（用于视频流）"""
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence = confidence.item() * 100

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: prob.item() * 100
                for i, prob in enumerate(probabilities[0])
            },
        }

