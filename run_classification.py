"""
红绿灯分类系统主执行脚本
一键训练和测试分类模型
"""

import argparse
import os
import sys

from infer_classifier import TrafficLightPredictor
from train_classifier import main as train_model


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import PIL
        import torch
        import torchvision

        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def train():
    """训练分类模型"""
    print("=" * 60)
    print("开始训练红绿灯分类模型")
    print("=" * 60)
    train_model()
    print("\n训练完成！模型已保存到 models/traffic_light_classifier.pth")


def test(image_path=None):
    """测试分类模型"""
    model_path = "models/traffic_light_classifier.pth"

    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        print("请先运行: python run_classification.py --train")
        return

    predictor = TrafficLightPredictor(model_path)

    if image_path:
        # 测试单张图片
        if not os.path.exists(image_path):
            print(f"✗ 图片文件不存在: {image_path}")
            return

        print("=" * 60)
        print(f"预测图片: {image_path}")
        print("=" * 60)
        result = predictor.predict_single(image_path)

        print(f"\n预测类别: {result['class']}")
        print(f"置信度: {result['confidence']:.2f}%")
        print("\n各类别概率:")
        for class_name, prob in result["probabilities"].items():
            print(f"  {class_name}: {prob:.2f}%")
    else:
        # 随机测试几张图片
        import glob

        test_images = []
        for color in ["red", "green", "yellow"]:
            test_dir = f"traffic_light_images/test/{color}"
            if os.path.exists(test_dir):
                images = glob.glob(f"{test_dir}/*.jpg")[:3]
                test_images.extend(images)

        if not test_images:
            print("✗ 未找到测试图片")
            return

        print("=" * 60)
        print(f"测试 {len(test_images)} 张图片")
        print("=" * 60)

        for img_path in test_images:
            result = predictor.predict_single(img_path)
            actual_color = img_path.split("/")[-2]
            status = "✓" if result["class"] == actual_color else "✗"
            print(
                f"{status} {img_path.split('/')[-1]}: "
                f"预测={result['class']} "
                f"(实际={actual_color}) "
                f"置信度={result['confidence']:.1f}%"
            )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="红绿灯分类系统")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--test", type=str, help="测试单张图片")
    parser.add_argument("--check", action="store_true", help="检查依赖")

    args = parser.parse_args()

    if args.check:
        check_dependencies()
        return

    if not check_dependencies():
        return

    if args.train:
        train()
    elif args.test:
        test(args.test)
    else:
        # 默认运行测试
        print("未指定操作，执行默认测试...")
        test()


if __name__ == "__main__":
    main()
