"""
演示：如何使用训练好的红绿灯分类模型预测单张图片
"""

import os

from infer_classifier import TrafficLightPredictor


def main():
    print("=" * 60)
    print("红绿灯分类模型预测演示")
    print("=" * 60)

    model_path = "models/traffic_light_classifier.pth"
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        print("请先运行训练: python train_classifier.py")
        return

    predictor = TrafficLightPredictor(model_path)
    print("✓ 预测器创建成功\n")

    # 2. 预测单张图片
    print("方法1: 预测单张图片")
    print("-" * 60)

    test_images = []
    for color in ["red", "green", "yellow"]:
        test_dir = f"traffic_light_images/test/{color}"
        if os.path.exists(test_dir):
            # 每个类别取第一张图测试
            images = os.listdir(test_dir)
            if images:
                test_images.append((f"{test_dir}/{images[0]}", color))

    for img_path, actual_color in test_images:
        result = predictor.predict_single(img_path)
        status = "✓" if result["class"] == actual_color else "✗"
        print(f"\n{status} 图片: {os.path.basename(img_path)}")
        print(f"  实际颜色: {actual_color}")
        print(f"  预测颜色: {result['class']}")
        print(f"  置信度: {result['confidence']:.2f}%")
        print(f"  各类别概率:")
        for class_name, prob in result["probabilities"].items():
            print(f"    {class_name}: {prob:.2f}%")

    # 3. 批量预测示例
    print("\n" + "=" * 60)
    print("方法2: 批量预测图片")
    print("-" * 60)

    all_test_images = []
    for color in ["red", "green", "yellow"]:
        test_dir = f"traffic_light_images/test/{color}"
        if os.path.exists(test_dir):
            images = [f"{test_dir}/{img}" for img in os.listdir(test_dir)[:3]]
            all_test_images.extend(images)

    if all_test_images:
        results = predictor.predict_batch(all_test_images)
        print(f"\n批量预测了 {len(results)} 张图片:")
        correct = 0
        for result in results:
            img_name = os.path.basename(result["path"])
            actual_color = result["path"].split("/")[-2]
            predicted_color = result["class"]
            confidence = result["confidence"]
            confidence = result['confidence']
            status = "✓" if predicted_color == actual_color else "✗"
            if predicted_color == actual_color:
                correct += 1
            print(f"{status} {img_name}: {predicted_color} ({confidence:.1f}%)")

        accuracy = 100.0 * correct / len(results)
        print(f"\n准确率: {accuracy:.1f}% ({correct}/{len(results)})")


if __name__ == "__main__":
    main()
