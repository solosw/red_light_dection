"""
闯红灯检测系统 - 完整使用脚本
一键运行所有功能：检测、跟踪、分析、可视化
"""

import argparse
import os
import sys
from datetime import datetime


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                闯红灯检测系统 v1.0                          ║
║          Traffic Violation Detection System                ║
║                                                              ║
║  🔍 功能特性:                                                ║
║    ✅ YOLOv8 目标检测 (车辆/行人/红绿灯)                    ║
║    🤖 红绿灯颜色分类 (红/绿/黄)                             ║
║    🔄 多目标跟踪 (无需训练)                                 ║
║    ⚠️ 闯红灯违规检测                                        ║
║    📊 详细分析和可视化报告                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies():
    """检查依赖"""
    print("\n" + "=" * 60)
    print("检查系统依赖")
    print("=" * 60)

    # 检查Python包
    required_packages = {
        "cv2": "opencv-python",
        "torch": "torch",
        "ultralytics": "ultralytics",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
    }

    missing_packages = []

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False

    # 检查模型文件
    print("\n检查模型文件...")
    models = {
        "yolov8s.pt": "YOLOv8检测模型",
        "models/traffic_light_classifier.pth": "红绿灯分类模型",
    }

    missing_models = []
    for model_file, description in models.items():
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ {description}: {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {description}: {model_file} - 未找到")
            missing_models.append(model_file)

    if missing_models:
        print(f"\n⚠️ 缺少模型文件: {', '.join(missing_models)}")
        print("请确保所有模型文件都在正确位置")
        return False

    # 检查输入视频
    print("\n检查输入文件...")
    input_video = "input.mp4"
    if os.path.exists(input_video):
        size_mb = os.path.getsize(input_video) / (1024 * 1024)
        print(f"✓ 输入视频: {input_video} ({size_mb:.1f} MB)")
    else:
        print(f"✗ 输入视频: {input_video} - 未找到")
        print(f"请将视频文件重命名为 '{input_video}' 并放在当前目录")
        return False

    print("\n✅ 所有依赖检查通过!")
    return True


def run_detection():
    """运行闯红灯检测"""
    print("\n" + "=" * 60)
    print("开始闯红灯检测...")
    print("=" * 60)

    from traffic_violation_detector import TrafficViolationDetector

    # 输入输出文件
    input_video = "input.mp4"
    output_video = "output_traffic_violation.mp4"

    # 创建检测器
    detector = TrafficViolationDetector(
        yolo_model_path="yolov8s.pt",
        classifier_model_path="models/traffic_light_classifier.pth",
    )

    # 处理视频
    try:
        detector.process_video(input_video, output_video)
        return True
    except Exception as e:
        print(f"\n✗ 检测过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False





def print_results_summary():
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("检测结果摘要")
    print("=" * 60)

    report_file = "traffic_violation_report.json"
    if not os.path.exists(report_file):
        print("未找到检测报告")
        return

    import json

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    stats = report["statistics"]
    violations = report["violations"]

    print(f"\n📹 视频处理:")
    print(f"   总帧数: {stats['total_frames']}")
    print(f"   处理时间: {report['processing_time']:.2f} 秒")

    print(f"\n🚗 检测统计:")
    print(f"   检测车辆次数: {stats['detected_vehicles']}")
    print(f"   检测红绿灯次数: {stats['detected_traffic_lights']}")

    print(f"\n🚦 红绿灯状态:")
    print(f"   红灯帧数: {stats['red_light_frames']}")
    print(f"   绿灯帧数: {stats['green_light_frames']}")
    print(f"   黄灯帧数: {stats['yellow_light_frames']}")

    print(f"\n⚠️ 违规统计:")
    print(f"   闯红灯违规总数: {stats['total_violations']}")

    if violations:
        print(f"\n违规详情:")
        for i, v in enumerate(violations, 1):
            timestamp = datetime.fromtimestamp(v["timestamp"])
            print(
                f"   {i}. 时间: {timestamp.strftime('%H:%M:%S')}, "
                f"车型: {v['vehicle_class']}, ID: {v['object_id']}"
            )
    else:
        print("   ✅ 未发现闯红灯违规行为")

    print(f"\n📁 输出文件:")
    print(f"   - 输出视频: output_traffic_violation.mp4")
    print(f"   - 检测报告: traffic_violation_report.json")
    print(f"   - 摘要报告: violation_summary.txt")
    print(f"   - 可视化图表: charts/ 目录")
    print(f"   - 标注视频: output_with_violation_annotations.mp4")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="闯红灯检测系统")
    parser.add_argument("--skip-deps-check", action="store_true", help="跳过依赖检查")
    parser.add_argument(
        "--skip-detection", action="store_true", help="跳过检测步骤（使用现有报告）"
    )
    parser.add_argument("--skip-viz", action="store_true", help="跳过可视化步骤")
    args = parser.parse_args()

    # 打印欢迎横幅
    print_banner()

    # 检查依赖
    if not args.skip_deps_check:
        if not check_dependencies():
            print("\n❌ 依赖检查失败，请解决问题后重试")
            sys.exit(1)

    # 运行检测
    if not args.skip_detection:
        success = run_detection()
        if not success:
            print("\n❌ 检测失败")
            sys.exit(1)
    else:
        print("\n⏭️ 跳过检测步骤")

    # 运行可视化
    if not args.skip_viz:

        if not success:
            print("\n❌ 可视化失败")
            sys.exit(1)
    else:
        print("\n⏭️ 跳过可视化步骤")

    # 打印结果摘要
    print_results_summary()

    print("\n" + "=" * 60)
    print("🎉 所有任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
