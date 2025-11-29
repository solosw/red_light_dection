"""
闯红灯检测系统 - 实时显示模式
专门用于实时查看检测结果的脚本
"""

import argparse
import os
import sys


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          闯红灯检测系统 - 实时监控模式 v1.0                  ║
║       Traffic Violation Detection - Realtime Mode          ║
║                                                              ║
║  🎯 功能特性:                                                ║
║    ✅ 实时视频流显示                                         ║
║    ✅ 即时检测和标注                                         ║
║    ✅ 交互式控制 (暂停/加速/截图)                            ║
║    ✅ 违规行为实时报警                                       ║
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


def run_realtime_detection(input_video="input.mp4", detection_zone=None):
    """运行实时检测"""
    print("\n" + "=" * 60)
    print("启动实时监控模式...")
    print("=" * 60)

    from traffic_violation_detector import TrafficViolationDetector

    # 创建检测器（开启实时显示）
    detector = TrafficViolationDetector(
        yolo_model_path="yolov8s.pt",
        classifier_model_path="models/traffic_light_classifier.pth",
        detection_zone=detection_zone,
        realtime_display=True,
        window_name="🚦 闯红灯检测系统 - 实时监控",
    )

    # 处理视频（不保存输出文件）
    try:
        detector.process_video(input_video, output_video=None)
        return True
    except Exception as e:
        print(f"\n✗ 检测过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="闯红灯检测系统 - 实时显示模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python realtime_demo.py                    # 使用默认 input.mp4
  python realtime_demo.py -i my_video.mp4   # 使用指定视频文件
  python realtime_demo.py --zone 300 200 700 600  # 自定义检测区域 (x1 y1 x2 y2)

快捷键说明:
  空格键: 暂停/继续播放
  ↑/↓:   调整播放速度 (0.25x - 5.0x)
  s:     保存当前帧截图
  q:     退出程序
  ESC:   退出程序
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        default="input.mp4",
        help="输入视频文件路径 (默认: input.mp4)",
    )
    parser.add_argument(
        "--zone",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="检测区域坐标 (x1 y1 x2 y2)，只有在此区域内的行人才会被判定为闯红灯",
    )
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="跳过依赖检查（不推荐）",
    )
    args = parser.parse_args()

    # 处理检测区域参数
    detection_zone = None
    if args.zone:
        detection_zone = tuple(args.zone)
        print(f"\n✓ 使用自定义检测区域: {detection_zone}")

    # 打印欢迎横幅
    print_banner()

    # 检查依赖
    if not args.skip_deps_check:
        if not check_dependencies():
            print("\n❌ 依赖检查失败，请解决问题后重试")
            sys.exit(1)

    # 检查输入视频
    if not os.path.exists(args.input):
        print(f"\n❌ 输入视频文件不存在: {args.input}")
        sys.exit(1)

    # 运行实时检测
    print("\n" + "=" * 80)
    print("准备启动实时监控...")
    print("=" * 80)
    print("⚠️  即将打开视频窗口，请确保您的系统支持图形界面")
    print("⚠️  使用 Ctrl+C 可以强制退出")
    print("=" * 80)

    success = run_realtime_detection(args.input, detection_zone)

    if success:
        print("\n" + "=" * 60)
        print("🎉 实时监控结束!")
        print("=" * 60)
    else:
        print("\n❌ 实时监控失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
