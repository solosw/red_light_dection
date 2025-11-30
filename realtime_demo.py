"""
闯红灯检测系统 - 实时显示模式
专门用于实时查看检测结果的脚本
"""

import argparse
import os
import sys
import time


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


def run_realtime_detection(
    input_video="input.mp4", detection_zone=None, polygon_points=None
):
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
        polygon_points=polygon_points,
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
  python realtime_demo.py                          # 必须输入4个点坐标
  python realtime_demo.py -i my_video.mp4         # 使用指定视频文件
  python realtime_demo.py --points 300 200 700 600 100 200 500 600  # 直接指定4个点
  python realtime_demo.py --zone 300 200 700 600   # 使用矩形检测区域

快捷键说明:
  空格键: 暂停/继续播放
  ↑/↓:   调整播放速度 (0.25x - 5.0x)
  s:     保存当前帧截图
  q:     退出程序
  ESC:   退出程序

4点坐标输入说明:
  • 按顺时针或逆时针顺序输入4个点
  • 每个点用 x y 格式输入
  • 4个点将围成一个多边形检测区域
  • 示例: (300,200) (700,200) (700,600) (300,600) 构成一个矩形
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
        "--points",
        nargs=8,
        type=int,
        metavar=("P1X", "P1Y", "P2X", "P2Y", "P3X", "P3Y", "P4X", "P4Y"),
        help="自定义多边形检测区域坐标，输入4个点 (x1 y1 x2 y2 x3 y3 x4 y4)",
    )
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="跳过依赖检查（不推荐）",
    )
    args = parser.parse_args()

    # 处理检测区域参数 - 4点坐标输入是必选项
    detection_zone = None
    polygon_points = None

    if args.points:
        # 使用命令行指定的4点定义多边形区域
        polygon_points = args.points
        print(f"\n✓ 使用命令行指定的4点检测区域")
        print(f"  4个点坐标: {polygon_points}")
    elif args.zone:
        # 使用命令行指定的矩形区域 (转换为4点)
        zone = args.zone
        x1, y1, x2, y2 = zone
        polygon_points = [x1, y1, x2, y1, x2, y2, x1, y2]  # 4个点构成矩形
        print(f"\n✓ 使用矩形区域，转换为4点: {polygon_points}")
    else:
        # 交互式要求用户输入4个点坐标
        print("\n" + "=" * 80)
        print("🎯 请输入4个点坐标定义检测区域")
        print("=" * 80)
        print("说明:")
        print("  • 按顺时针或逆时针顺序输入4个点")
        print("  • 每个点用 'x y' 格式输入（用空格分隔）")
        print("  • 4个点将围成一个多边形检测区域")
        print("\n示例:")
        print("  请输入第1个点: 300 200")
        print("  请输入第2个点: 700 200")
        print("  请输入第3个点: 700 600")
        print("  请输入第4个点: 300 600")
        print("\n" + "=" * 80)

        polygon_points = []
        point_names = ["第1个点", "第2个点", "第3个点", "第4个点"]

        for i, name in enumerate(point_names):
            while True:
                try:
                    coords = input(f"\n请输入{name}坐标 (x y): ").strip().split()
                    if len(coords) != 2:
                        print("  ⚠️ 错误：请输入两个数字，用空格分隔，如 '300 200'")
                        continue
                    x, y = int(coords[0]), int(coords[1])
                    polygon_points.extend([x, y])
                    print(f"  ✓ 已记录{name}: ({x}, {y})")
                    break
                except ValueError:
                    print("  ⚠️ 错误：请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n\n⚠️ 用户取消输入")
                    sys.exit(1)

        print("\n✓ 4个点坐标输入完成！")
        print(f"  坐标列表: {polygon_points}")

    # 保存多边形点到文件
    import json
    import os

    zone_config = {
        "polygon_points": polygon_points,
        "input_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs("configs", exist_ok=True)
    with open("configs/detection_zone.json", "w") as f:
        json.dump(zone_config, f, indent=2)
    print(f"\n✓ 检测区域配置已保存到: configs/detection_zone.json")

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

    success = run_realtime_detection(args.input, detection_zone, polygon_points)

    if success:
        print("\n" + "=" * 60)
        print("🎉 实时监控结束!")
        print("=" * 60)
    else:
        print("\n❌ 实时监控失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
