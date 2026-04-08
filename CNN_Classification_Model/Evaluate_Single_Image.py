import argparse
import os
import sys

import cv2

# Ensure project root is on sys.path when running this file directly.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from piece_detector import PieceDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one image with YOLO weights.pt")
    parser.add_argument("--weights", default="./weights.pt", help="Path to YOLO .pt model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Khong tim thay anh: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {args.image}")

    detector = PieceDetector(
        weights_path=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    label = detector.predict_piece(image, default_label="grid")
    print(f"Predicted label: {label}")


if __name__ == "__main__":
    main()
