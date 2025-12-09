import os
import time
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort

# ==========================
# CẤU HÌNH
# ==========================

LABELS: List[str] = [
    'b_jiang', 'b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
    'grid',
    'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang'
]

MODEL_PATHS: Dict[str, str] = {
    "fp32": "chinese_chess_model.onnx",
    "int8": "chinese_chess_model_int8.onnx",
}

DATASET_ROOT = "Dataset"  # chứa train/ và valid/
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ==========================
# HÀM TIỆN ÍCH
# ==========================

def analyze_input_layout(input_shape):
    """
    Phân tích shape input ONNX để suy ra:
      - layout: "NCHW" hoặc "NHWC"
      - H, W, C: height, width, channels (int)
    """
    # chuyển symbolic dims (None, 'batch', ...) -> None
    dims = [d if isinstance(d, int) else None for d in input_shape]

    layout = "NCHW"
    H = W = 56
    C = 3

    if len(dims) == 4:
        # TH1: NCHW: [N, C, H, W]
        if dims[1] in (1, 3) and dims[2] is not None and dims[3] is not None:
            layout = "NCHW"
            C = dims[1]
            H = dims[2]
            W = dims[3]
        # TH2: NHWC: [N, H, W, C]
        elif dims[3] in (1, 3) and dims[1] is not None and dims[2] is not None:
            layout = "NHWC"
            H = dims[1]
            W = dims[2]
            C = dims[3]
        else:
            # fallback (khó đoán): giữ default
            pass
    # in ra cho chắc chắn
    # print(f"[DEBUG] input_shape={input_shape}, layout={layout}, H={H}, W={W}, C={C}")
    return layout, H, W, C


def get_model_file_size_mb(path: str) -> float:
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def get_model_num_params(path: str) -> Optional[int]:
    """
    Đếm tổng số phần tử trong các initializer (trọng số) của model ONNX.
    Nếu lỗi thì trả về None.
    """
    try:
        model = onnx.load(path)
        total_params = 0
        for initializer in model.graph.initializer:
            n = 1
            for d in initializer.dims:
                n *= d
            total_params += n
        return total_params
    except Exception as e:
        print(f"[WARN] Không đếm được số param cho {path}: {e}")
        return None


def infer_input_hw(input_shape) -> Tuple[int, int]:
    """
    Suy ra H, W từ shape của input ONNX.
    Thường là [N, C, H, W] hoặc [N, H, W, C].
    """
    dims = [d for d in input_shape if isinstance(d, int)]
    if len(dims) >= 2:
        H, W = dims[-2], dims[-1]
    else:
        # fallback nếu không có thông tin
        H, W = 224, 224
    return H, W


def load_and_preprocess_image(img_path: str, input_shape) -> np.ndarray:
    """
    Đọc ảnh ở img_path, resize theo H,W từ input_shape,
    chuẩn hóa [0,1] và sắp xếp tensor đúng layout (NCHW hoặc NHWC).
    """
    layout, H, W, C = analyze_input_layout(input_shape)

    # Đọc ảnh với mọi kích thước, mọi mode, đưa về RGB
    img = Image.open(img_path).convert("RGB")

    # Resize về đúng kích thước mà model yêu cầu (VD: 56x56)
    img = img.resize((W, H))  # PIL nhận (width, height)

    img_arr = np.array(img).astype("float32") / 255.0  # [H, W, 3]

    if layout == "NCHW":
        # [H,W,3] -> [3,H,W] -> [1,3,H,W]
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_arr = np.expand_dims(img_arr, axis=0)
    else:  # NHWC
        # [H,W,3] -> [1,H,W,3]
        img_arr = np.expand_dims(img_arr, axis=0)

    return img_arr


def evaluate_model_on_split(
    session: ort.InferenceSession,
    split_dir: str,
    labels: List[str],
    warmup: int = 5
) -> Tuple[float, float, int]:
    """
    Đánh giá 1 model trên 1 tập (train hoặc valid).

    Trả về:
        accuracy (0-1),
        average_latency_ms (ms / mẫu),
        total_samples (số lượng ảnh được đánh giá)
    """
    if not os.path.isdir(split_dir):
        print(f"[WARN] Không tìm thấy thư mục {split_dir}")
        return 0.0, 0.0, 0

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    total = 0
    correct = 0
    latencies_ms: List[float] = []

    # Phân tích shape & layout input
    layout, H, W, C = analyze_input_layout(input_shape)

    # Warmup một vài lần với ảnh giả để "hâm nóng" session
    if layout == "NCHW":
        dummy_input = np.random.rand(1, C, H, W).astype("float32")
    else:  # NHWC
        dummy_input = np.random.rand(1, H, W, C).astype("float32")

    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in label_to_idx:
            print(f"[WARN] Nhãn {class_name} không nằm trong LABELS, bỏ qua.")
            continue

        true_idx = label_to_idx[class_name]

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(class_dir, fname)
            try:
                x = load_and_preprocess_image(img_path, input_shape)
            except Exception as e:
                print(f"[WARN] Lỗi đọc/tiền xử lý {img_path}: {e}")
                continue

            start = time.perf_counter()
            outputs = session.run(None, {input_name: x})
            end = time.perf_counter()

            latency_ms = (end - start) * 1000.0
            latencies_ms.append(latency_ms)

            logits = outputs[0]
            pred_idx = int(np.argmax(logits, axis=1)[0])

            if pred_idx == true_idx:
                correct += 1
            total += 1

    if total == 0:
        return 0.0, 0.0, 0

    accuracy = correct / total
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return accuracy, avg_latency_ms, total


# ==========================
# MAIN
# ==========================

def main():
    print("===== THÔNG TIN MÔ HÌNH =====")
    model_info = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.isfile(path):
            print(f"[ERROR] Không tìm thấy file model: {path}")
            continue

        size_mb = get_model_file_size_mb(path)
        num_params = get_model_num_params(path)
        model_info[name] = {
            "path": path,
            "size_mb": size_mb,
            "num_params": num_params,
        }
        print(f"- {name}: {path}")
        print(f"  + Kích thước file: {size_mb:.2f} MB")
        if num_params is not None:
            print(f"  + Số lượng tham số: {num_params:,}")
        else:
            print(f"  + Số lượng tham số: (không đọc được)")

    print("\n===== ĐÁNH GIÁ ĐỘ CHÍNH XÁC & ĐỘ TRỄ =====")

    for name, path in MODEL_PATHS.items():
        if not os.path.isfile(path):
            continue

        print(f"\n--- Model: {name} ({path}) ---")

        # Tạo session ONNX Runtime (CPU). Nếu dùng GPU, đổi providers.
        session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
        )

        for split in ["train", "valid"]:
            split_dir = os.path.join(DATASET_ROOT, split)
            acc, avg_latency_ms, n_samples = evaluate_model_on_split(
                session, split_dir, LABELS
            )

            print(f"[{split.upper()}]")
            print(f"  Số mẫu: {n_samples}")
            print(f"  Accuracy: {acc * 100:.2f}%")
            print(f"  Độ trễ trung bình: {avg_latency_ms:.3f} ms/mẫu")


if __name__ == "__main__":
    main()
