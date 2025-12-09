import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model_to_int8(input_model_path, output_model_path):
    print(f"Đang xử lý: {input_model_path} ...")

    # Sử dụng Dynamic Quantization (Lượng tử hóa động)
    # Đây là phương pháp nhanh nhất, không cần tập dữ liệu mẫu (Calibration data)
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        # Chuyển trọng số sang số nguyên 8-bit không dấu (QUInt8)
        weight_type=QuantType.QUInt8
    )

    print(f"Hoàn tất! Model đã được lưu tại: {output_model_path}")

    # So sánh kích thước
    import os
    size_old = os.path.getsize(input_model_path) / (1024 * 1024)
    size_new = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"Kích thước cũ (FP32): {size_old:.2f} MB")
    print(f"Kích thước mới (INT8): {size_new:.2f} MB")
    print(f"Đã giảm: {(1 - size_new / size_old) * 100:.2f}%")


if __name__ == "__main__":
    input_path = "chinese_chess_model.onnx"
    output_path = "chinese_chess_model_int8.onnx"

    # Kiểm tra file tồn tại
    import os

    if os.path.exists(input_path):
        quantize_model_to_int8(input_path, output_path)
    else:
        print(f"Lỗi: Không tìm thấy file {input_path}")