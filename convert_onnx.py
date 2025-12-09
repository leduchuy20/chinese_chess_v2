import tensorflow as tf
import tf2onnx
import os
from CNN_Classification_Model.model import cnn_model

# 1. Load model Keras giống hệt như trong code cũ
print("Đang load model Keras...")
model = cnn_model()
try:
    # Thử load file .keras trước
    model.load_weights('./h5_file/new_model_v2.keras')
except:
    # Nếu lỗi thì thử load .h5 (phòng hờ)
    model.load_weights('./h5_file/new_model_v2.h5')

# 2. Định nghĩa input signature (Kích thước ảnh đầu vào là 56x56, 3 kênh màu)
# Batch size để là None (linh động) hoặc 1
input_signature = [tf.TensorSpec([None, 56, 56, 3], tf.float32, name='input_image')]

# 3. Chuyển đổi sang ONNX
output_path = "chinese_chess_model.onnx"
print(f"Đang chuyển đổi sang {output_path}...")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=output_path
)

print("Chuyển đổi thành công! File đã lưu tại:", output_path)