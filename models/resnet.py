# Đây là mô phỏng, ResNet thường dùng cho phân loại ảnh
from PIL import Image
import os
import shutil

def detect_with_resnet(input_path):
    output_path = input_path.replace(".", "_classified.")
    shutil.copy(input_path, output_path)  # Chỉ copy lại làm ví dụ
    return output_path, input_path.endswith(('.jpg', '.png'))
