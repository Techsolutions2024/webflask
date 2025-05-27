from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from io import BytesIO
from PyQt5.QtGui import QPixmap, QImage
import sys

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Diffusion GUI (SDXL + PyQt5)")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.prompt_input = QTextEdit("A corgi dog in a field of flowers")
        self.generate_button = QPushButton("Generate Image")
        self.image_label = QLabel("Image will appear here")

        layout.addWidget(self.prompt_input)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # ✅ Load model 1 lần duy nhất khi app khởi động
        print("🔄 Đang load model SDXL...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        self.pipe.load_lora_weights("./corgy_dog_LoRA")
        print("✅ Model đã sẵn sàng!")

        self.generate_button.clicked.connect(self.generate_image)

    def generate_image(self):
        prompt = self.prompt_input.toPlainText()
        print(f"🔮 Generating: {prompt}")
        image = self.pipe(prompt).images[0]

        # Hiển thị ảnh lên QLabel
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(512, 512))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
