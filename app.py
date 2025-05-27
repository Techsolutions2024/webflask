from flask import Flask, request, render_template, send_file
from diffusers import StableDiffusionXLPipeline
import torch
import uuid
import os

app = Flask(__name__)

print("🔄 Đang load model SDXL và LoRA...")

# ✅ Load model và LoRA chỉ một lần
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe.load_lora_weights("./corgy_dog_LoRA")

print("✅ Model và LoRA đã sẵn sàng!")

# ✅ Thư mục lưu ảnh tạm
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        image = pipe(prompt).images[0]
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        image_path = f"/image/{filename}"
    return render_template("index.html", image_path=image_path)

@app.route("/image/<filename>")
def image(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
