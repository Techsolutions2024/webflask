import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch

print("🔄 Đang load model SDXL và LoRA...")

# ✅ Load model chỉ MỘT LẦN khi app khởi động
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# ✅ Load local LoRA (1 lần)
pipe.load_lora_weights("./corgy_dog_LoRA")

print("✅ Model và LoRA đã sẵn sàng!")

# Hàm tạo ảnh
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🐶 Corgi LoRA Generator\nTạo ảnh từ LoRA SDXL!")
    with gr.Row():
        prompt = gr.Textbox(label="Nhập prompt", placeholder="a cute corgi dog wearing sunglasses")
        output = gr.Image(label="Ảnh tạo ra")
    generate_btn = gr.Button("🎨 Tạo ảnh")
    generate_btn.click(fn=generate_image, inputs=prompt, outputs=output)

# Khởi động app
demo.launch()
