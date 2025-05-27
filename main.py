from diffusers import StableDiffusionXLPipeline
import torch

# Load base SDXL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# Load local LoRA weights
pipe.load_lora_weights("./corgy_dog_LoRA")  # đường dẫn chứa safetensors LoRA bạn train

# Sinh ảnh
prompt = "a cute corgi dog wearing helmet"
image = pipe(prompt).images[0]

# Lưu ảnh
image.save("output.png")
print("Ảnh đã lưu: output.png")
