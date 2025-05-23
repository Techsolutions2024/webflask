import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Tải model YOLO
model = YOLO("s.pt")  # Đường dẫn đến model của bạn

# Hàm xử lý ảnh và hiển thị kết quả
def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Đọc ảnh
    image = cv2.imread(file_path)

    # Chạy model
    results = model(image)

    # Lấy ảnh đã vẽ kết quả
    annotated = results[0].plot()

    # Chuyển từ BGR sang RGB
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Resize cho vừa khung hiển thị (nếu cần)
    annotated_pil = Image.fromarray(annotated)
    annotated_pil.thumbnail((600, 600))

    # Hiển thị trên GUI
    img_tk = ImageTk.PhotoImage(annotated_pil)
    label.config(image=img_tk)
    label.image = img_tk  # giữ tham chiếu tránh bị thu hồi bộ nhớ

# Tạo GUI với Tkinter
root = tk.Tk()
root.title("YOLOv8 Image Detection")
root.geometry("700x700")

btn = tk.Button(root, text="Tải ảnh lên và detect", command=detect_image, font=("Arial", 14))
btn.pack(pady=20)

label = tk.Label(root)
label.pack()

root.mainloop()
