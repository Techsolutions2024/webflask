from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8n.pt")

def detect_with_yolo(input_path):
    result = model(input_path)
    output_path = input_path.replace(".", "_result.")

    if input_path.endswith(('.jpg', '.png')):
        result[0].save(filename=output_path)
        return output_path, True
    else:
        result[0].save(filename=output_path)  # Save video output
        return output_path, False
