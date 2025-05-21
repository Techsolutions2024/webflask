import cv2
import threading
from ultralytics import YOLO

class VideoStreamProcessor:
    def __init__(self, source, model_type="yolo"):
        self.source = source
        self.model_type = model_type
        self.model = self.load_model(model_type)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def load_model(self, model_type):
        if model_type == "yolo":
            return YOLO("yolov8n.pt")
        else:
            # model khác: tạm return YOLO cho demo
            return YOLO("yolov8n.pt")

    def update(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            success, frame = cap.read()
            if not success:
                break
            results = self.model(frame)
            annotated = results[0].plot()
            with self.lock:
                self.frame = annotated
        cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                _, buffer = cv2.imencode('.jpg', self.frame)
                return buffer.tobytes()
            return None

    def stop(self):
        self.running = False
