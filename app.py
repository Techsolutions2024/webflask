from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import uuid
import threading
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VIDEO_UPLOADS'] = 'static/videos'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_UPLOADS'], exist_ok=True)

streams = {}
image_result = None

class VideoStreamProcessor:
    def __init__(self, source, model_type="yolo", confidence=0.5):
        self.source = source
        self.model_type = model_type
        self.confidence = confidence
        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.frame = None

        if self.model_type == "yolo":
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO("yolov8n.pt")  # Placeholder for other models

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.confidence)
            annotated = results[0].plot()
            _, jpeg = cv2.imencode('.jpg', annotated)
            self.frame = jpeg.tobytes()

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

@app.route("/")
def index():
    return render_template("index.html", streams=streams, image_result=image_result)

@app.route("/detect_image", methods=["POST"])
def detect_image():
    global image_result
    model_type = request.form.get("model_type")
    confidence = float(request.form.get("confidence", 0.5))
    file = request.files['image']

    filename = str(uuid.uuid4()) + ".jpg"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    model = YOLO("best.pt") if model_type == "yolo" else YOLO("yolov8n.pt")
    results = model(path, conf=confidence)
    output_path = path.replace(".jpg", "_result.jpg")
    results[0].save(filename=output_path)
    image_result = '/' + output_path
    return redirect(url_for("index"))

@app.route("/add_stream_webcam", methods=["POST"])
def add_stream_webcam():
    model_type = request.form.get("model_type", "yolo")
    confidence = float(request.form.get("confidence", 0.5))
    stream_id = str(uuid.uuid4())[:8]
    streams[stream_id] = VideoStreamProcessor(0, model_type, confidence)
    return redirect(url_for("index"))

@app.route("/add_stream_video", methods=["POST"])
def add_stream_video():
    file = request.files.get("video_file")
    model_type = request.form.get("model_type", "yolo")
    confidence = float(request.form.get("confidence", 0.5))
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        save_path = os.path.join(app.config['VIDEO_UPLOADS'], filename)
        file.save(save_path)

        stream_id = str(uuid.uuid4())[:8]
        streams[stream_id] = VideoStreamProcessor(save_path, model_type, confidence)
    return redirect(url_for("index"))

@app.route("/video_feed/<stream_id>")
def video_feed(stream_id):
    def gen():
        while True:
            frame = streams[stream_id].get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stop_stream/<stream_id>")
def stop_stream(stream_id):
    if stream_id in streams:
        streams[stream_id].stop()
        del streams[stream_id]
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
