from flask import Flask, render_template, request, Response, redirect, url_for
from stream_manager import VideoStreamProcessor
import uuid
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'
app.config['VIDEO_UPLOADS'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_UPLOADS'], exist_ok=True)

streams = {}
image_result = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", stream_ids=streams.keys(), image_result=image_result)

# Mở webcam
@app.route("/add_stream_webcam", methods=["POST"])
def add_stream_webcam():
    model_type = request.form.get("model_type", "yolo")
    stream_id = str(uuid.uuid4())[:8]
    streams[stream_id] = VideoStreamProcessor(0, model_type)  # Webcam = 0
    return redirect(url_for("index"))

# Upload video
@app.route("/add_stream_video", methods=["POST"])
def add_stream_video():
    file = request.files.get("video_file")
    model_type = request.form.get("model_type", "yolo")
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        save_path = os.path.join(app.config['VIDEO_UPLOADS'], filename)
        file.save(save_path)

        stream_id = str(uuid.uuid4())[:8]
        streams[stream_id] = VideoStreamProcessor(save_path, model_type)
    return redirect(url_for("index"))

@app.route("/video_feed/<stream_id>")
def video_feed(stream_id):
    def generate(stream):
        while True:
            frame = stream.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(streams[stream_id]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detect_image", methods=["POST"])
def detect_image():
    global image_result
    model_type = request.form.get("model_type")
    file = request.files['image']
    filename = str(uuid.uuid4()) + ".jpg"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    if model_type == "yolo":
        model = YOLO("best.pt")
    else:
        model = YOLO("yolov8n.pt")  # Placeholder cho model khác
    results = model(path)
    output_path = path.replace(".jpg", "_result.jpg")
    results[0].save(filename=output_path)
    image_result = '/' + output_path
    return redirect(url_for("index"))

@app.route("/shutdown", methods=["POST"])
def shutdown():
    for stream in streams.values():
        stream.stop()
    streams.clear()
    return "Stopped all streams"

if __name__ == "__main__":
    app.run(debug=True)
