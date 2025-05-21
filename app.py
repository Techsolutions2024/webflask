from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # hoặc 'best.pt' của bạn

video_source = 0  # Default webcam

def generate_frames():
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8 detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_source
    stream = False
    if request.method == 'POST':
        selected_source = request.form.get('source')
        if selected_source == 'webcam':
            video_source = 0
        elif selected_source == 'video':
            video_file = request.files.get('video_file')
            if video_file:
                filepath = os.path.join("temp_video.mp4")
                video_file.save(filepath)
                video_source = filepath
        stream = True

    return render_template('index.html', stream=stream)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
