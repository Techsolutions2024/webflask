from flask import Flask, request, send_file, Response, render_template
from ultralytics import YOLO
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tải mô hình YOLOv8
models = {
    'yolov8n': YOLO('models/yolov8n.pt'),
    'yolov8s': YOLO('models/yolov8s.pt'),
    'yolov8m': YOLO('models/yolov8m.pt'),
    'yolov8l': YOLO('models/yolov8l.pt')
}

# Danh sách định dạng được hỗ trợ
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return Response('No file part', status=400)
    file = request.files['file']
    if not file or file.filename == '':
        return Response('No selected file', status=400)
    
    model_name = request.form.get('model', 'yolov8n')
    confidence = float(request.form.get('confidence', 0.5))
    
    # Kiểm tra định dạng tệp
    if not (allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS) or 
            allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)):
        return Response('Unsupported file format', status=400)
    
    # Lưu tệp
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Chạy inference
        model = models.get(model_name, models['yolov8n'])
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        
        if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
            # Xử lý ảnh
            results = model(file_path, conf=confidence)
            for r in results:
                im_array = r.plot()  # Vẽ bounding box
                cv2.imwrite(result_path, im_array)
        else:
            # Xử lý video
            cap = cv2.VideoCapture(file_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, 30.0, 
                                (int(cap.get(3)), int(cap.get(4))))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=confidence)
                result_frame = results[0].plot()
                out.write(result_frame)
            
            cap.release()
            out.release()
        
        # Trả về URL của tệp kết quả
        result_url = f'/static/uploads/result_{filename}'
        return result_url
    except Exception as e:
        return Response(f'Error processing file: {str(e)}', status=500)

if __name__ == '__main__':
    app.run(debug=True)