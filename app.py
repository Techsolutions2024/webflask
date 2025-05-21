from flask import Flask, request, send_file, Response, render_template
from ultralytics import YOLO
import cv2
import os
import numpy as np
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def resize_frame(frame, max_size=(1280, 720)):
    """Resize frame giữ tỷ lệ khung hình, tối đa max_size."""
    h, w = frame.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

@app.route('/')
def index():
    logger.debug("Rendering index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return Response('No file part', status=400)
    file = request.files['file']
    if not file or file.filename == '':
        logger.error("No selected file")
        return Response('No selected file', status=400)
    
    model_name = request.form.get('model', 'yolov8n')
    confidence = float(request.form.get('confidence', 0.5))
    
    # Kiểm tra định dạng tệp
    if not (allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS) or 
            allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)):
        logger.error(f"Unsupported file format: {file.filename}")
        return Response('Unsupported file format', status=400)
    
    # Lưu tệp
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logger.debug(f"File saved: {file_path}")
    
    try:
        # Chạy inference
        model = models.get(model_name, models['yolov8n'])
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        
        if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
            logger.debug(f"Processing image: {filename}")
            img = cv2.imread(file_path)
            img = resize_frame(img, max_size=(1280, 720))  # Resize ảnh nếu cần
            results = model(img, conf=confidence)
            for r in results:
                im_array = r.plot()  # Vẽ bounding box
                cv2.imwrite(result_path, im_array)
                logger.debug(f"Image result saved: {result_path}")
        else:
            logger.debug(f"Processing video: {filename}")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {file_path}")
                return Response('Cannot open video', status=500)
            
            # Lấy thông tin video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            # Resize kích thước video nếu cần
            if width > 1280 or height > 720:
                width, height = min(1280, width), min(720, height)
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error(f"Cannot create output video: {result_path}")
                cap.release()
                return Response('Cannot create output video', status=500)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = resize_frame(frame, max_size=(1280, 720))  # Resize frame
                results = model(frame, conf=confidence)
                result_frame = results[0].plot()
                out.write(result_frame)
            
            cap.release()
            out.release()
            logger.debug(f"Video result saved: {result_path}")
        
        # Trả về URL của tệp kết quả
        result_url = f'/static/uploads/result_{filename}'
        logger.debug(f"Returning result URL: {result_url}")
        return result_url
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return Response(f'Error processing file: {str(e)}', status=500)

if __name__ == '__main__':
    app.run(debug=True)