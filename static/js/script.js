document.addEventListener('DOMContentLoaded', () => {
    const webcam = document.getElementById('webcam');
    const previewImage = document.getElementById('preview-image');
    const previewVideo = document.getElementById('preview-video');
    const resultImage = document.getElementById('result-image');
    const resultVideo = document.getElementById('result-video');
    const cameraSelect = document.getElementById('camera-select');
    const startWebcam = document.getElementById('start-webcam');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const modelSelect = document.getElementById('model-select');
    const confidenceSlider = document.getElementById('confidence');
    const confidenceValue = document.getElementById('confidence-value');
    const status = document.getElementById('status');
    let stream = null;

    // Cập nhật giá trị confidence
    confidenceSlider.addEventListener('input', () => {
        confidenceValue.textContent = confidenceSlider.value;
    });

    // Lấy danh sách camera
    navigator.mediaDevices.enumerateDevices().then(devices => {
        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${cameraSelect.length + 1}`;
                cameraSelect.appendChild(option);
            }
        });
    });

    // Bắt đầu webcam
    startWebcam.addEventListener('click', async () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        const deviceId = cameraSelect.value;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: deviceId ? { exact: deviceId } : undefined }
            });
            webcam.srcObject = stream;
            webcam.style.display = 'block';
            previewImage.style.display = 'none';
            previewVideo.style.display = 'none';
            status.textContent = 'Webcam started';
        } catch (err) {
            status.textContent = `Error starting webcam: ${err.message}`;
        }
    });

    // Xử lý upload
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            status.textContent = 'Please select a file';
            return;
        }

        // Hiển thị preview
        const fileURL = URL.createObjectURL(file);
        if (file.type.startsWith('image/')) {
            previewImage.src = fileURL;
            previewImage.style.display = 'block';
            previewVideo.style.display = 'none';
            webcam.style.display = 'none';
            resultImage.style.display = 'none';
            resultVideo.style.display = 'none';
        } else if (file.type.startsWith('video/')) {
            previewVideo.src = fileURL;
            previewVideo.style.display = 'block';
            previewImage.style.display = 'none';
            webcam.style.display = 'none';
            resultImage.style.display = 'none';
            resultVideo.style.display = 'none';
        }

        // Gửi yêu cầu đến server
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelSelect.value);
        formData.append('confidence', confidenceSlider.value);

        status.textContent = 'Processing...';
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const resultURL = await response.text();
                if (file.type.startsWith('image/')) {
                    resultImage.src = resultURL;
                    resultImage.style.display = 'block';
                    resultVideo.style.display = 'none';
                } else {
                    resultVideo.src = resultURL;
                    resultVideo.style.display = 'block';
                    resultImage.style.display = 'none';
                    resultVideo.load(); // Tải lại video để đảm bảo hiển thị
                }
                status.textContent = 'Completed';
            } else {
                const errorText = await response.text();
                status.textContent = `Error: ${errorText}`;
            }
        } catch (err) {
            status.textContent = `Error: ${err.message}`;
        }
    });
});