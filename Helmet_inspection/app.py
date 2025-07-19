from flask import Flask, request, render_template, send_from_directory
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO('yolov8_model/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        results = model(input_path)
        result_image = results[0].plot()
        output_filename = 'result_' + filename
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        cv2.imwrite(output_path, result_image)

        return render_template('index.html',
                               original_file=filename,
                               result_file=output_filename)

    elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = 'result_' + filename
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return render_template('index.html',
                               original_file=filename,
                               result_file=output_filename)
    else:
        return 'Unsupported file type', 400

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
