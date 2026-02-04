from flask import Flask, render_template, request, session
import os
import cv2
from ultralytics import YOLO
import uuid
import numpy as np

app = Flask(__name__)
app.secret_key = 'calvin-druggiee'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


window_model = YOLO('window_model.pt')
fan_model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    results_text = []
    dimension =[]
    air_flow=0

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_folder = os.path.join(UPLOAD_FOLDER, session['session_id'])
    os.makedirs(session_folder, exist_ok=True)

    if request.method == 'POST':
        files = []
        dimensions = []
        

        
        for i in range(1, 5):
            uploaded_file = request.files.get(f'file{i}')
            width = request.form.get(f'width{i}')
            height = request.form.get(f'height{i}')

            if not (uploaded_file and width and height):
                results_text.append("❌ Please provide all 5 images with their corresponding dimensions.")
                return render_template('index.html', results=results_text)

            files.append((uploaded_file, int(width), int(height)))

        # Get 5th image (fan detection)
        fan_file = request.files.get('file5')
        if not fan_file:
            results_text.append("❌ Please provide all 5 images with their corresponding dimensions.")
            return render_template('index.html', results=results_text)

        # Process first 4 images with window_model
        for i, (img_file, w_input, h_input) in enumerate(files):
            filename = f"file{i+1}.jpg"
            filepath = os.path.join(session_folder, filename)
            img_file.save(filepath)

            img = cv2.imread(filepath)
            h_img, w_img = img.shape[:2]
            total_area = h_img * w_img
            total_area_wall = w_input * h_input
            if w_input not in dimension:
                dimensions.append(w_input)
            h= h_input
            

            result = window_model(filepath)[0]
            names = result.names
            window_area = 0

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])
                if "window" in label.lower() and conf > 0.2:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width_box = x2 - x1
                    height_box = y2 - y1
                    area = width_box * height_box
                    window_area += area
            percentage = (window_area / total_area) * 100 if total_area else 0
            area_window = ((percentage * total_area_wall) / 100)*0.65
            air_flow+=area_window*2
            results_text.append(
                f"file{i+1}.jpg → Window area: {window_area}px² ({percentage:.2f}% area)"
            )


        if len(dimensions)==2:
            vol=dimensions[0]*dimensions[1]*h
        else:
            vol=dimensions[0]*dimensions[0]*h
        

        
        fan_filename = "file5.jpg"
        fan_filepath = os.path.join(session_folder, fan_filename)
        fan_file.save(fan_filepath)

        
        result = fan_model(fan_filepath)[0]
        num_fans = len(result.boxes)

        
        img = cv2.imread(fan_filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray)

        
        if average_intensity < 50:
            light_level = "Very Low"
        elif average_intensity < 100:
            light_level = "Low"
        elif average_intensity < 170:
            light_level = "Moderate"
        elif average_intensity < 220:
            light_level = "Good"
        else:
            light_level = "Very Good"

        num_fans=5
        results_text.append(f"file5.jpg → Number of fans detected: {num_fans}")
        results_text.append(f"Light Intensity: {light_level} ({average_intensity:.2f})")

        fan_airflow = 3
        total_airflow = air_flow + (num_fans * fan_airflow)

        ach = ((total_airflow * 3600) / vol)/100

        if ach <2:
            air_quality = "Very Poor"
        elif ach < 3:
            air_quality = "Poor"
        elif ach < 4:
            air_quality = "Moderate"
        elif ach < 5:
            air_quality = "Good"
        elif ach < 6:
            air_quality = "Very Good"
        else:
            air_quality = "Excellent"

        results_text.append(f"Air Flow Quality: {air_quality}")


    return render_template('index.html', results=results_text)

if __name__ == '__main__':
    app.run(debug=True)
