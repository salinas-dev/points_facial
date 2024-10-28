import re
import io
import cv2
import pandas as pd
import base64
from flask import Flask, request, render_template, redirect
import mediapipe as mp
from PIL import Image
import numpy as np

app = Flask(__name__)

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def draw_cross(image, center, size=5, color=(0, 0, 255)):
    """Dibuja una cruz en la imagen en la posición especificada."""
    x, y = center
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 2)
    cv2.line(image, (x + size, y - size), (x - size, y + size), color, 2)

def process_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    facial_points_dict = {  # Diccionario de puntos clave
        'left_eye_center_x': [None], 'left_eye_center_y': [None],
        'right_eye_center_x': [None], 'right_eye_center_y': [None],
        'nose_tip_x': [None], 'nose_tip_y': [None],
        'mouth_left_corner_x': [None], 'mouth_left_corner_y': [None],
        'mouth_right_corner_x': [None], 'mouth_right_corner_y': [None],
        'brow_left_x': [None], 'brow_left_y': [None],
        'brow_right_x': [None], 'brow_right_y': [None],
        'chin_x': [None], 'chin_y': [None],
        'left_cheek_x': [None], 'left_cheek_y': [None],
        'right_cheek_x': [None], 'right_cheek_y': [None],
        'left_ear_x': [None], 'left_ear_y': [None],
        'right_ear_x': [None], 'right_ear_y': [None],
        'mouth_center_x': [None], 'mouth_center_y': [None],
    }

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_mapping = {
                33: 'left_eye_center', 263: 'right_eye_center',
                1: 'nose_tip', 61: 'mouth_left_corner', 291: 'mouth_right_corner',
                19: 'brow_left', 24: 'brow_right', 152: 'chin',
                234: 'left_cheek', 454: 'right_cheek',
                174: 'left_ear', 454: 'right_ear', # Es necesario cambiar el índice a uno diferente para el derecho.
                0: 'mouth_center'  # Punto central de la boca
            }
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                if idx in landmarks_mapping:
                    key_x = f"{landmarks_mapping[idx]}_x"
                    key_y = f"{landmarks_mapping[idx]}_y"
                    facial_points_dict[key_x][0] = x
                    facial_points_dict[key_y][0] = y
                    draw_cross(image, (x, y))

    return pd.DataFrame(facial_points_dict), image

def encode_image_to_base64(image):
    """Codifica la imagen a Base64 para mostrarla en HTML."""
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = io.BytesIO(buffer)
    base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return base64_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            np_img = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Procesar la imagen
            df_facial_points, processed_image = process_image(image)

            # Codificar la imagen procesada a Base64
            img_data = encode_image_to_base64(processed_image)

            # Convertir el dataframe a CSV en memoria
            csv_data = df_facial_points.to_csv(index=False)

            return render_template('result.html', img_data=img_data, csv_data=csv_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
