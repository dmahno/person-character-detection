import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify, send_from_directory

app = Flask(__name__)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

# Загрузка сохраненных весов модели
model_path = 'models/clothing_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Преобразования для новых изображений (без аугментации)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Функция для анализа изображения
def analyze_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Добавляем размер батча

    with torch.no_grad():
        outputs = model(image)

    # Используем сигмоиду для получения вероятностей
    probabilities = torch.sigmoid(outputs).squeeze().numpy()

    # Преобразование вероятностей в бинарные значения (если вероятность больше 0.5)
    characteristics = {
        'Пол': 'Мужчина' if probabilities[0] > 0.5 else 'Женщина',
        'Суровость': 'Да' if probabilities[1] > 0.5 else 'Нет',
        'Доброта': 'Да' if probabilities[2] > 0.5 else 'Нет',
        'Нежность': 'Да' if probabilities[3] > 0.5 else 'Нет',
        'Любит быть в центре внимания': 'Да' if probabilities[4] > 0.5 else 'Нет',
        'Оптимист': 'Да' if probabilities[5] > 0.5 else 'Нет'
    }

    return characteristics

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла в запросе'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Анализ изображения с помощью модели
    try:
        characteristics = analyze_image(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'characteristics': characteristics, 'image_path': f'/uploads/{file.filename}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
