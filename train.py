import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Аугментация цвета
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Кастомный Dataset для загрузки изображений и их метаданных из CSV, модель обучалась на подготовленных данных по изображениям
class ClothingDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file, sep=';')
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        label_gender = 1 if self.data.iloc[idx, 1] == 'male' else 0
        label_strict = self.data.iloc[idx, 3]
        label_kind = self.data.iloc[idx, 4]
        label_gentle = self.data.iloc[idx, 5]
        label_attention = self.data.iloc[idx, 6]
        label_optimist = self.data.iloc[idx, 7]

        # Собираем все метки в один массив
        labels = torch.tensor([label_gender, label_strict, label_kind, label_gentle, label_attention, label_optimist], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Пути к данным
data_dir = './images'
csv_file = './data/descriptions.csv'

# Создаем экземпляры Dataset и DataLoader
train_dataset = ClothingDataset(csv_file=csv_file, image_folder=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)

# Используем предобученную модель ResNet18 и изменяем выходной слой для многозадачной классификации
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

# Определяем функцию потерь и оптимизатор
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Тренировка модели
num_epochs = 10
model = model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Сохранение обученной модели
torch.save(model.state_dict(), 'models/clothing_model.pth')
