import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_torchvision(image_path):
    # Загрузка модели
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    
    # Препроцессинг
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Загрузка и обработка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # Постобработка
    output_predictions = output.argmax(0)
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_predictions.byte().cpu().numpy())
    plt.title('Segmentation')
    plt.axis('off')
    plt.show()

# Использование
# segment_torchvision('path/to/your/image.jpg')