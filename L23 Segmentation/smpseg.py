import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_smp(image_path):
    # Определение модели
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=21,  # количество классов (например, для Pascal VOC)
        activation=None,
    )
    
    # Загрузка весов (нужно предварительно обучить или загрузить предобученные)
    # model.load_state_dict(torch.load('path/to/weights.pth'))
    model.eval()
    
    # Препроцессинг
    def preprocess(image):
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return torch.tensor(image).unsqueeze(0)
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Предсказание
    with torch.no_grad():
        input_tensor = preprocess(image)
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Постобработка
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Segmentation Mask')
    plt.axis('off')
    plt.show()

# segment_smp('path/to/your/image.jpg')