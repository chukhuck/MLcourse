import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_opencv(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Конвертация в HSV для цветовой сегментации
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Пример: сегментация по цвету (настройте диапазоны под вашу задачу)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Создание маски
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Применение маски к оригинальному изображению
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented)
    plt.title('Segmented Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# segment_opencv('path/to/your/image.jpg')