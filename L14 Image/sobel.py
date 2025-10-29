import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage
from skimage import filters

class Segmenter:
    def __init__(self):
        self.image = None
        self.mask = None
        self.filtered_image = None
    
    def load_image(self, image_path):
        """Загрузка изображения и конвертация в grayscale"""
        self.image = Image.open(image_path).convert('L')  # Конвертируем сразу в grayscale
        return np.array(self.image)
    
    def find_object_by_contrast(self, image_array):
        """
        Нахождение объекта используя детектор краев Sobel
        """
        gray = image_array
        
        # Нормализуем изображение
        gray = gray.astype(np.float32)       
        enhancer = ImageEnhance.Contrast(self.image)
        gray = np.array(enhancer.enhance(100))     
        gray = (gray - gray.min()) /(gray.max() - gray.min()) 
        
        # Адаптивный порог на комбинированной карте
        threshold = filters.threshold_otsu(gray)
        binary_mask = gray > threshold
        
        return binary_mask
    
    def apply_filters(self, image_array, mask):
        """Применение фильтров на основе маски"""                
        white_bg = np.ones_like(image_array) * 255
        result = np.where(mask == False, image_array, white_bg)
        return Image.fromarray(result)
    
    def process_image(self, image_path, show_results=True):
        """Полный процесс обработки изображения"""
        
        # Загрузка изображения
        image_array = self.load_image(image_path)
        
        # Нахождение объекта по контрасту
        self.mask = self.find_object_by_contrast(image_array)
        
        # Применение фильтров
        self.filtered_image = self.apply_filters(image_array, self.mask)
        
        # Показ результатов
        if show_results:
            self.display_results(image_array, self.mask, self.filtered_image)
        
        return self.filtered_image, self.mask
    
    def display_results(self, original, mask, filtered):
        """Отображение результатов"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Исходное изображение

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')
        
        # Маска
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Маска объекта2")
        axes[1].axis('off')
        
        # Результат
        axes[2].imshow(filtered, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title("Результат фильтрации1")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_path):
        """Сохранение результата"""
        if self.filtered_image is not None:
            self.filtered_image.save(output_path)
            print(f"Результат сохранен в: {output_path}")
        else:
            print("Нет результата для сохранения!")