class SmileDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Инициализация детекторов
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.7
        )
        
    def detect_smile(self, image_path):
        """
        Детектирует улыбку на изображении
        Возвращает True если улыбка обнаружена, False в противном случае
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            return False
            
        # Конвертация BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        try:
            # Детекция лиц
            face_results = self.face_detection.process(image_rgb)
            
            if not face_results.detections:
                return False
                
            # Используем Face Mesh для более точного определения улыбки
            mesh_results = self.face_mesh.process(image_rgb)
            
            if not mesh_results.multi_face_landmarks:
                return False
                
            # Анализируем положение губ для определения улыбки
            for face_landmarks in mesh_results.multi_face_landmarks:
                if self._is_smiling(face_landmarks.landmark):
                    return True
                    
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            
        return False
    
    def _is_smiling(self, landmarks):
        """
        Определяет улыбку по landmarks лица
        Использует соотношение ширины рта к расстоянию между глазами
        """
        # Индексы ключевых точек для губ (MediaPipe Face Mesh)
        # Внешние уголки губ
        left_lip_corner = 61
        right_lip_corner = 291
        
        # Центральные точки верхней и нижней губы
        upper_lip_center = 13
        lower_lip_center = 14
        
        # Получаем координаты точек
        left_x = landmarks[left_lip_corner].x
        right_x = landmarks[right_lip_corner].x
        upper_y = landmarks[upper_lip_center].y
        lower_y = landmarks[lower_lip_center].y
        
        # Вычисляем ширину рта и высоту открытия
        mouth_width = abs(right_x - left_x)
        mouth_height = abs(lower_y - upper_y)
        
        # Вычисляем соотношение ширины к высоте
        # При улыбке ширина рта увеличивается относительно высоты
        mouth_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        print(f"  Mouth ratio: {mouth_ratio:.2f}")
        # Эмпирически подобранный порог для определения улыбки
        smile_threshold_min = 2
        smile_threshold_max = 20
        
        return mouth_ratio > smile_threshold_min and mouth_ratio < smile_threshold_max
    
    def process_directory(self, input_dir, output_dir):
        """
        Обрабатывает все изображения в каталоге
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Создаем выходной каталог если не существует
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поддерживаемые форматы изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Счетчики
        total_images = 0
        smile_images = 0
        
        print(f"Начинаем обработку каталога: {input_dir}")
        
        # Обрабатываем все файлы в каталоге
        for image_file in input_path.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                total_images += 1
                print(f"Обрабатывается: {image_file.name}")
                
                if self.detect_smile(str(image_file)):
                    smile_images += 1
                    # Копируем изображение с улыбкой в выходной каталог
                    output_file = output_path / image_file.name
                    shutil.copy2(image_file, output_file)
                    print(f"  ✓ Улыбка обнаружена, копируем: {image_file.name}")
                else:
                    print(f"  ✗ Улыбка не обнаружена: {image_file.name}")
        
        print(f"\nОбработка завершена!")
        print(f"Всего обработано изображений: {total_images}")
        print(f"Найдено улыбающихся: {smile_images}")
        print(f"Скопировано в: {output_dir}")