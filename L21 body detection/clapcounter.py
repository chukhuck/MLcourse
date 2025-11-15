import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class ClapCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Счетчик хлопков
        self.clap_count = 0
        self.last_clap_frame = 0
        self.clap_cooldown = 30  # минимальное количество кадров между хлопками
        
        # Для отслеживания движения рук
        self.hand_distances = deque(maxlen=5)
        self.above_head_frames = 0
        
        # Пороговые значения
        self.CLAP_DISTANCE_THRESHOLD = 0.1  # минимальное расстояние для хлопка
        self.HEAD_THRESHOLD = 0.8  # порог для положения "над головой"
        self.MIN_ABOVE_HEAD_FRAMES = 3  # минимальное время над головой
        
    def calculate_distance(self, point1, point2):
        """Вычисляет расстояние между двумя точками"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_above_head(self, wrist, nose, shoulder):
        """Проверяет, находится ли запястье над головой"""
        return wrist.y < nose.y and wrist.y < shoulder.y
    
    def detect_clap(self, landmarks, frame_count):
        """Обнаруживает хлопок над головой"""
        if not landmarks:
            return False
        
        # Ключевые точки для анализа
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Проверяем, что обе руки над головой
        left_above = self.is_above_head(left_wrist, nose, left_shoulder)
        right_above = self.is_above_head(right_wrist, nose, right_shoulder)
        
        if left_above and right_above:
            self.above_head_frames += 1
        else:
            self.above_head_frames = 0
        
        # Вычисляем расстояние между руками
        hand_distance = self.calculate_distance(left_wrist, right_wrist)
        self.hand_distances.append(hand_distance)
        
        # Проверяем условия для хлопка
        if (self.above_head_frames >= self.MIN_ABOVE_HEAD_FRAMES and
            hand_distance < self.CLAP_DISTANCE_THRESHOLD and
            frame_count - self.last_clap_frame > self.clap_cooldown and
            len(self.hand_distances) >= 3):
            
            # Проверяем, что было движение сближения рук
            prev_distance = list(self.hand_distances)[-3]
            if hand_distance < prev_distance * 0.7:  # руки значительно сблизились
                self.clap_count += 1
                self.last_clap_frame = frame_count
                return True
        
        return False
    
    def process_video(self, video_path, output_path=None):
        """Обрабатывает видео и подсчитывает хлопки"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео {video_path}")
            return
        
        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем VideoWriter для выходного видео (если указан путь)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        frame_count = 0
        clap_detected = False
        
        print("Начинаем обработку видео...")
        print("Нажмите 'q' для выхода, 'p' для паузы")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Конвертируем BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Детекция позы
            results = self.pose.process(frame_rgb)
            
            # Обратно в BGR для отображения
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # Рисуем landmarks
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Детекция хлопка
                clap_detected = self.detect_clap(results.pose_landmarks.landmark, frame_count)
                
                # Получаем координаты запястий для визуализации
                left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                
                # Рисуем линию между руками
                left_pos = (int(left_wrist.x * width), int(left_wrist.y * height))
                right_pos = (int(right_wrist.x * width), int(right_wrist.y * height))
                cv2.line(frame_bgr, left_pos, right_pos, (255, 255, 0), 2)
            
            # Добавляем информацию на кадр
            cv2.putText(frame_bgr, f"Clap count: {self.clap_count}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if clap_detected:
                cv2.putText(frame_bgr, "CLAPP!", (width//2 - 100, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                clap_detected = False
            
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Отображаем кадр
            cv2.imshow('Clap Counter', frame_bgr)
            
            # Сохраняем кадр если нужно
            if out:
                out.write(frame_bgr)
            
            # Управление
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        # Освобождаем ресурсы
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена!")
        print(f"Всего хлопков над головой: {self.clap_count}")
        print(f"Обработано кадров: {frame_count}")
        
        return self.clap_count