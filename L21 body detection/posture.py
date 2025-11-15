import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class PostureTracker:
    def __init__(self):
        # Инициализация MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Статистика и метрики осанки
        self.posture_history = deque(maxlen=30)
        self.shoulder_angles = deque(maxlen=10)
        self.back_angles = deque(maxlen=10)
        
        # Пороговые значения для осанки
        self.SHOULDER_SLANT_THRESHOLD = 5.0  # градусы для наклона плеч
        self.BACK_CURVE_THRESHOLD = 160.0    # градусы для прямой спины
        self.SLOUCH_THRESHOLD = 0.08         # смещение головы вперед
        
        # Состояния осанки
        self.good_posture_frames = 0
        self.bad_posture_frames = 0
        self.total_frames = 0
        
        # Цвета для визуализации
        self.COLOR_GOOD = (0, 255, 0)      # Зеленый
        self.COLOR_WARNING = (0, 255, 255) # Желтый
        self.COLOR_BAD = (0, 0, 255)       # Красный
        
    def get_landmark_point(self, landmarks, landmark_type):
        """Безопасное получение точки landmark"""
        if landmarks and hasattr(landmarks, 'landmark'):
            return landmarks.landmark[landmark_type]
        return None
        
    def calculate_angle(self, a, b, c):
        """Вычисляет угол между тремя точками в градусах"""
        if a is None or b is None or c is None:
            return 0
            
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_slope(self, point1, point2):
        """Вычисляет наклон линии между двумя точками в градусах"""
        if point1 is None or point2 is None:
            return 0
            
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        return np.degrees(np.arctan2(dy, dx))
    
    def analyze_shoulders(self, landmarks):
        """Анализирует положение плеч"""
        left_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        
        if left_shoulder is None or right_shoulder is None:
            return 0, 0
        
        # Вычисляем наклон линии плеч
        shoulder_slope = self.calculate_slope(left_shoulder, right_shoulder)
        
        # Абсолютное значение наклона (нам важна величина, не направление)
        shoulder_slant = abs(shoulder_slope)
        
        return shoulder_slant, shoulder_slope
    
    def analyze_back(self, landmarks):
        """Анализирует изгиб спины"""
        left_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_ear = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR)
        
        if None in [left_shoulder, right_shoulder, left_hip, right_hip, left_ear]:
            return 0, 0
        
        # Средняя точка плеч
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Оценка сутулости (голова выдвинута вперед)
        head_forward = left_ear.x - shoulder_mid_x
        
        # Угол спины (плечи-бедра)
        back_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
        
        return back_angle, head_forward
    
    def evaluate_posture(self, landmarks):
        """Оценивает осанку по нескольким параметрам"""
        if not landmarks:
            return "НЕТ ДАННЫХ", self.COLOR_WARNING, 0, [], {
                'shoulder_angle': 0,
                'back_angle': 0,
                'head_forward': 0
            }
        
        # Анализ плеч
        shoulder_slant, shoulder_slope = self.analyze_shoulders(landmarks)
        
        # Анализ спины
        back_angle, head_forward = self.analyze_back(landmarks)
        
        # Сохраняем историю
        self.shoulder_angles.append(shoulder_slant)
        self.back_angles.append(back_angle)
        
        # Усредняем значения для сглаживания
        avg_shoulder_angle = np.mean(list(self.shoulder_angles)) if self.shoulder_angles else 0
        avg_back_angle = np.mean(list(self.back_angles)) if self.back_angles else 0
        
        # Оценка осанки
        score = 100
        issues = []
        
        # Проверка наклона плеч
        if avg_shoulder_angle > self.SHOULDER_SLANT_THRESHOLD:
            score -= 30
            issues.append(f"Angle of sholder: {avg_shoulder_angle:.1f}°")
        
        # Проверка изгиба спины
        if avg_back_angle < self.BACK_CURVE_THRESHOLD:
            score -= 30
            issues.append(f"Bad back: {avg_back_angle:.1f}°")
        
        # Проверка сутулости (голова вперед)
        if abs(head_forward) > self.SLOUCH_THRESHOLD:
            score -= 20
            direction = "to the left" if head_forward > 0 else "to the right"
            issues.append(f"Head decline {direction}")
        
        # Определение категории осанки
        if score >= 80:
            posture_status = "Excellent"
            color = self.COLOR_GOOD
            self.good_posture_frames += 1
        elif score >= 60:
            posture_status = "good"
            color = self.COLOR_WARNING
            self.bad_posture_frames += 1
        else:
            posture_status = "bad"
            color = self.COLOR_BAD
            self.bad_posture_frames += 1
        
        self.total_frames += 1
        
        return posture_status, color, score, issues, {
            'shoulder_angle': avg_shoulder_angle,
            'back_angle': avg_back_angle,
            'head_forward': head_forward
        }
    
    def draw_posture_analysis(self, frame, landmarks, posture_info):
        """Рисует анализ осанки на кадре"""
        posture_status, color, score, issues, metrics = posture_info
        
        height, width = frame.shape[:2]
        
        # Рисуем landmarks позы
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Рисуем дополнительные линии для анализа
        self.draw_posture_lines(frame, landmarks, metrics, width, height)
        
        # Отображаем статус осанки
        cv2.putText(frame, f"Posture: {posture_status}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(frame, f"Mark: {score}/100", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Отображаем метрики
        y_offset = 120
        cv2.putText(frame, f"Angle of shoulder: {metrics['shoulder_angle']:.1f}°", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Angle of back: {metrics['back_angle']:.1f}°", (20, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Отображаем проблемы
        y_offset += 70
        for i, issue in enumerate(issues):
            cv2.putText(frame, f"! {issue}", (20, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 1)
        
        # Статистика
        if self.total_frames > 0:
            good_percentage = (self.good_posture_frames / self.total_frames * 100)
            cv2.putText(frame, f"good posture: {good_percentage:.1f}%", (width - 300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_posture_lines(self, frame, landmarks, metrics, width, height):
        """Рисует дополнительные линии для визуализации осанки"""
        if not landmarks:
            return
            
        # Получаем точки плеч
        left_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self.get_landmark_point(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        
        if None in [left_shoulder, right_shoulder, left_hip]:
            return
        
        # Линия плеч
        left_shoulder_pos = (int(left_shoulder.x * width), int(left_shoulder.y * height))
        right_shoulder_pos = (int(right_shoulder.x * width), int(right_shoulder.y * height))
        
        # Цвет линии плеч в зависимости от наклона
        shoulder_color = (0, 255, 0)  # зеленый по умолчанию
        if metrics['shoulder_angle'] > self.SHOULDER_SLANT_THRESHOLD:
            shoulder_color = (0, 165, 255)  # оранжевый при наклоне
        if metrics['shoulder_angle'] > self.SHOULDER_SLANT_THRESHOLD * 2:
            shoulder_color = (0, 0, 255)  # красный при сильном наклоне
        
        cv2.line(frame, left_shoulder_pos, right_shoulder_pos, shoulder_color, 3)
        
        # Линия спины (плечи - бедра)
        left_hip_pos = (int(left_hip.x * width), int(left_hip.y * height))
        cv2.line(frame, left_shoulder_pos, left_hip_pos, (255, 255, 0), 2)
    
    def process_video(self, video_path, output_path=None):
        """Обрабатывает видео и анализирует осанку"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео {video_path}")
            return
        
        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем VideoWriter для выходного видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        print("Запуск анализа осанки...")
        print("Нажмите 'q' для выхода, 'p' для паузы")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Конвертируем BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Детекция позы
            results = self.pose.process(frame_rgb)
            
            # Обратно в BGR для отображения
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # Анализ осанки
                posture_info = self.evaluate_posture(results.pose_landmarks)
                
                # Рисуем анализ на кадре
                self.draw_posture_analysis(frame_bgr, results.pose_landmarks, posture_info)
            else:
                # Если поза не обнаружена
                cv2.putText(frame_bgr, "Pose is not define", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_WARNING, 2)
            
            # Отображаем кадр
            cv2.imshow('Posture Tracker', frame_bgr)
            
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
        
        # Выводим итоговую статистику
        self.print_final_stats()
    
    def print_final_stats(self):
        """Выводит итоговую статистику по осанке"""
        print("\n" + "="*50)
        print("ИТОГОВАЯ СТАТИСТИКА ОСАНКИ")
        print("="*50)
        
        if self.total_frames > 0:
            good_percentage = (self.good_posture_frames / self.total_frames) * 100
            bad_percentage = (self.bad_posture_frames / self.total_frames) * 100
            
            print(f"Всего проанализировано кадров: {self.total_frames}")
            print(f"Кадры с хорошей осанкой: {self.good_posture_frames} ({good_percentage:.1f}%)")
            print(f"Кадры с плохой осанкой: {self.bad_posture_frames} ({bad_percentage:.1f}%)")
            
            if good_percentage >= 70:
                print("🎉 Отличный результат! Осанка в основном правильная!")
            elif good_percentage >= 50:
                print("👍 Хороший результат, но есть над чем поработать")
            else:
                print("💪 Нужно уделить больше внимания осанке!")
        else:
            print("Не удалось проанализировать ни одного кадра")