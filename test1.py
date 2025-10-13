import cv2
import face_recognition
import numpy as np
import mediapipe as mp
import os
from collections import deque
import time

# ==============================
# 1️⃣ CẢI THIỆN ĐỌC VÀ XỬ LÝ ẢNH
# ==============================
known_encodings = []
known_names = []

main_dir = r"D:\python\nhận diện khuôn mặt\known_faces"

print("📂 Đang tải dữ liệu khuôn mặt...\n")

def load_and_process_image(file_path):
    """Tải và xử lý ảnh với nhiều phương pháp"""
    try:
        image = face_recognition.load_image_file(file_path)
        
        # Thử nhiều model và tham số
        face_encs = face_recognition.face_encodings(image)
        if not face_encs:
            # Thử với model khác
            face_encs = face_recognition.face_encodings(image, model='large')
        
        return face_encs
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh {file_path}: {e}")
        return []

for person_name in os.listdir(main_dir):
    person_folder = os.path.join(main_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    encodings = []
    valid_images = 0
    
    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(person_folder, filename)
            face_encs = load_and_process_image(file_path)
            
            if face_encs:
                # Lưu nhiều encoding từ cùng một ảnh (nếu có nhiều khuôn mặt)
                for enc in face_encs:
                    encodings.append(enc)
                valid_images += 1
                print(f"✅ {person_name} - {filename} ({len(face_encs)} khuôn mặt)")
            else:
                print(f"⚠️ Không phát hiện khuôn mặt trong {filename}")

    if encodings:
        # Sử dụng trung bình của tất cả encoding
        mean_encoding = np.mean(encodings, axis=0)
        known_encodings.append(mean_encoding)
        known_names.append(person_name)
        print(f"👤 {person_name}: {valid_images} ảnh hợp lệ, {len(encodings)} encoding")

print(f"\n🧠 Tổng số người đã học: {len(known_names)}\n")

# ==============================
# 2️⃣ CẢI THIỆN FACE DETECTION
# ==============================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Sử dụng model chính xác hơn
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # Model 1 cho độ chính xác cao hơn
    min_detection_confidence=0.7
)

# ==============================
# 3️⃣ THÊM BỘ LỌC VÀ THEO DÕI
# ==============================
class FaceRecognizer:
    def __init__(self, known_encodings, known_names, tolerance=0.45):
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.tolerance = tolerance
        self.name_history = {}  # Lưu lịch sử nhận diện
        self.confidence_threshold = 0.6
        
    def recognize_face(self, face_encoding):
        """Nhận diện khuôn mặt với bộ lọc confidence"""
        if len(self.known_encodings) == 0:
            return "unknown", 0
            
        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Chuyển distance thành confidence score (0-1)
        confidence = max(0, 1 - best_distance)
        
        if confidence >= self.confidence_threshold:
            return self.known_names[best_match_index], confidence
        else:
            return "unknown", confidence

# Khởi tạo recognizer
recognizer = FaceRecognizer(known_encodings, known_names)

# ==============================
# 4️⃣ CẢI THIỆN VIDEO PROCESSING
# ==============================
def preprocess_face_crop(face_crop):
    """Tiền xử lý ảnh khuôn mặt để cải thiện nhận diện"""
    if face_crop.size == 0:
        return None
        
    # Resize ảnh nếu quá nhỏ
    h, w = face_crop.shape[:2]
    if h < 80 or w < 80:
        scale_factor = max(80/h, 80/w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        face_crop = cv2.resize(face_crop, (new_w, new_h))
    
    # Tăng cường độ tương phản
    lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
    face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return face_crop

# ==============================
# 5️⃣ VÒNG LẶP CHÍNH ĐƯỢC TỐI ƯU
# ==============================
cap = cv2.VideoCapture(0)

# Cài đặt camera để chất lượng tốt hơn
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 Đang bật camera... Nhấn 'q' để thoát.\n")

# Biến để đo FPS
fps_counter = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật frame để có hiệu ứng gương
    frame = cv2.flip(frame, 1)
    
    # Chuyển đổi màu
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt với MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Lấy bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)
            
            # Giới hạn hợp lệ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Kiểm tra kích thước khuôn mặt
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width < 50 or face_height < 50:  # Bỏ qua khuôn mặt quá nhỏ
                continue

            # Cắt vùng khuôn mặt (tăng thêm biên)
            pad = 30
            face_crop = rgb_frame[max(0, y1-pad):min(h, y2+pad),
                                max(0, x1-pad):min(w, x2+pad)]
            
            name = "unknown"
            confidence = 0
            color = (0, 0, 255)  # Mặc định màu đỏ

            if face_crop.size > 0:
                # Tiền xử lý ảnh
                processed_face = preprocess_face_crop(face_crop)
                
                if processed_face is not None:
                    try:
                        # Nhận diện khuôn mặt
                        encodings = face_recognition.face_encodings(processed_face)
                        
                        if encodings:
                            name, confidence = recognizer.recognize_face(encodings[0])
                            
                            # Đổi màu dựa trên confidence
                            if name != "unknown":
                                if confidence > 0.8:
                                    color = (0, 255, 0)  # Xanh lá - confidence cao
                                else:
                                    color = (255, 255, 0)  # Vàng - confidence trung bình
                            
                    except Exception as e:
                        print(f"Lỗi nhận diện: {e}")

            # Vẽ kết quả
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            
            # Hiển thị tên và confidence
            display_text = f"{name} ({confidence:.2f})" if name != "unknown" else "unknown"
            cv2.putText(frame, display_text, (x1 + 6, y2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tính và hiển thị FPS
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI Face Recognition - Improved Version", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()