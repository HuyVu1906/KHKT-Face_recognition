import cv2
import face_recognition
import numpy as np
import mediapipe as mp
import os
from collections import deque, Counter
import time
import pickle
from datetime import datetime
import pandas as pd

# ==============================
# 1️⃣ LƯU TRỮ VÀ QUẢN LÝ DỮ LIỆU
# ==============================
class FaceDatabase:
    def __init__(self, database_path="face_database.pkl"):
        self.database_path = database_path
        self.known_encodings = []
        self.known_names = []
        self.face_metadata = {}  # Lưu thêm thông tin
        self.load_database()
    
    def load_database(self):
        """Tải database từ file"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.face_metadata = data.get('metadata', {})
            print(f"📂 Đã tải database: {len(self.known_names)} người")
    
    def save_database(self):
        """Lưu database vào file"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names,
            'metadata': self.face_metadata
        }
        with open(self.database_path, 'wb') as f:
            pickle.dump(data, f)
        print("💾 Đã lưu database")
    
    def add_person(self, name, encodings, metadata=None):
        """Thêm người mới vào database"""
        if encodings:
            mean_encoding = np.mean(encodings, axis=0)
            self.known_encodings.append(mean_encoding)
            self.known_names.append(name)
            
            self.face_metadata[name] = {
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'recognition_count': 0,
                'images_count': len(encodings),
                'custom_data': metadata or {}
            }
            self.save_database()
            return True
        return False

# ==============================
# 2️⃣ THEO DÕI KHUÔN MẶT (FACE TRACKING)
# ==============================
class FaceTracker:
    def __init__(self, max_history=10):
        self.face_history = {}
        self.max_history = max_history
        self.next_face_id = 0
    
    def track_faces(self, detections, frame_shape):
        """Theo dõi khuôn mặt qua các frame"""
        current_faces = {}
        
        for detection in detections:
            bbox = self.get_bounding_box(detection, frame_shape)
            face_id = self.assign_face_id(bbox)
            current_faces[face_id] = bbox
            
        self.update_history(current_faces)
        return current_faces
    
    def get_bounding_box(self, detection, frame_shape):
        h, w = frame_shape[:2]
        bboxC = detection.location_data.relative_bounding_box
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = int((bboxC.xmin + bboxC.width) * w)
        y2 = int((bboxC.ymin + bboxC.height) * h)
        return (x1, y1, x2, y2)
    
    def assign_face_id(self, bbox):
        """Gán ID cho khuôn mặt dựa trên vị trí"""
        for face_id, history in self.face_history.items():
            last_bbox = history[-1] if history else None
            if last_bbox and self.calculate_iou(bbox, last_bbox) > 0.3:
                return face_id
        self.next_face_id += 1
        return self.next_face_id
    
    def calculate_iou(self, box1, box2):
        """Tính Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return inter_area / (box1_area + box2_area - inter_area)
    
    def update_history(self, current_faces):
        """Cập nhật lịch sử khuôn mặt"""
        for face_id in list(self.face_history.keys()):
            if face_id not in current_faces:
                if len(self.face_history[face_id]) > self.max_history:
                    del self.face_history[face_id]
        
        for face_id, bbox in current_faces.items():
            if face_id not in self.face_history:
                self.face_history[face_id] = deque(maxlen=self.max_history)
            self.face_history[face_id].append(bbox)

# ==============================
# 3️⃣ HỆ THỐNG VOTE VÀ LÀM MỊN
# ==============================
class VotingSystem:
    def __init__(self, vote_threshold=3):
        self.vote_threshold = vote_threshold
        self.vote_history = {}
    
    def vote(self, face_id, name, confidence):
        """Thêm vote cho khuôn mặt"""
        if face_id not in self.vote_history:
            self.vote_history[face_id] = deque(maxlen=10)
        
        self.vote_history[face_id].append((name, confidence, time.time()))
        
        # Lọc votes cũ (trong 5 giây gần nhất)
        current_time = time.time()
        self.vote_history[face_id] = deque(
            [vote for vote in self.vote_history[face_id] if current_time - vote[2] < 5],
            maxlen=10
        )
    
    def get_consensus(self, face_id):
        """Lấy kết quả đồng thuận"""
        if face_id not in self.vote_history or not self.vote_history[face_id]:
            return "unknown", 0
        
        votes = [vote[0] for vote in self.vote_history[face_id]]
        vote_count = Counter(votes)
        
        if vote_count:
            most_common_name, count = vote_count.most_common(1)[0]
            if count >= self.vote_threshold:
                # Tính confidence trung bình
                confidences = [vote[1] for vote in self.vote_history[face_id] if vote[0] == most_common_name]
                avg_confidence = np.mean(confidences)
                return most_common_name, avg_confidence
        
        return "unknown", 0

# ==============================
# 4️⃣ GHI LOG VÀ THỐNG KÊ
# ==============================
class RecognitionLogger:
    def __init__(self, log_file="recognition_log.csv"):
        self.log_file = log_file
        self.initialize_log()  # ĐẢM BẢO CÓ DÒNG NÀY
    
    def initialize_log(self):
        """Khởi tạo file log với các cột mong muốn"""
        if not os.path.exists(self.log_file):
            # ĐỊNH NGHĨA CÁC CỘT - THEO MỤC 1 CỦA BẠN
            df = pd.DataFrame(columns=[
                'timestamp',     # Cột ngày giờ
                'name'          # Cột tên
                # ĐÃ XÓA CÁC CỘT KHÁC THEO YÊU CẦU
            ])
            df.to_csv(self.log_file, index=False)
            print(f"📝 Đã tạo file log mới: {self.log_file}")
    
    def log_recognition(self, name, confidence, face_id):
        """Ghi nhận diện vào log với định dạng cột"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # DỮ LIỆU THEO CỘT - CHỈ 2 CỘT NHƯ MỤC 1
        new_data = pd.DataFrame([{
            'timestamp': timestamp,    # Cột 1: Ngày giờ
            'name': name              # Cột 2: Tên
        }])
        
        new_data.to_csv(self.log_file, mode='a', header=False, index=False)

# ==============================
# 5️⃣ CẢI THIỆN TIỀN XỬ LÝ ẢNH
# ==============================
class ImageEnhancer:
    @staticmethod
    def enhance_face(face_image):
        """Tăng cường chất lượng ảnh khuôn mặt"""
        if face_image.size == 0:
            return None
        
        # Resize ảnh
        h, w = face_image.shape[:2]
        if h < 100 or w < 100:
            scale_factor = max(100/h, 100/w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            face_image = cv2.resize(face_image, (new_w, new_h))
        
        # Tăng cường độ sáng và tương phản
        lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Giảm nhiễu
        face_image = cv2.medianBlur(face_image, 3)
        
        # Tăng độ sắc nét
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        face_image = cv2.filter2D(face_image, -1, kernel)
        
        return face_image

# ==============================
# 6️⃣ HỆ THỐNG CHÍNH TÍCH HỢP
# ==============================
class AdvancedFaceRecognitionSystem:
    def __init__(self):
        self.database = FaceDatabase()
        self.tracker = FaceTracker()
        self.voter = VotingSystem()
        self.logger = RecognitionLogger()
        self.enhancer = ImageEnhancer()
        
        # Khởi tạo MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        
        self.recognition_stats = {}
    
    def recognize_face(self, face_encoding):
        """Nhận diện khuôn mặt nâng cao"""
        if len(self.database.known_encodings) == 0:
            return "unknown", 0
        
        face_distances = face_recognition.face_distance(
            self.database.known_encodings, face_encoding
        )
        
        if len(face_distances) == 0:
            return "unknown", 0
        
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        confidence = max(0, 1 - best_distance)
        
        # Ngưỡng confidence linh hoạt
        confidence_threshold = 0.5
        if confidence >= confidence_threshold:
            name = self.database.known_names[best_match_index]
            
            # Cập nhật thống kê
            if name not in self.recognition_stats:
                self.recognition_stats[name] = 0
            self.recognition_stats[name] += 1
            
            # Cập nhật metadata
            if name in self.database.face_metadata:
                self.database.face_metadata[name]['last_seen'] = datetime.now()
                self.database.face_metadata[name]['recognition_count'] += 1
            
            return name, confidence
        
        return "unknown", confidence
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            tracked_faces = self.tracker.track_faces(results.detections, frame.shape)
            
            for detection, (face_id, bbox) in zip(results.detections, tracked_faces.items()):
                x1, y1, x2, y2 = bbox
                
                # Cắt và xử lý khuôn mặt
                pad = 25
                h, w = frame.shape[:2]
                face_crop = rgb_frame[
                    max(0, y1-pad):min(h, y2+pad),
                    max(0, x1-pad):min(w, x2+pad)
                ]
                
                name, confidence = "unknown", 0
                color = (0, 0, 255)  # Mặc định màu ĐỎ
                
                if face_crop.size > 0:
                    enhanced_face = self.enhancer.enhance_face(face_crop)
                    
                    if enhanced_face is not None:
                        try:
                            encodings = face_recognition.face_encodings(enhanced_face)
                            if encodings:
                                # Nhận diện
                                name, confidence = self.recognize_face(encodings[0])
                                
                                # Voting system
                                self.voter.vote(face_id, name, confidence)
                                final_name, final_confidence = self.voter.get_consensus(face_id)
                                

                                if name != "unknown":
                                    if confidence > 0.8:
                                        color = (0, 255, 0)      
                                    elif confidence > 0.6:
                                        color = (255, 255, 0)    
                                    else:
                                        color = (0, 165, 255)    
                                else:
                                    color = (0, 0, 255)         
                                    
                                # Log kết quả
                                if name != "unknown":
                                    self.logger.log_recognition(name, confidence, face_id)
                                    
                        except Exception as e:
                            print(f"Lỗi nhận diện: {e}")
                
                # Vẽ kết quả
                self.draw_result(frame, bbox, name, confidence, color, face_id)
        
        return frame
    
    def draw_result(self, frame, bbox, name, confidence, color, face_id):
        """Vẽ kết quả lên frame"""
        x1, y1, x2, y2 = bbox
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ background cho text
        cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), color, cv2.FILLED)
        
        # Hiển thị thông tin
        info_text = f"ID:{face_id} {name} ({confidence:.2f})"
        cv2.putText(frame, info_text, (x1 + 6, y2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hiển thị ID phía trên
        cv2.putText(frame, f"Face {face_id}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ==============================
# 7️⃣ CHƯƠNG TRÌNH CHÍNH
# ==============================
def main():
    print("🚀 Khởi động hệ thống nhận diện nâng cao...")
    
    # Khởi tạo hệ thống
    system = AdvancedFaceRecognitionSystem()
    
    # Tải dữ liệu khuôn mặt
    main_dir = r"D:\python\nhận diện khuôn mặt\known_faces"
    
    print("📂 Đang tải dữ liệu khuôn mặt...")
    for person_name in os.listdir(main_dir):
        person_folder = os.path.join(main_dir, person_name)
        if os.path.isdir(person_folder):
            encodings = []
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    file_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(file_path)
                    face_encs = face_recognition.face_encodings(image)
                    if face_encs:
                        encodings.extend(face_encs)
                        print(f"✅ {person_name} - {filename}")
            
            if encodings and person_name not in system.database.known_names:
                system.database.add_person(person_name, encodings)
    
    print(f"\n🧠 Đã tải: {len(system.database.known_names)} người")
    
    # Khởi động camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_counter = 0
    fps_time = time.time()
    
    print("🎥 Camera đã sẵn sàng. Nhấn 'q' để thoát, 's' để thống kê")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Xử lý frame
        processed_frame = system.process_frame(frame)
        
        # Hiển thị FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
            #cv2.putText(processed_frame, f"FPS: {fps}", (10, 30),
                       #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị thống kê
        #stats_text = f"Faces tracked: {len(system.tracker.face_history)}"
        #cv2.putText(processed_frame, stats_text, (10, 60),
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("AI Face Recognition - Advanced System", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n📊 Thống kê nhận diện:")
            for name, count in system.recognition_stats.items():
                print(f"   {name}: {count} lần")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Lưu database trước khi thoát
    system.database.save_database()
    print("💾 Đã lưu dữ liệu và thoát.")

if __name__ == "__main__":
    main()
