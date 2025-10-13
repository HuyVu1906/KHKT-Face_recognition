import cv2
import numpy as np
import os
from collections import deque, Counter
import time
import pickle
from datetime import datetime
import pandas as pd
from insightface.app import FaceAnalysis

# ==============================
# 1️⃣ LƯU TRỮ VÀ QUẢN LÝ DỮ LIỆU
# ==============================
class FaceDatabase:
    def __init__(self, database_path="face_database.pkl"):
        self.database_path = database_path
        self.known_encodings = []
        self.known_names = []
        self.face_metadata = {}
        self.load_database()
    
    def load_database(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.face_metadata = data.get('metadata', {})
            print(f"📂 Đã tải database: {len(self.known_names)} người")
    
    def save_database(self):
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names,
            'metadata': self.face_metadata
        }
        with open(self.database_path, 'wb') as f:
            pickle.dump(data, f)
        print("💾 Đã lưu database")
    
    def add_person(self, name, encodings, metadata=None):
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
    
    def assign_face_id(self, bbox):
        for face_id, history in self.face_history.items():
            last_bbox = history[-1] if history else None
            if last_bbox and self.calculate_iou(bbox, last_bbox) > 0.3:
                return face_id
        self.next_face_id += 1
        return self.next_face_id

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def update_history(self, face_id, bbox):
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
        if face_id not in self.vote_history:
            self.vote_history[face_id] = deque(maxlen=10)
        self.vote_history[face_id].append((name, confidence, time.time()))
        now = time.time()
        self.vote_history[face_id] = deque(
            [v for v in self.vote_history[face_id] if now - v[2] < 5],
            maxlen=10
        )
    
    def get_consensus(self, face_id):
        if face_id not in self.vote_history:
            return "unknown", 0
        votes = [v[0] for v in self.vote_history[face_id]]
        counts = Counter(votes)
        if not counts:
            return "unknown", 0
        name, count = counts.most_common(1)[0]
        if count >= self.vote_threshold:
            confs = [v[1] for v in self.vote_history[face_id] if v[0] == name]
            return name, np.mean(confs)
        return "unknown", 0


# ==============================
# 4️⃣ GHI LOG VÀ THỐNG KÊ
# ==============================
class RecognitionLogger:
    def __init__(self, log_file="recognition_log.csv"):
        self.log_file = log_file
        self.initialize_log()
    
    def initialize_log(self):
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=["timestamp", "name"])
            df.to_csv(self.log_file, index=False)
            print(f"📝 Tạo file log mới: {self.log_file}")
    
    def log_recognition(self, name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([{"timestamp": timestamp, "name": name}])
        df.to_csv(self.log_file, mode='a', header=False, index=False)


# ==============================
# 5️⃣ HỆ THỐNG CHÍNH VỚI INSIGHTFACE
# ==============================
class AdvancedFaceRecognitionSystem:
    def __init__(self):
        self.database = FaceDatabase()
        self.tracker = FaceTracker()
        self.voter = VotingSystem()
        self.logger = RecognitionLogger()
        self.recognition_stats = {}

        # InsightFace app
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def recognize_face(self, face_embedding):
        if len(self.database.known_encodings) == 0:
            return "unknown", 0

        embeddings = np.array(self.database.known_encodings)
        sims = np.dot(embeddings, face_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(face_embedding) + 1e-10
        )
        best_idx = np.argmax(sims)
        confidence = sims[best_idx]
        if confidence >= 0.35:
            name = self.database.known_names[best_idx]
            self.database.face_metadata[name]['last_seen'] = datetime.now()
            self.database.face_metadata[name]['recognition_count'] += 1
            return name, confidence
        return "unknown", confidence

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        for face in faces:
            bbox = [int(v) for v in face.bbox]
            x1, y1, x2, y2 = bbox
            embedding = face.embedding

            face_id = self.tracker.assign_face_id(bbox)
            self.tracker.update_history(face_id, bbox)

            name, conf = self.recognize_face(embedding)
            self.voter.vote(face_id, name, conf)
            final_name, final_conf = self.voter.get_consensus(face_id)

            color = (0, 255, 0) if final_name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{face_id} {final_name} ({final_conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if final_name != "unknown":
                self.logger.log_recognition(final_name)
                self.recognition_stats[final_name] = self.recognition_stats.get(final_name, 0) + 1

        return frame


# ==============================
# 6️⃣ CHƯƠNG TRÌNH CHÍNH
# ==============================
def main():
    print("🚀 Khởi động hệ thống nhận diện bằng InsightFace (buffalo_l)...")

    system = AdvancedFaceRecognitionSystem()

    # Tải dữ liệu khuôn mặt
    main_dir = r"D:\python\nhận diện khuôn mặt\known_faces"
    print("📂 Đang tải dữ liệu khuôn mặt...")

    for person_name in os.listdir(main_dir):
        folder = os.path.join(main_dir, person_name)
        if os.path.isdir(folder):
            encodings = []
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(folder, filename)
                    image = cv2.imread(img_path)
                    faces = system.app.get(image)
                    for f in faces:
                        encodings.append(f.embedding)
                    print(f"✅ {person_name} - {filename}")
            if encodings and person_name not in system.database.known_names:
                system.database.add_person(person_name, encodings)

    print(f"🧠 Đã tải {len(system.database.known_names)} người")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🎥 Nhấn 'q' để thoát, 's' để xem thống kê")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        processed = system.process_frame(frame)
        cv2.imshow("AI Face Recognition - InsightFace", processed)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n📊 Thống kê nhận diện:")
            for name, count in system.recognition_stats.items():
                print(f"  {name}: {count} lần")

    cap.release()
    cv2.destroyAllWindows()
    system.database.save_database()
    print("💾 Đã lưu dữ liệu và thoát.")


if __name__ == "__main__":
    main()
