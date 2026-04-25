import cv2, os, time, numpy as np, queue
from insightface.app import FaceAnalysis
from core.FaceDatabase import FaceDatabase
from core.Facetracker import FaceTracker
from core.Votingsystem import VotingSystem
from core.IOWorker import IOWorker
from core.FaissIndex import FaissIndex
from datetime import datetime
class AdvancedFaceRecognitionSystem:
    def __init__(self):
        self.database = FaceDatabase()
        self.tracker = FaceTracker()
        self.voter = VotingSystem()
        self.last_save = time.time()
        self.last_logged_time = {}
        self.cooldown_seconds = 3
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(480,480))
        self.faiss_index = FaissIndex(dim=512)
        self.io_q = queue.Queue()
        self.io_worker = IOWorker(self.io_q)
        self.io_worker.start()
    def build_ann_index(self):
        if len(self.database.known_encodings) == 0:
            print("⚠️ Chưa có dữ liệu khuôn mặt để tạo index.")
            return
        print("⚙️ Đang chuẩn hóa embedding ...")

        all_embs = []
        all_names = []
        for name, info in self.database.face_metadata.items():
            if "embeddings" in info:
                for vec in info["embeddings"]:
                    all_embs.append(vec)
                    all_names.append(name)

        if not all_embs:
            
            all_embs = self.database.known_encodings
            all_names = self.database.known_names

        if not all_embs:
            print("⚠️ Không có embeddings để tạo ANN index.")
            return

        self.faiss_index.build_index(all_embs, all_names)
        print(f"✅ FAISS index rebuilt từ {len(all_embs)} vectors của {len(set(all_names))} người")
    def add_person(self, name, folder_path):
        if not os.path.exists(folder_path):
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
            return

        encs = []
        for fn in os.listdir(folder_path):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, fn)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            faces = self.app.get(img)
            for f in faces:
                encs.append(f.embedding)

        if not encs:
            print(f"⚠️ Không tìm thấy khuôn mặt hợp lệ trong thư mục {folder_path}")
            return

        mean_vec = np.mean(encs, axis=0)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-10
        self.database.known_names.append(name)
        self.database.known_encodings.append(mean_vec.astype(np.float32))
        self.database.face_metadata[name] = {
            "embeddings": [e.astype(np.float32) for e in encs],
            "added": datetime.now().isoformat(),
            "num_images": len(encs)
        }
        self.database.save_database()
        self.faiss_index.build_index(self.database.known_encodings, self.database.known_names)
        print(f"✅ Đã thêm {name} vào database ({len(encs)} ảnh)")
    def rescan_known_faces(self, main_dir="d:\python\face\known_faces"):

        known_set = set(self.database.known_names)
        new_people = []

        for root, dirs, files in os.walk(main_dir):
            for person_name in dirs:
                folder = os.path.join(root, person_name)
                if not os.path.isdir(folder):
                    continue
                class_name = os.path.basename(os.path.dirname(folder))
                label_name = f"{person_name}_{class_name}"
                if person_name in known_set:
                    continue 

                encs = []
                for fn in os.listdir(folder):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = os.path.join(folder, fn)
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    faces = self.app.get(img)
                    for f in faces:
                        encs.append(f.embedding)

                if encs:
                    mean_vec = np.mean(encs, axis=0)
                    mean_vec /= np.linalg.norm(mean_vec) + 1e-10
                    self.database.known_names.append(person_name)
                    self.database.known_encodings.append(mean_vec.astype(np.float32))
                    new_people.append(person_name)
                    print(f"✅ Đã thêm {person_name} ({len(encs)} ảnh)")               
        if new_people:
            self.database.save_database()
            self.faiss_index.build_index(self.database.known_encodings, self.database.known_names)
            print(f"💾 Database cập nhật với {len(new_people)} người mới.")
        else:
            print("📂 Không có người mới nào được thêm.")
    def build_database_from_images(self, main_dir="known_faces"):
        print("🧠 Đang tạo database khuôn mặt từ ảnh...")

        people = []
        for root, dirs, files in os.walk(main_dir):
            for person_name in dirs:
                folder = os.path.join(root, person_name)
                if not os.path.isdir(folder):
                    continue
                class_name = os.path.basename(os.path.dirname(folder))
                label_name = f"{person_name}_{class_name}"
                encodings = []
                for filename in os.listdir(folder):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(folder, filename)
                        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        faces = self.app.get(img)
                        for face in faces:
                            encodings.append(face.embedding)

                if encodings:
                    mean_encoding = np.mean(encodings, axis=0)
                    mean_encoding = mean_encoding / (np.linalg.norm(mean_encoding) + 1e-10)
                    people.append((label_name, mean_encoding))
                    print(f"✅ {label_name}: {len(encodings)} ảnh → lưu 1 vector")
        if people:
            self.database.known_names = []
            self.database.known_encodings = []
            for name, vec in people:
                self.database.known_names.append(name)
                self.database.known_encodings.append(vec.astype(np.float32))
            self.database.save_database()
            print(f"💾 Đã lưu database: {len(people)} người")
            self.faiss_index.build_index(self.database.known_encodings, self.database.known_names)
    def process_frame(self, frame):
        small = cv2.resize(frame, (480, 360))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        if not faces:
            return frame

        bboxes, embeddings, face_ids = [], [], []
        scale_x = frame.shape[1] / 480
        scale_y = frame.shape[0] / 360
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            bbox = [x1, y1, x2, y2]
            embedding = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
            face_id = self.tracker.assign_face_id(bbox)
            self.tracker.update_history(face_id, bbox)
            bboxes.append(bbox)
            embeddings.append(embedding)
            face_ids.append(face_id)

        names, confs = [], []
        for emb in embeddings:
            n, c = self.faiss_index.search(emb)
            names.append(n)
            confs.append(c)
        now_dt = datetime.now()
        timestamp_text = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Vẽ nền cho chữ để dễ đọc hơn (màu đen)
        cv2.rectangle(frame, (8, 8), (400, 40), (0, 0, 0), -1) 
        
        # Vẽ text ngày giờ lên góc khung hình (ví dụ: góc trên bên trái)
        cv2.putText(frame, timestamp_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        for bbox, name, conf, face_id in zip(bboxes, names, confs, face_ids):
            x1, y1, x2, y2 = bbox
            self.voter.vote(face_id, name, conf)
            final_name, final_conf = self.voter.get_consensus(face_id)

            # --- Màu khung ---
            color = (0, 255, 0) if final_name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # --- Xử lý text hiển thị ---
            name_parts = final_name.split("_", 1)
            if len(name_parts) == 2:
                person_name, class_name = name_parts
                show_text = f"{person_name} - {class_name}"
            else:
                show_text = final_name

            # --- Vẽ nền chữ để dễ đọc ---
            text_size = cv2.getTextSize(show_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, show_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # --- Log và lưu ---
            if final_name != "unknown":
                now = time.time()
                last_time = self.last_logged_time.get(final_name, 0)
                
                # Kiểm tra ngưỡng đồng thuận và cooldown
                if self.voter.get_consensus(face_id)[0] == final_name and now - last_time >= self.cooldown_seconds:
                    self.io_q.put({"type": "log", "name": final_name})
                    # THAY ĐỔI: Gửi TOÀN KHUNG HÌNH (frame) với loại task là "save_full"
                    self.io_q.put({"type": "save_full", "frame": frame.copy(), "name": final_name}) 
                    self.last_logged_time[final_name] = now

        return frame