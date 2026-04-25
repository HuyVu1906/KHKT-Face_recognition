from core.AdvancedFaceRecognitionSystem import AdvancedFaceRecognitionSystem
import cv2, threading, queue, os, numpy as np

def main():
    system = AdvancedFaceRecognitionSystem()
    resource_dir = r"E:\du_an\KHKT\khkt\resources"
    db_path = os.path.join(resource_dir, "face_database.pkl")
    known_dir = os.path.join(resource_dir, "known_faces")

    # --- Xây dựng hoặc tải database ---
    if not os.path.exists(db_path):
        system.database.database_path = db_path
        system.build_database_from_images(known_dir)
        system.build_ann_index()
    else:
        print("📂 Database có sẵn, đang tải...")
        system.database.load_database()
        system.faiss_index.build_index(system.database.known_encodings, system.database.known_names)

    # --- Biến toàn cục cho thread ---
    stop_evt = threading.Event()
    frame_q = queue.Queue(maxsize=2)

    # --- Luồng đọc camera ---
    def camera_thread():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không mở được camera.")
            stop_evt.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            if frame_q.full():
                try:
                    _ = frame_q.get_nowait()
                except queue.Empty:
                    pass
            frame_q.put(frame)

        cap.release()

    # --- Luồng nhận diện khuôn mặt ---
    def recognition_thread():
        while not stop_evt.is_set():
            try:
                frame = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            processed = system.process_frame(frame)
            cv2.imshow("AI Face Recognition", processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_evt.set()
            elif key == ord('a'):
                print("🔄 Đang quét lại thư mục known_faces để tìm người mới...")
                system.rescan_known_faces(known_dir)
                system.build_ann_index()

        cv2.destroyAllWindows()

    # --- Khởi chạy 2 luồng ---
    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=recognition_thread, daemon=True)

    print("🎥 Đang khởi động camera...")
    print("""
    🎮 Phím điều khiển:
    [q] hoặc [ESC] - Thoát chương trình
    [a] - Quét lại thư mục known_faces để cập nhật người mới""")
    t1.start()
    t2.start()

    # --- Chờ luồng nhận diện kết thúc ---
    t2.join()
    stop_evt.set()

    # --- Dọn dẹp an toàn ---
    cv2.destroyAllWindows()
    system.database.save_database()
    print("✅ Đã lưu database & thoát an toàn.")

if __name__ == "__main__":
    main()
