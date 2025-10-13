import cv2
import face_recognition
import numpy as np
import mediapipe as mp
import os

# ==============================
# 1️⃣ ĐỌC ẢNH TỪ CÁC THƯ MỤC CON
# ==============================
known_encodings = []
known_names = []

main_dir = r"D:\python\nhận diện khuôn mặt\known_faces"

print("📂 Đang tải dữ liệu khuôn mặt...\n")

for person_name in os.listdir(main_dir):
    person_folder = os.path.join(main_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    encodings = []
    for filename in os.listdir(person_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(person_folder, filename)
            image = face_recognition.load_image_file(file_path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                print(f"✅ {person_name} - đã thêm ảnh: {filename}")
            else:
                print(f"⚠️ Không phát hiện khuôn mặt trong {filename}")

    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        known_encodings.append(mean_encoding)
        known_names.append(person_name)

print(f"\n🧠 Tổng số người đã học: {len(known_names)}\n")


# ==============================
# 2️⃣ KHỞI TẠO MEDIAPIPE FACE DETECTION
# ==============================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Model 0: nhanh, tối ưu webcam / Model 1: chính xác hơn (ảnh lớn)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# ==============================
# 3️⃣ NHẬN DIỆN TRỰC TIẾP
# ==============================
cap = cv2.VideoCapture(0)
print("🎥 Đang bật camera... Nhấn 'q' để thoát.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            # Giới hạn hợp lệ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Cắt vùng khuôn mặt (tăng thêm biên một chút)
            pad = 20
            face_crop = rgb_frame[max(0, y1 - pad):min(h, y2 + pad),
                                max(0, x1 - pad):min(w, x2 + pad)]

            name = "unknown"
            color = (0, 0, 255)

            if face_crop.size > 0:
                # Lấy vị trí khuôn mặt trong vùng crop
                sub_locations = face_recognition.face_locations(face_crop, model='hog')

                if sub_locations:
                    if face_crop.size > 0:
                        # Đảm bảo ảnh là RGB, uint8, và có 3 kênh
                        if face_crop.ndim == 3 and face_crop.shape[2] == 3:
                            face_crop = face_crop.astype(np.uint8)
                            # Kiểm tra kích thước khuôn mặt đủ lớn
                            if face_crop.shape[0] > 40 and face_crop.shape[1] > 40:
                                try:
                                    encodings = face_recognition.face_encodings(face_crop)
                                    if encodings:
                                        face_encoding = encodings[0]
                                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                                except Exception as e:
                                    print("Encoding error:", e)
                            else:
                                print("Face too small, skipping.")
                        else:
                            print("Invalid face_crop shape:", face_crop.shape)
                    else:
                        print("Empty face_crop.")
                    encodings = face_recognition.face_encodings(face_crop)
                    if encodings:
                        face_encoding = encodings[0]
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if len(face_distances) > 0 and matches[best_match_index]:
                            name = known_names[best_match_index]
                            color = (0, 255, 0)

            # Vẽ khung và tên
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("AI by Thanh ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
