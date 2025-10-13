import cv2
import face_recognition
import numpy as np
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
        # Lấy trung bình encoding (tăng độ ổn định)
        mean_encoding = np.mean(encodings, axis=0)
        known_encodings.append(mean_encoding)
        known_names.append(person_name)

print(f"\n🧠 Tổng số người đã học: {len(known_names)}\n")


# ==============================
# 2️⃣ NHẬN DIỆN TRỰC TIẾP TỪ WEBCAM
# ==============================
cap = cv2.VideoCapture(0)
print("🎥 Đang bật camera... Nhấn 'q' để thoát.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "idk"
        color = (0, 0, 255)

        if len(known_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                color = (0, 255, 0)

        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("AI by Thanh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
