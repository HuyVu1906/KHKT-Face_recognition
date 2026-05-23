from fastapi import FastAPI, UploadFile, File
from backend.database.db import engine
from backend.database.models import Base, Student
import numpy as np
import cv2

from core.AdvancedFaceRecognitionSystem import AdvancedFaceRecognitionSystem

app = FastAPI()
Base.metadata.create_all(bind=engine)
from sqlalchemy.orm import Session
from fastapi import Depends
from backend.database.db import SessionLocal
system = AdvancedFaceRecognitionSystem()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.on_event("startup")
def load_model():
    print("🚀 Loading model...")
    system.database.load_database()
    system.faiss_index.build_index(
        system.database.known_encodings,
        system.database.known_names
    )

# API nhận diện khuôn mặt
@app.post("/recognize-face")
async def recognize_face(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    contents = await file.read()

    # bytes -> image
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Không đọc được ảnh"}

    ai_results = system.recognize_image(img)

    if len(ai_results) == 0:
        return {
            "result": "Không phát hiện khuôn mặt"
        }

    results = []

    for item in ai_results:

        name = item["name"]
        score = item["score"]

        # tìm học sinh theo face_label
        student = db.query(Student).filter(
            Student.face_label == name
        ).first()

        if student:
            results.append({
                "student_code": student.student_code,
                "full_name": student.full_name,
                "class_name": student.class_name,
                "phone": student.phone,
                "parent_phone": student.parent_phone,
                "score": float(score)
            })
        else:
            results.append({
                "face_label": name,
                "score": float(score),
                "message": "Chưa có hồ sơ học sinh"
            })

    return {"faces": results}
@app.post("/students")
def create_student(
    student_code: str,
    full_name: str,
    class_name: str,
    face_label: str,
    phone: str = "",
    parent_phone: str = "",
    db: Session = Depends(get_db)
):
    student = Student(
        student_code=student_code,
        full_name=full_name,
        class_name=class_name,
        face_label=face_label,
        phone=phone,
        parent_phone=parent_phone
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return {
        "message": "Student created",
        "id": student.id
    }
@app.get("/students")
def get_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()

    return [
        {
            "id": s.id,
            "student_code": s.student_code,
            "full_name": s.full_name,
            "class_name": s.class_name
        }
        for s in students
    ]
@app.get("/students/{student_id}")
def get_student(student_id: int,
                db: Session = Depends(get_db)):

    student = db.query(Student).filter(
        Student.id == student_id
    ).first()

    if not student:
        return {"error": "Không tìm thấy học sinh"}

    return {
        "id": student.id,
        "student_code": student.student_code,
        "full_name": student.full_name,
        "class_name": student.class_name,
        "phone": student.phone,
        "parent_phone": student.parent_phone
    }