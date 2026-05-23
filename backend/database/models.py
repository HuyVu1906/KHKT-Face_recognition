from sqlalchemy import Column, Integer, String
from .db import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)

    student_code = Column(String, unique=True)

    full_name = Column(String)

    class_name = Column(String)

    # Tên AI trả về
    face_label = Column(String, unique=True)

    phone = Column(String)

    parent_phone = Column(String)