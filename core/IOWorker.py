import os, cv2, queue, pandas as pd
import threading
from datetime import datetime
class IOWorker(threading.Thread):
        def __init__(self, q, log_file="recognition_log.csv"):
            super().__init__(daemon=True)
            self.q = q
            self.log_file = log_file
            if not os.path.exists(self.log_file):
                pd.DataFrame(columns=["timestamp", "name", "class"]).to_csv(self.log_file, index=False)

        def run(self):
            while True:
                try:
                    task = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    if task is None:
                        break

                    if task["type"] == "save":
                        frame, box, name = task["frame"], task["box"], task["name"]
                        x1, y1, x2, y2 = box
                        crop = frame[y1:y2, x1:x2]
                        folder = "captures/known" if name != "unknown" else "captures/unknown"
                        os.makedirs(folder, exist_ok=True)
                        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(folder, filename), crop)
                    elif task["type"] == "save_full":
                        frame, name = task["frame"], task["name"]
                        
                        # Tạo thư mục mới để lưu ảnh toàn khung hình
                        folder = "captures/full_frame"
                        os.makedirs(folder, exist_ok=True)
                        
                        # Tên file bao gồm tên người và dấu thời gian
                        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_full.jpg" 
                        cv2.imwrite(os.path.join(folder, filename), frame)
                    elif task["type"] == "log":
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        name = task["name"]
                        if "_" in name:
                            ten, lop = name.split("_", 1)
                        else:
                            ten, lop = name, ""

                        pd.DataFrame([{
                            "timestamp": ts,
                            "name": ten,
                            "class": lop
                        }]).to_csv(
                            self.log_file, mode="a", header=False, index=False
                        )

                except Exception as e:
                    print("[IOWorker error]", e)
                finally:
                    self.q.task_done()