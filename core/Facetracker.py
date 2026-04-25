from collections import deque
class FaceTracker:
    def __init__(self, max_history=10):
        self.face_history = {}
        self.max_history = max_history
        self.next_face_id = 0
    
    def assign_face_id(self, bbox):
        for face_id, history in self.face_history.items():
            last_bbox = history[-1] if history else None
            if last_bbox and self.calculate_iou(bbox, last_bbox) > 0.25:
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