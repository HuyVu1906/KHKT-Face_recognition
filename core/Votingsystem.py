import time
import numpy as np
from collections import deque, Counter
class VotingSystem:
    def __init__(self, vote_threshold=5):
        self.vote_threshold = vote_threshold
        self.vote_history = {}
    
    def vote(self, face_id, name, confidence):
        if face_id not in self.vote_history:
            self.vote_history[face_id] = deque(maxlen=15)
        self.vote_history[face_id].append((name, confidence, time.time()))
        now = time.time()
        self.vote_history[face_id] = deque(
            [v for v in self.vote_history[face_id] if now - v[2] < 5],
            maxlen=15
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