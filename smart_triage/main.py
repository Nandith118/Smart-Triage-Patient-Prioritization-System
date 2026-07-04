from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="AI-Powered Emergency Room Triage Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# loading trained AI model
try:
    with open('triage_model.pkl', 'rb') as f:
        ai_model = pickle.load(f)
    print("AI Triage loaded successfully!")
except FileNotFoundError:
    print(" 'triage_model.pkl' missing. Run train_triage_ai.py first.")
    ai_model = None


# BINARY MAX-HEAP
class PatientQueue:
    def __init__(self):
        # The heap array holds dictionaries: {"name": str, "score": float}
        self.heap = []

    def get_parent_index(self, i): return (i - 1) // 2
    def get_left_child_index(self, i): return (2 * i) + 1
    def get_right_child_index(self, i): return (2 * i) + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, patient):
        """Inserts a new patient and bubbles them up in O(log N) time"""
        self.heap.append(patient)
        self._bubble_up(len(self.heap) - 1)

    def _bubble_up(self, index):
        """The Leapfrog logic: Compare with parent and swap upwards"""
        while index > 0:
            parent_idx = self.get_parent_index(index)
            # if current patient urgency score is higher than their parent's, swap!
            if self.heap[index]["score"] > self.heap[parent_idx]["score"]:
                self.swap(index, parent_idx)
                index = parent_idx
            else:
                break

    def get_live_triage_list(self):
        # Return the underlying tree hierarchy ordered by current structural placement
        # For a true full drain extraction it would be O(N log N), but for live view 
        # we can show the heap structure or sort it quickly for display.
        return sorted(self.heap, key=lambda x: x["score"], reverse=True)


# initializing global active ER waiting room queue
er_waiting_room = PatientQueue()

# defining data that kiosk sends to the API
class KioskInput(BaseModel):
    patient_name: str
    age: int
    heart_rate: float
    oxygen_sat: float
    pain_score: int
    systolic_bp: float

@app.post("/api/admit-patient")
def admit_patient(data: KioskInput):
    if not ai_model:
        return {"error": "AI model not ready."}

    features = np.array([[data.age, data.heart_rate, data.oxygen_sat, data.pain_score, data.systolic_bp]])
    predicted_score = float(ai_model.predict(features)[0])

    # patient profile
    new_patient = {
        "name": data.patient_name,
        "score": round(predicted_score, 2),
        "vitals": f"O2: {data.oxygen_sat}%, HR: {data.heart_rate} bpm"
    }

    # performs O(log N) structural insertion into max-heap pyramid
    er_waiting_room.insert(new_patient)

    # fetching updated live triage lineup
    current_lineup = er_waiting_room.get_live_triage_list()

    print(f"\n ALERT: {new_patient['name']} admitted with AI Priority Score: {new_patient['score']}")
    print(" CURRENT LIVE ER PRIORITY LINEUP:")
    for rank, p in enumerate(current_lineup, 1):
        print(f"   Rank {rank}: {p['name']} (Score: {p['score']}) -> {p['vitals']}")

    return {
        "status": "Success",
        "admitted_patient": new_patient,
        "total_waiting": len(current_lineup),
        "next_patient_to_call": current_lineup[0] if current_lineup else None
    }