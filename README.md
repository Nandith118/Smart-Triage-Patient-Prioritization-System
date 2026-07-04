# Smart-Triage-Patient-Prioritization-System
An AI powered emergency room triage system that scores incoming patients from raw vitals using a trained machine learning model, then maintains a live, always sorted priority queue using a custom built binary max heap. Exposed as a FastAPI backend, tested directly via API requests.

Benchmarked against a naive array based approach with re-sorting, the binary max heap achieves around *36.4x speedup* on sequential patient insertions (0.0267s → 0.0007s for 1,000 insertions), confirming that its O(log n) design holds up under real, measured load rather than staying purely theoretical.

## Why This Exists

In a real ER, patients are usually queued by arrival time, which does not reflect actual medical urgency. This project explores replacing that with an automated first pass triage score computed the moment a patient's vitals are entered, so the most critical patient is always identifiable first, regardless of when they walked in.

Two problems needed solving:

1. *Turning raw vitals into one comparable number.* Age, heart rate, oxygen saturation, pain score, and blood pressure do not mean much side by side. They needed to collapse into a single Urgency Score that can be directly compared across patients.
2. *Maintaining a constantly changing priority order efficiently.* Patients arrive at unpredictable times. Re sorting a full waiting list on every new arrival does not scale. Insertion needed to place a new patient into the correct priority position without touching every existing patient.

## System Workflow

1. *Kiosk input (API request):* patient vitals (age, heart rate, oxygen saturation, pain score, systolic blood pressure) are sent to the backend via a POST request to /api/admit-patient.
2. *AI risk prediction (inference stage):* a trained Random Forest Regressor converts these vitals into a single Urgency Score.
3. *Binary max heap insertion:* the new patient, tagged with their Urgency Score, is inserted into a custom built binary max heap, which bubbles them upward until the heap property is restored.
4. *Live priority response:* the API returns the newly admitted patient, the total number waiting, and the current highest priority patient to call next.

## Architecture

### 1. Urgency Scoring Model
A Random Forest Regressor is trained on simulated historical ER admission data (age, heart rate, oxygen saturation, pain score, systolic blood pressure) to predict a continuous Urgency Score from 0 to 100. The training target itself is derived from weighted clinical risk logic, for example low oxygen saturation and abnormal heart rate both increase the score, and age above 65 applies a multiplier, so the model learns reasoning similar to a clinician's first pass judgment.

### 2. Custom Binary Max Heap
Patients are stored in a binary max heap keyed on their Urgency Score, implemented from scratch with standard array based parent and child indexing:

- *Insertion:* a new patient is appended to the end of the heap array, then bubbled upward by repeatedly comparing against its parent and swapping while its score is higher. This keeps insertion at O(log n).
- *Live lineup view:* for display purposes, the current heap contents are sorted before being returned. This keeps the heap's O(log n) insertion guarantee intact for the write path, while accepting a one time O(n log n) cost only when a human actually needs to view the full ordered lineup, rather than on every single insertion.

### 3. FastAPI Serving Layer
A single POST endpoint, /api/admit-patient, accepts kiosk style vitals through a Pydantic schema, runs them through the trained model, inserts the result into the shared in memory heap, and returns the admitted patient's score alongside the current next patient to call. CORS is open for direct API testing without a frontend.

## Benchmark Results

To validate that the binary max heap actually earns its place over a simpler approach, insertion performance was benchmarked against a standard array with re-sorting, using 1,000 sequential insertions.

| Approach | Time (1,000 insertions) |
|---|---|
| Standard array with re-sorting | 0.0267 seconds |
| Binary max heap (this project) | 0.0007 seconds |
| *Improvement* | *36.4x faster* |

Benchmarking Performance Proof :

<img width="878" height="232" alt="image" src="https://github.com/user-attachments/assets/f6ba4dcb-5396-492a-a074-1bff64ac783a" />

*Takeaway:* as the number of waiting patients grows, re-sorting the entire list on every new arrival becomes increasingly expensive, while the heap's O(log n) insertion stays cheap. At 1,000 insertions this difference is already a 36.4x gap, and it widens further at larger scale.


## Example Run

<img width="1090" height="552" alt="Screenshot 2026-07-04 094823" src="https://github.com/user-attachments/assets/83bcb991-9648-43ca-9e86-bd3471096e89" />


Sample API response after admitting Alex:

<img width="692" height="303" alt="Screenshot 2026-07-04 094943" src="https://github.com/user-attachments/assets/86113cb2-bbd4-4abd-bada-5cf969bed819" />


Despite Ram arriving first, both Sam and Alex are correctly prioritized ahead of him based on Urgency Score, exactly the behavior a manually sorted arrival queue would fail to capture.

## Tech Stack

- *Language:* Python
- *Backend:* FastAPI, Pydantic
- *Machine Learning:* scikit learn (RandomForestRegressor)
- *Data Handling:* NumPy, pandas
- *Core Data Structure:* binary max heap, implemented from scratch with array based indexing

## Setup Instructions

```bash
git clone https://github.com/Nandith118/Smart-Triage-Patient-Prioritization-System.git
cd max-heap-er-triage-api
pip install fastapi uvicorn numpy pandas scikit-learn
python train_triage_ai.py
uvicorn main:app --reload
```


Once running, send a POST request to http://127.0.0.1:8000/api/admit-patient with a JSON body matching the KioskInput schema (patient_name, age, heart_rate, oxygen_sat, pain_score, systolic_bp) using any API client.

## Key Engineering Decisions

- Chose a Random Forest over a simpler linear model because urgency is not linear across vitals. For example, oxygen saturation only becomes dangerous below a threshold, which a tree based model captures naturally.
- Implemented the binary max heap from scratch rather than using a library heap, to keep full control over the comparison key (Urgency Score) and to demonstrate the underlying array indexing and bubble up mechanics directly.
- Separated the write path (O(log n) heap insertion) from the read path (sorted view for display), rather than forcing every insertion to maintain a fully sorted list, since sorting is only needed when a human looks at the lineup.
- Simulated the training target as a weighted function of real clinical risk factors (oxygen drop, heart rate deviation, blood pressure spikes, pain score, age) rather than random labels, so the model has something clinically reasonable to learn from.
- Kept the system backend only, since the goal was to validate the scoring and queuing logic itself, not build a UI around it.
