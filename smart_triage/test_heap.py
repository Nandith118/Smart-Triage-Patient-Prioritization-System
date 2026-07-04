import unittest
import time
import random
from main import PatientQueue

class TestTriageHeapEfficiency(unittest.TestCase):

    def test_heap_sorting_logic(self):
        pq = PatientQueue()
        
        pq.insert({"name": "Low Priority", "score": 10.5, "vitals": ""})
        pq.insert({"name": "Critical Patient", "score": 99.1, "vitals": ""})
        pq.insert({"name": "Medium Priority", "score": 50.0, "vitals": ""})
        
        self.assertEqual(pq.heap[0]["name"], "Critical Patient")

    def test_algorithmic_scale_proof(self):
        pq = PatientQueue()
        num_patients = 100000
        
        for i in range(num_patients):
            score = random.uniform(0, 95)
            pq.insert({"name": f"Patient_{i}", "score": score, "vitals": ""})
            
        new_patient_data = {"name": "CRITICAL_BOOST", "score": 100.0, "vitals": ""}
        
        start_time = time.perf_counter()
        pq.insert(new_patient_data)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"\nCustom Max-Heap Insertion Time (100k nodes): {execution_time_ms:.5f} ms")
        
        self.assertLess(execution_time_ms, 1.0)
        self.assertEqual(pq.heap[0]["name"], "CRITICAL_BOOST")

if __name__ == "__main__":
    unittest.main()