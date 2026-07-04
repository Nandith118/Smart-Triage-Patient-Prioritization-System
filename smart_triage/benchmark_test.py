# benchmark_test.py
import time
import random

# standard approach (array + full re-sorting)
def standard_array_insert(array, new_patient):
    array.append(new_patient)
    # Simulates sorting the entire queue by priority score on every insertion
    array.sort(key=lambda x: x['score'], reverse=True) 

# Binary Max-Heap approach
class BinaryMaxHeap:
    def __init__(self):
        self.heap = []
    
    def insert(self, patient):
        self.heap.append(patient)
        self._bubble_up(len(self.heap) - 1)
        
    def _bubble_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.heap[index]['score'] > self.heap[parent]['score']:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._bubble_up(parent)

if __name__ == "__main__":
    # (1000 patients arriving sequentially)
    SCALE_SIZE = 1000
    mock_dataset = [{"name": f"Patient_{i}", "score": random.uniform(0, 100)} for i in range(SCALE_SIZE)]
    
    print(f"📊 Benchmarking Performance Proof (for {SCALE_SIZE} Sequential Insertions) \n")
    
    # standard array
    standard_queue = []
    start_time = time.time()
    for patient in mock_dataset:
        standard_array_insert(standard_queue, patient)
    standard_duration = time.time() - start_time
    print(f"standard array with re-sorting: {standard_duration:.4f} seconds")
    
    # binary max-heap
    heap_queue = BinaryMaxHeap()
    start_time = time.time()
    for patient in mock_dataset:
        heap_queue.insert(patient)
    heap_duration = time.time() - start_time
    print(f"binary max-heap:  {heap_duration:.4f} seconds")
    
    # exact improvement factor
    speedup = standard_duration / heap_duration
    print(f"\nRESULT: binary max-heap architecture is {speedup:.1f}x faster at scale!")