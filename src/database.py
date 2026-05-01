import os
import numpy as np
from src.core import calculate_hamming_distance

class IrisDatabase:
    def __init__(self, db_path="db"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
    def enroll_user(self, user_id, iris_code, mask):
        """Enroll a new user by saving their IrisCode and Mask."""
        file_path = os.path.join(self.db_path, f"{user_id}.npz")
        np.savez_compressed(file_path, iris_code=iris_code, mask=mask)
        return file_path
        
    def identify_user(self, new_code, new_mask, threshold=0.32):
        """Search the database for the closest match."""
        best_match = None
        min_dist = float('inf')
        
        for filename in os.listdir(self.db_path):
            if filename.endswith(".npz"):
                user_id = os.path.splitext(filename)[0]
                file_path = os.path.join(self.db_path, filename)
                
                try:
                    data = np.load(file_path)
                    saved_code = data['iris_code']
                    saved_mask = data['mask']
                    
                    dist = calculate_hamming_distance(new_code, new_mask, saved_code, saved_mask)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = user_id
                except Exception as e:
                    print(f"[-] Error reading data for user {user_id}: {e}")
                    
        if best_match is not None and min_dist < threshold:
            return best_match, min_dist
        else:
            return None, min_dist
