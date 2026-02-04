import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hashlib

def get_image_signature(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    # Very small resize to create a "visual fingerprint"
    img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    return img.flatten()

def audit_leakage(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Use original filenames if possible (last part of source or path)
    # But path is processed, so we don't have original filename.
    
    train_df = df[df['split'] == 'train'].sample(min(2000, len(df[df['split'] == 'train'])))
    test_df = df[df['split'] == 'test'].sample(min(500, len(df[df['split'] == 'test'])))
    
    print(f"Auditing {len(train_df)} train vs {len(test_df)} test samples...")
    
    train_sigs = []
    for p in tqdm(train_df['path'], desc="Indexing Train"):
        sig = get_image_signature(Path(p))
        if sig is not None: train_sigs.append(sig)
    
    train_sigs = np.array(train_sigs)
    
    leaks = 0
    for p in tqdm(test_df['path'], desc="Checking Test"):
        sig = get_image_signature(Path(p))
        if sig is None: continue
        
        # Check Euclidean distance to all train sigs
        dists = np.linalg.norm(train_sigs - sig, axis=1)
        if np.min(dists) < 50: # Arbitrary threshold for "very similar"
            leaks += 1
            
    print(f"\nPotential Leaks found: {leaks} / {len(test_df)} sampled")
    print(f"Leakage Rate: {leaks / len(test_df) * 100:.2f}%")

if __name__ == "__main__":
    audit_leakage(Path("data/processed/metadata.csv"))
