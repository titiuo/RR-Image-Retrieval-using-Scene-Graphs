import json
import numpy as np
import pickle
import os
import random
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

# --- IMPORTS ---
# Assuming you saved the platt tool in platt.py
from platt import train_platt_params

# --- CONFIGURATION ---
VAL_ANNOTATIONS_PATH = '../../sg_dataset/splits/val_split.json'
GMM_MODEL_PATH = '../trained_models/binary_potentials.pkl'
OUTPUT_PATH = '../trained_models/platt_params_binary_potentials.pkl'
NEGATIVE_RATIO = 3  # For every 1 positive, generate 3 negatives

def extract_spatial_features(sub_bbox, obj_bbox):
    """ Same feature extractor as training """
    x, y, w, h = sub_bbox['x'], sub_bbox['y'], sub_bbox['w'], sub_bbox['h']
    xp, yp, wp, hp = obj_bbox['x'], obj_bbox['y'], obj_bbox['w'], obj_bbox['h']
    
    # Safety
    w, h = max(w, 1.0), max(h, 1.0)
    
    return [
        (x - xp) / w,
        (y - yp) / h,
        wp / w,
        hp / h
    ]

def collect_scores(val_path, models):
    """
    Runs the GMMs on the validation set to collect (score, label) pairs.
    """
    print(f"Loading validation data from {val_path}...")
    with open(val_path, 'r') as f:
        val_data = json.load(f)

    # Storage: key -> {'scores': [], 'labels': []}
    # Keys will be strings for Generic and tuples for Specific
    calibration_data = defaultdict(lambda: {'scores': [], 'labels': []})
    
    generic_gmms = models['generic']
    specific_gmms = models['specific']

    print("Mining positive and negative samples...")
    
    for img_idx, img in enumerate(val_data):
        if img_idx % 100 == 0: print(f"Processing image {img_idx}/{len(val_data)}")

        objects = img['objects']
        relationships = img['relationships']

        # 1. Map Objects
        obj_lookup = {}
        for idx, obj in enumerate(objects):
            obj_lookup[idx] = {'name': obj['names'][0], 'bbox': obj['bbox']}

        # 2. Identify POSITIVE pairs (Ground Truth)
        # Store as set of (sub_idx, obj_idx) -> rel_name
        gt_pairs = defaultdict(set)
        
        for rel in relationships:
            s_idx, o_idx = rel['objects']
            r_name = rel['relationship']
            gt_pairs[(s_idx, o_idx)].add(r_name)

            # --- PROCESS POSITIVE SAMPLE ---
            sub = obj_lookup[s_idx]
            obj = obj_lookup[o_idx]
            features = np.array(extract_spatial_features(sub['bbox'], obj['bbox'])).reshape(1, -1)

            # A. Score Generic Model
            if r_name in generic_gmms:
                score = generic_gmms[r_name].score_samples(features)[0]
                calibration_data[r_name]['scores'].append(score)
                calibration_data[r_name]['labels'].append(1) # Positive Label

            # B. Score Specific Model (if exists)
            triplet = (sub['name'], r_name, obj['name'])
            if triplet in specific_gmms:
                score = specific_gmms[triplet].score_samples(features)[0]
                calibration_data[triplet]['scores'].append(score)
                calibration_data[triplet]['labels'].append(1)

        # 3. Identify NEGATIVE pairs (Mining)
        # We need negatives to train the Logistic Regression (otherwise it learns P=1 always)
        
        # Get all possible indices
        all_indices = list(obj_lookup.keys())
        if len(all_indices) < 2: continue

        # Iterate all pairs to find valid negatives
        # (For efficiency in large images, you might want to random sample, 
        # but full iteration is safer for coverage)
        for s_idx in all_indices:
            for o_idx in all_indices:
                if s_idx == o_idx: continue
                
                # Retrieve actual relations for this pair (if any)
                true_relations = gt_pairs.get((s_idx, o_idx), set())
                
                sub = obj_lookup[s_idx]
                obj = obj_lookup[o_idx]
                features = np.array(extract_spatial_features(sub['bbox'], obj['bbox'])).reshape(1, -1)
                
                # --- STRATEGY: RANDOM NEGATIVES ---
                # We can't test against ALL models (too slow). 
                # We randomly pick a few Generic models to test this pair against.
                # If the pair is NOT related by 'riding', it's a negative for 'riding'.
                
                # Select a random subset of generic models to generate negatives for
                # (e.g., check 5 random relations per pair)
                sampled_rels = random.sample(list(generic_gmms.keys()), min(5, len(generic_gmms)))
                
                for r_target in sampled_rels:
                    if r_target not in true_relations:
                        # This pair is NOT 'r_target', so it's a negative example
                        score = generic_gmms[r_target].score_samples(features)[0]
                        calibration_data[r_target]['scores'].append(score)
                        calibration_data[r_target]['labels'].append(0) # Negative Label

                # Also try to generate specific negatives?
                # Usually Generic calibration is sufficient to approximate the Specific curve shape,
                # but if we have specific data, we should try.
                # For now, let's stick to rigorous Generic calibration.
                # Specific calibration often falls back to Generic params if data is scarce.

    return calibration_data

def train_and_save(cal_data):
    """
    Trains (A, B) using a strict 1:3 Positive/Negative ratio.
    """
    platt_params = {}
    
    print("\n--- Training Platt Parameters (Logistic Regression) ---")
    
    for key, data in cal_data.items():
        # Convert to numpy arrays
        scores = np.array(data['scores'])
        labels = np.array(data['labels'])
        
        # 1. Separate Positives and Negatives
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        # 2. Skip if insufficient data
        if n_pos < 2 or n_neg < 2:
            # We need at least 2 of each to draw a curve
            continue

        # 3. Apply 1:3 Ratio (Hard Negative Mining)
        # We want at most (n_pos * 3) negatives
        n_neg_keep = min(n_neg, n_pos * NEGATIVE_RATIO)
        
        # Randomly select the negatives to keep
        if n_neg > n_neg_keep:
            np.random.seed(42) # Ensure reproducibility
            keep_neg_indices = np.random.choice(neg_indices, size=n_neg_keep, replace=False)
        else:
            keep_neg_indices = neg_indices

        # 4. Combine
        final_indices = np.concatenate([pos_indices, keep_neg_indices])
        X = scores[final_indices]
        y = labels[final_indices]
        
        # 5. Train
        try:
            A, B = train_platt_params(X, y)
            platt_params[key] = (A, B)
        except Exception as e:
            print(f"Failed to train for {key}: {e}")

    print(f"Calibrated {len(platt_params)} models.")
    return platt_params

if __name__ == "__main__":
    # 1. Load GMMs
    print(f"Loading GMMs from {GMM_MODEL_PATH}...")
    with open(GMM_MODEL_PATH, 'rb') as f:
        models = pickle.load(f)

    # 2. Collect Scores
    data = collect_scores(VAL_ANNOTATIONS_PATH, models)

    # 3. Train
    params = train_and_save(data)

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(params, f)

    print(f"Saved Platt parameters to {OUTPUT_PATH}")