import json
import numpy as np
import pickle
import os
from sklearn.mixture import GaussianMixture
from collections import defaultdict

# --- CONFIGURATION ---
TRAIN_ANNOTATIONS_PATH = '../../sg_dataset/splits/train_split.json'
OUTPUT_PATH = '../trained_models/binary_potentials.pkl'
MIN_INSTANCES_SPECIFIC = 30  # Paper constraint 
GMM_COMPONENTS = 6           # Standard heuristic (Paper doesn't specify k, 4-8 is typical) but not mentioned in paper

def extract_spatial_features(sub_bbox, obj_bbox):
    """
    Implements Eq (4) from the paper.
    f(sub, obj) = ( (x-x')/w, (y-y')/h, w'/w, h'/h )
    """
    # Unpack boxes: [x, y, w, h]
    x, y, w, h = sub_bbox['x'], sub_bbox['y'], sub_bbox['w'], sub_bbox['h']
    xp, yp, wp, hp = obj_bbox['x'], obj_bbox['y'], obj_bbox['w'], obj_bbox['h']

    # Safety: Avoid division by zero
    w = max(w, 1.0)
    h = max(h, 1.0)

    feat_1 = (x - xp) / w
    feat_2 = (y - yp) / h
    feat_3 = wp / w
    feat_4 = hp / h

    return [feat_1, feat_2, feat_3, feat_4]

def load_and_extract_features(json_path):
    """
    Parses the dataset and buckets features into 'Specific' and 'Generic' bins.
    """
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Buckets to store feature vectors (N, 4)
    # Keys: (sub_name, rel_name, obj_name) -> List of lists
    specific_buckets = defaultdict(list)
    
    # Keys: rel_name -> List of lists
    generic_buckets = defaultdict(list)

    count_rels = 0
    
    for img_data in data:
        objects = img_data['objects']
        relationships = img_data['relationships']
        
        # Map object index -> Object Data (Name + Bbox)
        # Note: The dataset might have multiple names, we take the first one as class
        obj_lookup = {}
        for idx, obj in enumerate(objects):
            # We need the class name (string) and the bbox
            obj_lookup[idx] = {
                'name': obj['names'][0], 
                'bbox': obj['bbox']
            }

        for rel in relationships:
            sub_idx = rel['objects'][0]
            obj_idx = rel['objects'][1]
            rel_name = rel['relationship']

            subject = obj_lookup[sub_idx]
            target = obj_lookup[obj_idx]

            # 1. Extract Features
            feats = extract_spatial_features(subject['bbox'], target['bbox'])

            # 2. Store for Specific Model: P(f | c, r, c')
            triplet_key = (subject['name'], rel_name, target['name'])
            specific_buckets[triplet_key].append(feats)

            # 3. Store for Generic Model: P(f | r)
            generic_buckets[rel_name].append(feats)
            
            count_rels += 1

    print(f"Processed {len(data)} images.")
    print(f"Extracted features for {count_rels} relationship instances.")
    return specific_buckets, generic_buckets

def train_gmms(specific_data, generic_data):
    """
    Trains GMMs according to the '30 instances' rule.
    """
    trained_models = {
        'specific': {},
        'generic': {}
    }

    generic_skip_count = 0

    print("\n--- Training Generic Models P(f|r) ---")
    for rel_name, features in generic_data.items():
        X = np.array(features)

        if len(X) < 2:
            print(f"Skipping generic GMM for '{rel_name}' due to insufficient data ({len(X)} samples).")
            generic_skip_count += 1
            continue

        gmm = GaussianMixture(n_components=min(GMM_COMPONENTS, len(X)), covariance_type='diag', random_state=42)        # Maybe different from the paper
        gmm.fit(X)
        trained_models['generic'][rel_name] = gmm
        # print(f"Trained generic GMM for '{rel_name}' with {len(X)} samples.")

    print(f"Total Generic Models: {len(trained_models['generic'])}")
    print(f"Skipped Generic Relations (<2 instances): {generic_skip_count}")

    print("\n--- Training Specific Models P(f|c,r,c') ---")
    count_trained = 0
    count_skipped = 0
    
    for triplet_key, features in specific_data.items():
        X = np.array(features)
        
        # STRICT RULE: "If there are fewer than 30 instances... fall back" 
        if len(X) >= MIN_INSTANCES_SPECIFIC:
            gmm = GaussianMixture(n_components=min(GMM_COMPONENTS, len(X)), covariance_type='diag', random_state=42)
            gmm.fit(X)
            trained_models['specific'][triplet_key] = gmm
            count_trained += 1
        else:
            count_skipped += 1

    print(f"Total Specific Models: {count_trained}")
    print(f"Skipped Triplets (<30 instances): {count_skipped}")

    return trained_models

if __name__ == "__main__":
    # 1. Extract
    specific_feats, generic_feats = load_and_extract_features(TRAIN_ANNOTATIONS_PATH)

    # 2. Train
    models = train_gmms(specific_feats, generic_feats)

    # 3. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\nSuccess! Models saved to {OUTPUT_PATH}")
    print("Structure: {'specific': {(sub,rel,obj): GMM}, 'generic': {rel: GMM}}")