import ast
import csv
import json
import os
import pickle
import random
import time  # <--- NEW: For timing
from math import log

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

# --- IMPORTS ---
try:
    from CRF import train_unary 
    from graph import load_scene_graphs
except ImportError:
    print("ERROR: Run this script as a module: python -m CRF.test_pipeline")
    exit(1)

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'data'))
SG_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'sg_dataset'))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'model'))
TRAINED_DIR = os.path.join(ROOT_DIR, 'CRF', 'trained_models')

# Files
CSV_PATH = os.path.join(DATA_DIR, 'rcnn_training_cleaned.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'rcnn_testing_cleaned.csv')
TEST_ANNOTATIONS_PATH = os.path.join(SG_DIR, 'sg_test_annotations.json')
VAL_SPLIT_PATH = os.path.join(SG_DIR, 'splits', 'val_split.json')

# Models
UNARY_MODEL_PATH = os.path.join(TRAINED_DIR, 'unary_potentials.pkl')
BINARY_MODEL_PATH = os.path.join(TRAINED_DIR, 'binary_potentials.pkl')
BINARY_PLATT_PATH = os.path.join(TRAINED_DIR, 'platt_params_binary_potentials.pkl')
RCNN_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'rcnn_finetuned.pth')

# Image Roots
IMAGE_ROOT_CANDIDATES = [
    '/home/mamane/porject_data/sg_test_images', 
    os.path.join(SG_DIR, 'sg_train_images'),
    os.path.join(SG_DIR, 'sg_test_images'),
    os.path.join(SG_DIR, 'images'),
]

# --- CONFIGURATION ---
RUN_INFERENCE = os.getenv('RUN_INFERENCE', '1') == '1'
USE_FEATURE_CACHE = True  # Hardcode to True for this run
FEATURE_CACHE_PATH = os.getenv('FEATURE_CACHE_PATH', os.path.join(DATA_DIR, 'rcnn_test_features.pkl'))
MAX_PROPOSALS = int(os.getenv('MAX_PROPOSALS', '2000'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '128'))
GRAPH_INDEX = int(os.getenv('GRAPH_INDEX', '0'))
QUERY_SPLIT = os.getenv('QUERY_SPLIT', 'test') 
TOP_K = int(os.getenv('TOP_K', '10'))
MAX_IMAGES = int(os.getenv('MAX_IMAGES', '1000'))

# --- CLASS DEFINITIONS ---

class CRFInference:
    def __init__(self, unary_path, binary_path, binary_platt_path=None):
        print("Loading CRF models...")
        with open(unary_path, 'rb') as f:
            self.unaries = pickle.load(f)
        with open(binary_path, 'rb') as f:
            self.binaries = pickle.load(f)

        self.binary_platt = {}
        if binary_platt_path and os.path.exists(binary_platt_path):
            with open(binary_platt_path, 'rb') as f:
                self.binary_platt = pickle.load(f)

    def _compute_spatial_features_matrix(self, boxes):
        """
        Vectorized computation of spatial features for ALL pairs (N x N).
        Returns: (N, N, 4) matrix
        """
        N = len(boxes)
        boxes = np.array(boxes) # Ensure numpy (N, 4)
        
        # Expand dimensions to broadcast
        # sub: (N, 1, 4), obj: (1, N, 4)
        sub = boxes[:, np.newaxis, :] 
        obj = boxes[np.newaxis, :, :] 
        
        x, y, w, h = sub[..., 0], sub[..., 1], sub[..., 2], sub[..., 3]
        xp, yp, wp, hp = obj[..., 0], obj[..., 1], obj[..., 2], obj[..., 3]
        
        # Avoid division by zero
        w = np.maximum(w, 1.0)
        h = np.maximum(h, 1.0)
        
        # Features: [ (x-xp)/w, (y-yp)/h, wp/w, hp/h ]
        f1 = (x - xp) / w
        f2 = (y - yp) / h
        f3 = wp / w
        f4 = hp / h
        
        # Stack into (N, N, 4)
        feats = np.stack([f1, f2, f3, f4], axis=-1)
        return feats

    def beam_search(self, proposals, features, query_graph, beam_width=5):
        proposals = np.array(proposals)
        if isinstance(features, tuple):
            features = features[0]
        features = np.array(features)
        num_boxes = len(proposals)
        num_objects = len(query_graph['objects'])
        
        # --- STEP 1: PRECOMPUTE UNARY SCORES (Matrix Operation) ---
        # Result: (Num_Query_Objs, Num_Boxes)
        unary_matrix = np.full((num_objects, num_boxes), -10.0)
        
        for i, obj_name in enumerate(query_graph['objects']):
            if obj_name in self.unaries['svms']:
                svm = self.unaries['svms'][obj_name]
                # BATCH CALL: Score all 100 boxes at once
                raw_scores = svm.decision_function(features) 
                
                # BATCH PLATT SCALING
                A, B = self.unaries['platt'].get(obj_name, (-1.0, 0.0))
                logits = np.clip(A * raw_scores + B, -50, 50)
                probs = 1.0 / (1.0 + np.exp(-logits))
                unary_matrix[i] = np.log(np.maximum(probs, 1e-10))

        # --- STEP 2: PRECOMPUTE BINARY SCORES (Matrix Operation) ---
        # We only compute for relationships that actually exist in the query
        # Store as dict: (s_idx, o_idx) -> (N, N) score matrix
        binary_matrices = {}
        
        # Only compute geometry if we have relationships
        if query_graph['relationships']:
            # Calculate (N, N, 4) spatial features once
            spatial_feats = self._compute_spatial_features_matrix(proposals)
            # Flatten to (N*N, 4) for GMM batch scoring
            flat_feats = spatial_feats.reshape(-1, 4)
            
            for rel in query_graph['relationships']:
                s_idx, r_name, o_idx = rel
                sub_name = query_graph['objects'][s_idx]
                obj_name = query_graph['objects'][o_idx]
                triplet = (sub_name, r_name, obj_name)
                
                # Resolve Model
                gmm = None
                platt_key = None
                
                if triplet in self.binaries['specific']:
                    gmm = self.binaries['specific'][triplet]
                    platt_key = triplet
                elif r_name in self.binaries['generic']:
                    gmm = self.binaries['generic'][r_name]
                    platt_key = r_name
                
                if gmm:
                    # BATCH CALL: Score 10,000 pairs at once
                    log_density = gmm.score_samples(flat_feats)
                    
                    # Batch Platt
                    if platt_key in self.binary_platt:
                        A, B = self.binary_platt[platt_key]
                        logits = np.clip(A * log_density + B, -50, 50)
                        probs = 1.0 / (1.0 + np.exp(logits)) # Note: Binary Platt often uses positive A
                        scores = np.log(np.maximum(probs, 1e-10))
                    else:
                        scores = log_density
                    
                    # Reshape back to (N, N)
                    binary_matrices[(s_idx, o_idx)] = scores.reshape(num_boxes, num_boxes)
                else:
                    # Missing model penalty
                    binary_matrices[(s_idx, o_idx)] = np.full((num_boxes, num_boxes), -10.0)

        # --- STEP 3: LIGHTWEIGHT BEAM SEARCH (Array Lookups only) ---
        beam = [(0.0, [])] 

        for i in range(num_objects):
            new_beam = []
            
            # Get precomputed unary scores for this object (Array of size N)
            u_scores_i = unary_matrix[i] 

            for path_score, assignment in beam:
                # We want to extend this path with every possible box_idx for object 'i'
                # Candidates: All boxes (0..N-1)
                
                # Start with Unary Score
                # shape: (N,)
                current_scores = np.full(num_boxes, path_score) + u_scores_i
                
                # Add Binary Scores (Look backwards)
                for rel in query_graph['relationships']:
                    s_idx, r_name, o_idx = rel
                    
                    # Case A: We are Object (i), Subject (s_idx < i) is already assigned
                    if o_idx == i and s_idx < i:
                        prev_box_idx = assignment[s_idx]
                        # Lookup row 'prev_box_idx' in the precomputed matrix
                        # shape: (N,) -> scores for all possible 'curr_box'
                        b_scores = binary_matrices[(s_idx, o_idx)][prev_box_idx, :]
                        current_scores += b_scores
                        
                    # Case B: We are Subject (i), Object (o_idx < i) is already assigned
                    elif s_idx == i and o_idx < i:
                        prev_box_idx = assignment[o_idx]
                        # Lookup col 'prev_box_idx' (because we are subject, prev is object)
                        # shape: (N,)
                        b_scores = binary_matrices[(s_idx, o_idx)][:, prev_box_idx]
                        current_scores += b_scores

                # Create new paths
                # We now have 'current_scores' array of size N, where index is box_idx
                for box_idx, score in enumerate(current_scores):
                    new_beam.append((score, assignment + [box_idx]))

            # Prune
            if not new_beam:
                return (-9999.0, [])
            
            # Optimizing sort: use heapq.nlargest or partial sort if N is huge, 
            # but for N=100 * K=5, standard sort is fine.
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]

        return beam[0]

# --- HELPER FUNCTIONS ---

def normalize_box(box):
    if len(box) != 4: raise ValueError(f"Box error: {box}")
    x, y, w, h = box
    return (x, y, x + w, y + h)

def resolve_image_root(preferred_subdir=None):
    if preferred_subdir:
        preferred = os.path.join(SG_DIR, preferred_subdir)
        if os.path.isdir(preferred): return preferred
    for candidate in IMAGE_ROOT_CANDIDATES:
        if os.path.isdir(candidate): return candidate
    raise FileNotFoundError("No image root found.")

def build_query_from_graph(graph, obj_vocab, rel_vocab):
    objects = [obj_vocab.get_word(obj['class_id']) for obj in graph.objects]
    relationships = []
    for rel in graph.relationships:
        relationships.append([
            rel['subject_idx'],
            rel_vocab.get_word(rel['relation_id']),
            rel['object_idx']
        ])
    return {'objects': objects, 'relationships': relationships}

def extract_features(img_path, boxes):
    # Initializes R-CNN on demand
    train_unary.RCNN_WEIGHTS_PATH = RCNN_WEIGHTS_PATH
    train_unary._init_rcnn()
    device = train_unary._rcnn_device
    transform = train_unary._rcnn_transform

    feats = []
    forward_time = 0.0
    with Image.open(img_path).convert('RGB') as image:
        with torch.no_grad():
            for start in range(0, len(boxes), BATCH_SIZE):
                batch_boxes = boxes[start:start + BATCH_SIZE]
                batch_tensors = []
                for box in batch_boxes:
                    x1, y1, x2, y2 = normalize_box(box)
                    batch_tensors.append(transform(image.crop((x1, y1, x2, y2))))
                
                if not batch_tensors: continue
                batch = torch.stack(batch_tensors).to(device, non_blocking=True)
                t0 = time.time()
                out = train_unary._rcnn_model.backbone(batch).cpu().numpy()
                forward_time += time.time() - t0
                feats.append(out)

    if not feats:
        print('WARNING: No features extracted for image:', img_path)
        return np.zeros((0, 4096), dtype=np.float32), forward_time
    return np.vstack(feats), forward_time

def find_image_path(image_name):
    for candidate in IMAGE_ROOT_CANDIDATES:
        img_path = os.path.join(candidate, image_name)
        if os.path.exists(img_path): return img_path
    return None

def build_csv_index():
    print("Building CSV Index (Proposals)...")
    csv_paths = [TEST_CSV_PATH, CSV_PATH]
    index = {}
    for path in csv_paths:
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path, usecols=['image', 'box'])
        except ValueError:
            df = pd.read_csv(path)
        if df.empty: continue
        
        df['box'] = df['box'].apply(ast.literal_eval)
        for img, box in zip(df['image'], df['box']):
            if img not in index: index[img] = []
            index[img].append(box)
    return index

def load_feature_cache():
    if not USE_FEATURE_CACHE or not os.path.exists(FEATURE_CACHE_PATH): return None
    print(f"Loading features from pickle: {FEATURE_CACHE_PATH}")
    with open(FEATURE_CACHE_PATH, 'rb') as f:
        try:
            header = pickle.load(f)
        except EOFError: return None
        if isinstance(header, dict) and header.get('_stream') is True:
            cache = {}
            count = 0
            MAX_LOAD = 2000 
            while count < MAX_LOAD:
                try:
                    entry = pickle.load(f)
                    count += 1
                except EOFError: break
                if isinstance(entry, dict) and entry.get('image'):
                    cache[entry['image']] = {
                        'boxes': entry.get('boxes', []), 
                        'features': entry.get('features')
                    }
            return cache
        return header

def _load_query_and_gt_filename():
    if QUERY_SPLIT == 'test':
        graphs, obj_vocab, _, rel_vocab = load_scene_graphs(TEST_ANNOTATIONS_PATH)
    else:
        graphs, obj_vocab, _, rel_vocab = load_scene_graphs(VAL_SPLIT_PATH)

    idx = min(GRAPH_INDEX, len(graphs) - 1)
    target_graph = graphs[idx]
    
    print(f"Selected Query Graph Index: {idx}")
    print(f"Ground Truth Image ID: {target_graph.filename}")
    
    query = build_query_from_graph(target_graph, obj_vocab, rel_vocab)
    return query, target_graph.filename

def plot_results(query_filename, results, image_root):
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # Query
    ax_query = plt.subplot2grid((3, 5), (0, 2)) 
    query_path = find_image_path(query_filename)
    if query_path:
        ax_query.imshow(Image.open(query_path))
        ax_query.set_title(f"QUERY / TARGET\n{query_filename}", fontsize=14, color='blue', fontweight='bold')
    else:
        ax_query.text(0.5, 0.5, "Image Not Found", ha='center')
    ax_query.axis('off')

    # Results
    for i, (fname, score) in enumerate(results[:10]):
        row = 1 + (i // 5)
        col = i % 5
        ax = plt.subplot2grid((3, 5), (row, col))
        img_path = find_image_path(fname)
        if img_path:
            try: ax.imshow(Image.open(img_path))
            except Exception: ax.text(0.5, 0.5, "Load Error", ha='center')
        else: ax.text(0.5, 0.5, "Not Found", ha='center')
        
        is_match = (fname == query_filename)
        title = f"Rank {i+1}\nScore: {score:.2f}" + ("\n(CORRECT)" if is_match else "")
        ax.set_title(title, fontsize=11, color='green' if is_match else 'black', fontweight='bold' if is_match else 'normal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN SANITY CHECK LOGIC ---

def run_retrieval_sanity_check():
    test_graphs, _, _, _ = load_scene_graphs(TEST_ANNOTATIONS_PATH)
    query, gt_filename = _load_query_and_gt_filename()
    
    print(f"\nQUERY STRUCTURE: {query}")
    
    image_root = resolve_image_root()
    csv_index = build_csv_index()
    
    # Load Cache (Timed inside the load function)
    if not USE_FEATURE_CACHE:
        print("NOTE: Cache is DISABLED. R-CNN will run live.")
        feature_cache = {}
    else:
        feature_cache = load_feature_cache() or {}

    pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)
    results = []

    target_graphs = test_graphs[:MAX_IMAGES] if MAX_IMAGES > 0 else test_graphs
    gt_in_pool = any(g.filename == gt_filename for g in target_graphs)
    if not gt_in_pool:
        gt_graph_obj = next((g for g in test_graphs if g.filename == gt_filename), None)
        if gt_graph_obj: target_graphs.append(gt_graph_obj)

    print(f"\nScanning {len(target_graphs)} images...")

    # --- TIMING CONTAINERS ---
    times = {
        'cache_lookup': [],
        'rcnn_compute': [],
        'inference': [],
        'total_per_image': []
    }

    for idx, graph in enumerate(target_graphs):
        t_start_image = time.time()
        img_filename = graph.filename
        
        proposals = []
        features = None
        
        # 1. CACHE LOOKUP TIMING
        t_start_cache = time.time()
        found_in_cache = False
        
        if img_filename in feature_cache:
            proposals = feature_cache[img_filename]['boxes']
            features = feature_cache[img_filename]['features']
            found_in_cache = True
        elif img_filename in csv_index:
            proposals = csv_index[img_filename]
        
        t_end_cache = time.time()
        
        # Fallback Proposals
        if not proposals:
            for obj in graph.objects:
                bbox = obj['bbox']
                proposals.append([bbox['x'], bbox['y'], bbox['w'], bbox['h']])
        
        if not proposals: continue

        if MAX_PROPOSALS > 0:
            proposals = proposals[:MAX_PROPOSALS]
            if features is not None: features = features[:MAX_PROPOSALS]

        # 2. R-CNN COMPUTE TIMING (Only runs on cache miss)
        t_start_rcnn = time.time()
        if features is None:
            img_path = find_image_path(img_filename)
            if not img_path: continue
            try:
                features, _ = extract_features(img_path, proposals)
            except Exception: continue
        t_end_rcnn = time.time()
        
        if len(features) == 0: continue

        # 3. CRF INFERENCE TIMING
        t_start_infer = time.time()
        best_score, _ = pipeline.beam_search(proposals, features, query)
        t_end_infer = time.time()
        
        if best_score > -9000:
            results.append((img_filename, best_score))

        t_end_image = time.time()
        
        # Record stats
        times['cache_lookup'].append(t_end_cache - t_start_cache)
        if not found_in_cache:
             # Only log compute time if we actually computed
             times['rcnn_compute'].append(t_end_rcnn - t_start_rcnn)
        times['inference'].append(t_end_infer - t_start_infer)
        times['total_per_image'].append(t_end_image - t_start_image)

        if (idx+1) % 10 == 0: print(f"Scored {idx+1}/{len(target_graphs)}...")

    # --- PERFORMANCE REPORT ---
    print("\n" + "="*40)
    print("       PERFORMANCE REPORT       ")
    print("="*40)
    
    n_processed = len(times['total_per_image'])
    if n_processed > 0:
        avg_cache = sum(times['cache_lookup']) / n_processed
        
        n_rcnn = len(times['rcnn_compute'])
        avg_rcnn = sum(times['rcnn_compute']) / n_rcnn if n_rcnn > 0 else 0.0
        
        avg_infer = sum(times['inference']) / n_processed
        avg_total = sum(times['total_per_image']) / n_processed
        
        print(f"Images Processed: {n_processed}")
        print(f"Cache Hits:       {n_processed - n_rcnn}/{n_processed}")
        print("-" * 30)
        print(f"Avg Cache Lookup: {avg_cache:.6f} sec")
        print(f"Avg RCNN Compute: {avg_rcnn:.4f} sec (on miss)")
        print(f"Avg Inference:    {avg_infer:.4f} sec")
        print(f"Avg Total Time:   {avg_total:.4f} sec")
        print("-" * 30)
        
        if avg_rcnn > 0:
            speedup = avg_rcnn / (avg_cache + 0.000001)
            print(f"Cache Speedup Factor: {speedup:.1f}x faster than RCNN")
    else:
        print("No images were successfully processed.")

    # Sort & Rank
    results.sort(key=lambda x: x[1], reverse=True)
    gt_rank = -1
    for rank, (fname, score) in enumerate(results, start=1):
        if fname == gt_filename:
            gt_rank = rank
            break
            
    print("\n" + "="*40)
    print("       RETRIEVAL SANITY CHECK       ")
    print("="*40)
    if gt_rank != -1: print(f"✅ SUCCESS: GT found at Rank {gt_rank}/{len(results)}")
    else: print(f"❌ FAILURE: GT NOT found.")

    print("\nTop 10 Results:")
    for i, (fname, score) in enumerate(results[:10]):
        print(f"{i+1:02d}. {fname} \t Score: {score:.4f}{' <--- GT' if fname == gt_filename else ''}")

    plot_results(gt_filename, results, image_root)

# ... [Keep Imports and CRFInference Class exactly as before] ...

def run_full_evaluation():
    print("\n" + "="*50)
    print(f"   STARTING FULL EVALUATION (Cache Only)")
    print("="*50)

    # 1. Load Data
    test_graphs, _, _, _ = load_scene_graphs(TEST_ANNOTATIONS_PATH)
    
    # Define the Search Pool 
    pool_graphs = test_graphs[:MAX_IMAGES]
    
    # 2. Pre-load Features (STRICT CACHE MODE)
    print(f"\n[Phase 1] Loading features from cache...")
    
    feature_cache = load_feature_cache()
    if not feature_cache:
        print("CRITICAL ERROR: No cache found. Cannot run cache-only eval.")
        return

    pool_data = {} 
    skipped_count = 0
    
    for graph in pool_graphs:
        fname = graph.filename
        
        # --- STRICT CACHE CHECK ---
        if fname in feature_cache:
            # HIT: Load from RAM
            props = feature_cache[fname]['boxes']
            feats = feature_cache[fname]['features']
            
            # Slice if needed (optimization)
            if MAX_PROPOSALS > 0 and len(feats) > MAX_PROPOSALS:
                 props = props[:MAX_PROPOSALS]
                 feats = feats[:MAX_PROPOSALS]
            
            if len(feats) > 0:
                pool_data[fname] = (props, feats)
        else:
            # MISS: Skip entirely (Do not use R-CNN)
            skipped_count += 1

    print(f"Loaded {len(pool_data)} images from cache.")
    print(f"Skipped {skipped_count} images (not in cache).")

    if len(pool_data) == 0:
        print("No data available. Exiting.")
        return

    # 3. Evaluation Loop (unchanged logic, just runs faster)
    pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)
    _, obj_vocab, _, rel_vocab = load_scene_graphs(TEST_ANNOTATIONS_PATH)
    
    ranks = []
    
    print(f"\n[Phase 2] Running queries against {len(pool_data)} images...")
    start_time = time.time()

    # Only run queries for images we actually have data for
    valid_graphs = [g for g in pool_graphs if g.filename in pool_data]

    for i, query_graph in enumerate(valid_graphs):
        gt_filename = query_graph.filename
        query = build_query_from_graph(query_graph, obj_vocab, rel_vocab)
        
        scores = []
        # Search against strict pool
        for target_fname, (t_props, t_feats) in pool_data.items():
            score, _ = pipeline.beam_search(t_props, t_feats, query)
            scores.append((target_fname, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Find Rank
        rank = -1
        for r, (fname, s) in enumerate(scores, start=1):
            if fname == gt_filename:
                rank = r
                break
        
        if rank == -1: rank = len(pool_data) # Penalty is size of valid pool
        ranks.append(rank)
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(valid_graphs)} queries...")

    total_time = time.time() - start_time
    
    # 4. Metrics
    ranks = np.array(ranks)
    r1 = np.mean(ranks <= 1) * 100
    r5 = np.mean(ranks <= 5) * 100
    r10 = np.mean(ranks <= 10) * 100
    med_r = np.median(ranks)
    mean_r = np.mean(ranks)
    
    print("\n" + "="*50)
    print("         FINAL RESULTS (CACHE ONLY)         ")
    print("="*50)
    print(f"Valid Pool:   {len(pool_data)}")
    print(f"Total Time:   {total_time:.2f}s")
    print("-" * 30)
    print(f"Recall@1:     {r1:.1f}%")
    print(f"Recall@5:     {r5:.1f}%")
    print(f"Recall@10:    {r10:.1f}%")
    print(f"Median Rank:  {med_r}")
    print("="*50)

if __name__ == "__main__":
    # If run directly, execute full eval
    if RUN_INFERENCE:
        run_retrieval_sanity_check()
        #run_full_evaluation()
