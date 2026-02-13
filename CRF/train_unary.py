import pandas as pd
import numpy as np
import pickle
import json
import os
import ast
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
CSV_PATH = '../../data/rcnn_training_cleaned.csv'  # Your new file
TRAIN_SPLIT_PATH = '/home/mamane/project_data/splits/train_split.json'
VAL_SPLIT_PATH = '/home/mamane/project_data/splits/val_split.json'
OUTPUT_PATH = '../trained_models/unary_potentials.pkl'
RCNN_WEIGHTS_PATH = '../../model/rcnn_finetuned.pth'
RCNN_INPUT_SIZE = (227, 227)
IMAGE_ROOT_CANDIDATES = [
    '../../sg_dataset/sg_train_images',
    '../../sg_dataset/sg_test_images',
    '../../sg_dataset/images',
]
MAX_BATCHES = int(os.getenv('MAX_BATCHES', '0'))  # 0 means no limit
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '128'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))
SAMPLE_RATIO = float(os.getenv('SAMPLE_RATIO', '0.5'))
MAX_BG_SAMPLES = int(os.getenv('MAX_BG_SAMPLES', '200000'))  # 0 means no limit
MAX_POS_SAMPLES_PER_CLASS = int(os.getenv('MAX_POS_SAMPLES_PER_CLASS', '50000'))  # 0 means no limit

# --- R-CNN BACKBONE FEATURE EXTRACTOR ---
_rcnn_model = None
_rcnn_transform = None
_rcnn_device = None
_image_root = None

class ProposalDataset(Dataset):
    def __init__(self, df, image_root, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['image'])
        box = row['box']
        label = row['class']
        is_bg = bool(row['is_background'])

        with Image.open(img_path).convert('RGB') as image:
            x1, y1, x2, y2 = _normalize_box(box)
            cropped = image.crop((x1, y1, x2, y2))
            tensor = self.transform(cropped) if self.transform else cropped

        return tensor, label, is_bg

def _init_rcnn():
    global _rcnn_model, _rcnn_transform, _rcnn_device

    if _rcnn_model is not None:
        return _rcnn_model

    rcnn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RCNN'))
    if rcnn_dir not in sys.path:
        sys.path.append(rcnn_dir)

    from rcnn import MultiHeadRCNN

    _rcnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[RCNN] Using device: {_rcnn_device}")

    _rcnn_model = MultiHeadRCNN(num_objects=1, num_attributes=1)

    if os.path.exists(RCNN_WEIGHTS_PATH):
        state = torch.load(RCNN_WEIGHTS_PATH, map_location=_rcnn_device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state.items()
            if k.startswith('backbone.')
        }

        if backbone_state:
            _rcnn_model.backbone.load_state_dict(backbone_state, strict=False)
            print(f"[RCNN] Loaded backbone weights from {RCNN_WEIGHTS_PATH}")
        else:
            print(f"[RCNN] No backbone weights found in {RCNN_WEIGHTS_PATH}, using ImageNet backbone")
    else:
        print(f"[RCNN] Weights not found at {RCNN_WEIGHTS_PATH}, using ImageNet backbone")

    _rcnn_model.to(_rcnn_device)
    _rcnn_model.eval()

    _rcnn_transform = transforms.Compose([
        transforms.Resize(RCNN_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return _rcnn_model

def _normalize_box(box):
    if len(box) != 4:
        raise ValueError(f"Box must have 4 values, got {box}")

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
    return (x1, y1, x2, y2)

def extract_box_features(img_path, box):
    model = _init_rcnn()

    with Image.open(img_path).convert('RGB') as image:
        x1, y1, x2, y2 = _normalize_box(box)
        cropped = image.crop((x1, y1, x2, y2))
        tensor = _rcnn_transform(cropped).unsqueeze(0).to(_rcnn_device)

    with torch.no_grad():
        features = model.backbone(tensor).squeeze(0).cpu().numpy()
    return features

def _resolve_image_root():
    global _image_root
    if _image_root is not None:
        return _image_root

    for candidate in IMAGE_ROOT_CANDIDATES:
        if os.path.isdir(candidate):
            _image_root = candidate
            print(f"[Data] Using image root: {_image_root}")
            return _image_root

    raise FileNotFoundError(
        "No image root found. Tried: " + ", ".join(IMAGE_ROOT_CANDIDATES)
    )

def get_split_filenames(json_path):
    """Returns a set of image filenames belonging to a split."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            print(f"[Split] {json_path} is empty")
            return set()
        sample = data[0]
        if isinstance(sample, dict):
            if 'filename' in sample:
                filenames = {x['filename'] for x in data if 'filename' in x}
                print(f"[Split] {json_path}: found {len(filenames)} filenames via 'filename'")
                return filenames
            if 'image_path' in sample:
                filenames = {x['image_path'].split('/')[-1] for x in data if 'image_path' in x}
                print(f"[Split] {json_path}: found {len(filenames)} filenames via 'image_path'")
                return filenames
        if isinstance(sample, str):
            print(f"[Split] {json_path}: list of filenames")
            return set(data)

    raise ValueError(f"Unsupported split format in {json_path}")

def load_and_extract_from_csv(
    csv_path,
    target_filenames,
    sample_ratio=1.0,
    batch_size=128,
    num_workers=4,
    max_batches=None,
    max_bg_samples=0,
    max_pos_samples_per_class=0,
):
    """
    Reads CSV, filters by split, and extracts R-CNN features.
    Returns: 
        positives: { 'class_name': [vectors...] }
        negatives: [vectors...] (Background samples)
    """
    print(f"[Data] Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by split (Train or Val)
    # We check if the csv 'image' column exists in our target_filenames set
    df = df[df['image'].isin(target_filenames)]
    print(f"[Data] After split filter: {len(df)} rows")
    
    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42)
        print(f"[Data] After sampling ({sample_ratio}): {len(df)} rows")

    # Pre-parse boxes once
    df = df.copy()
    df['box'] = df['box'].apply(ast.literal_eval)

    pos_features = defaultdict(list)
    neg_features = []  # Generic background bucket
    pos_seen = defaultdict(int)
    bg_seen = 0

    print(f"[Data] Extracting features for {len(df)} proposals...")

    _init_rcnn()
    image_root = _resolve_image_root()
    dataset = ProposalDataset(df, image_root, _rcnn_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(_rcnn_device.type == 'cuda'),
        shuffle=False,
    )

    total_batches = len(dataloader)

    for batch_idx, (images, labels, is_bg) in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(
                f"  Batch {batch_idx + 1}/{total_batches}"
                f" - processed {batch_idx * batch_size} boxes..."
            )

        images = images.to(_rcnn_device, non_blocking=True)
        with torch.no_grad():
            feats = _rcnn_model.backbone(images).cpu().numpy()

        labels_list = list(labels)
        is_bg_list = is_bg.tolist() if hasattr(is_bg, 'tolist') else list(is_bg)

        for i, feat in enumerate(feats):
            label = labels_list[i]
            if is_bg_list[i] or label == 'background':
                if max_bg_samples and max_bg_samples > 0:
                    bg_seen += 1
                    if len(neg_features) < max_bg_samples:
                        neg_features.append(feat)
                    else:
                        j = random.randint(0, bg_seen - 1)
                        if j < max_bg_samples:
                            neg_features[j] = feat
                else:
                    neg_features.append(feat)
            else:
                if max_pos_samples_per_class and max_pos_samples_per_class > 0:
                    pos_seen[label] += 1
                    bucket = pos_features[label]
                    if len(bucket) < max_pos_samples_per_class:
                        bucket.append(feat)
                    else:
                        j = random.randint(0, pos_seen[label] - 1)
                        if j < max_pos_samples_per_class:
                            bucket[j] = feat
                else:
                    pos_features[label].append(feat)

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            print(f"[Data] Stopping early after {max_batches} batch(es).")
            break

    return pos_features, neg_features

def train_svms(pos_dict, background_list):
    """
    Trains One-vs-Rest SVMs using explicit background data.
    """
    svms = {}
    classes = list(pos_dict.keys())
    
    # Convert background list to numpy once
    bg_feats = np.array(background_list)
    
    print(f"\n--- Training {len(classes)} SVMs ---")
    
    for cls in classes:
        pos_feats = np.array(pos_dict[cls])
        
        # --- NEGATIVE MINING STRATEGY ---
        # Negatives = (Background Samples) + (Samples from other classes)
        
        # 1. Sample from Background (e.g., 2x positives)
        n_bg = min(len(bg_feats), len(pos_feats) * 2)
        if n_bg > 0:
            bg_indices = np.random.choice(len(bg_feats), n_bg, replace=False)
            current_negatives = list(bg_feats[bg_indices])
        else:
            current_negatives = []

        # 2. Sample from Other Classes (Hard Negatives)
        # e.g. "Dog" is a negative for "Cat"
        for other in classes:
            if other == cls: continue
            others = pos_dict[other]
            # Take small sample
            n_take = min(len(others), 20)
            current_negatives.extend(random.sample(others, n_take))
            
        current_negatives = np.array(current_negatives)
        
        if len(pos_feats) < 5 or len(current_negatives) < 5:
            print(f"Skipping {cls}: Insufficient data.")
            continue
            
        # Prepare Data
        X = np.vstack([pos_feats, current_negatives])
        y = np.hstack([np.ones(len(pos_feats)), np.zeros(len(current_negatives))])
        
        # Train
        clf = LinearSVC(random_state=42, dual=False, class_weight='balanced', max_iter=2000)
        clf.fit(X, y)
        svms[cls] = clf
        
    return svms

def calibrate_platt(svms, val_pos_dict, val_bg_list):
    """
    Learns Platt Scaling on Validation Set.
    """
    platt = {}
    bg_feats = np.array(val_bg_list)
    
    print(f"\n--- Calibrating ---")
    
    for cls, clf in svms.items():
        if cls not in val_pos_dict: continue
        
        pos_feats = np.array(val_pos_dict[cls])
        
        # Negatives for calibration
        # Combine some background + some other classes
        neg_feats = []
        
        # Add background
        if len(bg_feats) > 0:
            n_bg = min(len(bg_feats), 50)
            neg_feats.extend(bg_feats[np.random.choice(len(bg_feats), n_bg, replace=False)])
            
        # Add others
        for other in val_pos_dict:
            if other == cls: continue
            neg_feats.extend(random.sample(val_pos_dict[other], min(len(val_pos_dict[other]), 10)))
            
        if len(pos_feats) < 2 or len(neg_feats) < 2: continue
        
        neg_feats = np.array(neg_feats)
        
        # Get Scores
        scores_pos = clf.decision_function(pos_feats)
        scores_neg = clf.decision_function(neg_feats)
        
        X_cal = np.concatenate([scores_pos, scores_neg]).reshape(-1, 1)
        y_cal = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_cal, y_cal)
        
        platt[cls] = (lr.coef_[0][0], lr.intercept_[0])
        
    return platt

if __name__ == "__main__":
    # 1. Identify Split Files
    train_files = get_split_filenames(TRAIN_SPLIT_PATH)
    val_files = get_split_filenames(VAL_SPLIT_PATH)
    
    print(f"Found {len(train_files)} training images and {len(val_files)} validation images.")

    # 2. Extract Training Data
    print("\n--- PROCESSING TRAINING SET ---")
    train_pos, train_neg = load_and_extract_from_csv(
        CSV_PATH,
        train_files,
        sample_ratio=SAMPLE_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_batches=(MAX_BATCHES or None),
        max_bg_samples=MAX_BG_SAMPLES,
        max_pos_samples_per_class=MAX_POS_SAMPLES_PER_CLASS,
    )  # Subsample if too slow
    
    # 3. Train SVMs
    svms = train_svms(train_pos, train_neg)
    
    # 4. Extract Validation Data
    print("\n--- PROCESSING VALIDATION SET ---")
    val_pos, val_neg = load_and_extract_from_csv(
        CSV_PATH,
        val_files,
        sample_ratio=SAMPLE_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_batches=(MAX_BATCHES or None),
        max_bg_samples=MAX_BG_SAMPLES,
        max_pos_samples_per_class=MAX_POS_SAMPLES_PER_CLASS,
    )
    
    # 5. Calibrate
    platt = calibrate_platt(svms, val_pos, val_neg)
    
    # 6. Save
    model_bundle = {'svms': svms, 'platt': platt}
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_bundle, f)
    print("Done!")