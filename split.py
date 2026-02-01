import json
import random
import os

# CONFIGURATION
INPUT_PATH = '../sg_dataset/sg_train_annotations.json'
OUTPUT_DIR = '../sg_dataset/splits'
VAL_PERCENTAGE = 0.2  # 20% for validation, 80% for training

def split_dataset():
    print(f"Loading full training data from {INPUT_PATH}...")
    with open(INPUT_PATH, 'r') as f:
        full_data = json.load(f)

    total_images = len(full_data)
    val_size = int(total_images * VAL_PERCENTAGE)
    
    # Shuffle to ensure random distribution
    # Set seed for reproducibility (CRITICAL for scientific consistency)
    random.seed(42) 
    random.shuffle(full_data)

    # Slice the list
    val_data = full_data[:val_size]
    train_data = full_data[val_size:]

    print(f"Total Images: {total_images}")
    print(f"Training Split: {len(train_data)} images")
    print(f"Validation Split: {len(val_data)} images")

    # Save to new files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_out = os.path.join(OUTPUT_DIR, 'train_split.json')
    val_out = os.path.join(OUTPUT_DIR, 'val_split.json')

    with open(train_out, 'w') as f:
        json.dump(train_data, f)
        
    with open(val_out, 'w') as f:
        json.dump(val_data, f)

    print(f"Saved splits to {OUTPUT_DIR}/")

if __name__ == "__main__":
    split_dataset()