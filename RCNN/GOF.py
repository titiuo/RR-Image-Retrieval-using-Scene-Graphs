import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import os
class ObjectProposalPipeline:
    def __init__(self, json_data, image_path=None):
        """
        Initialize the pipeline with annotations and image.
        If image_path is None, a black image will be created for demo.
        """
        self.annotations = json_data
        self.image_path = image_path
        def save_results_to_json(results: List[Dict], output_file: str = "output.json"):
            """
            Save processed image names and bounding boxes to a JSON file.
            
            Args:
                results: List of results from process_json_annotations
                output_file: Path to output JSON file
            """
            output_data = []
            
            for result in results:
                image_data = {
                    "photo_id": result['photo_id'],
                    "bounding_boxes": [
                        {
                            "bbox": list(item['bbox']),
                            "class": item['class'],
                            "attributes": item['attributes'],
                            "iou": item['iou']
                        }
                        for item in result['training_data']
                        if item['class'] != 'background'
                    ]
                }
                output_data.append(image_data)
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"[INFO] Results saved to {output_file}")
        self.image = None
        self.proposals = []
        self.training_data = []

    def load_image(self):
        """Load image or create a dummy image for demonstration.
        Uses a unicode-safe loader on Windows (np.fromfile + cv2.imdecode).
        Falls back to PIL if cv2 can't decode the bytes.
        Raises ValueError if the image cannot be loaded.
        """
        if self.image_path and Path(self.image_path).exists():
            try:
                # Read raw bytes - works with Unicode paths on Windows
                img_bytes = np.fromfile(self.image_path, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    # convert to RGB for consistency
                    self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to PIL as a secondary attempt
                    try:
                        from PIL import Image
                        with Image.open(self.image_path) as im:
                            im = im.convert('RGB')
                            self.image = np.array(im)
                    except Exception:
                        raise ValueError(f"Could not load image: {self.image_path}")
            except Exception as e:
                # Propagate as ValueError with a helpful message
                raise ValueError(f"Could not load image: {self.image_path}. {e}")
        else:
            h = self.annotations.get('height', 500)
            w = self.annotations.get('width', 500)
            self.image = np.zeros((h, w, 3), dtype=np.uint8)
            print(f"[INFO] Image not found, created black image {w}x{h}")

    def run_gop_simulation(self):
        """Simulate the GOP algorithm with Selective Search"""
        print("[INFO] Starting proposal generation (GOP simulation)...")
        
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(self.image)
        ss.switchToSelectiveSearchFast()
        
        rects = ss.process()
        print(f"[INFO] {len(rects)} candidate regions generated.")
        
        self.proposals = rects[:2000]
        return self.proposals

    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def match_proposals_to_ground_truth(self, iou_threshold=0.5):
        """Match proposals to ground truth annotations"""
        ground_truth_objects = self.annotations.get('objects', [])
        labeled_regions = []

        print("[INFO] Matching proposals with Ground Truth...")

        for prop_rect in self.proposals:
            best_iou = 0
            best_label = "background"
            best_attributes = []

            for obj in ground_truth_objects:
                gt_bbox = (obj['bbox']['x'], obj['bbox']['y'], 
                          obj['bbox']['w'], obj['bbox']['h'])
                
                iou = self.calculate_iou(prop_rect, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    if iou > iou_threshold:
                        best_label = obj['names'][0]
                        best_attributes = [attr['attribute'] 
                                         for attr in obj.get('attributes', [])]

            labeled_regions.append({
                "bbox": prop_rect,
                "class": best_label,
                "attributes": best_attributes,
                "iou": best_iou
            })

        self.training_data = labeled_regions
        return labeled_regions

    def extract_features_and_labels(self):
        """Extract features and labels for CNN/SVM"""
        X_batch = []
        Y_labels = []
        
        for item in self.training_data:
            if item['class'] == "background":
                continue 

            x, y, w, h = item['bbox']
            roi = self.image[max(0, int(y)):min(int(y+h), self.image.shape[0]), 
                            max(0, int(x)):min(int(x+w), self.image.shape[1])]
            
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                X_batch.append(roi_resized)
                Y_labels.append(item['class'])
                
        print(f"[INFO] {len(X_batch)} positive regions extracted.")
        return X_batch, Y_labels

    def visualize_results(self, max_boxes=20):
        """Visualize the last processed image with detected objects"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(self.image)
        
        count = 0
        for item in self.training_data:
            if item['class'] != 'background' and count < max_boxes:
                x, y, w, h = item['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                label = f"{item['class']} ({item['iou']:.2f})"
                ax.text(x, y-5, label, color='red', fontsize=8, 
                       bbox=dict(facecolor='yellow', alpha=0.5))
                count += 1
        
        ax.set_title(f"Detected Objects (showing {count})")
        ax.axis('off')
        plt.tight_layout()
        return fig


def process_json_annotations(json_file_path: str, images_folder: str = None, 
                             iou_threshold: float = 0.5,
                             missing_images_log: str = "missing_images.txt") -> List[Dict]:
    """
    Process all images from a JSON file containing scene graph annotations.
    
    Args:
        json_file_path: Path to JSON file containing annotations
        images_folder: Path to folder containing images (optional)
        iou_threshold: IoU threshold for matching
        missing_images_log: Path to log file for missing images
    
    Returns:
        List of results for each image
    """
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        raise ValueError(f"JSON file not found: {json_file_path}")
    
    with open(json_path, 'r') as f:
        annotations_list = json.load(f)
    
    results = []
    missing_images = []
    
    print(f"[INFO] Found {len(annotations_list)} images in JSON file")
    
    for idx, annotation in enumerate(annotations_list):
        photo_id = annotation.get('photo_id', f'image_{idx}')
        print(f"\n[INFO] Processing image {idx+1}/{len(annotations_list)}: {photo_id}")
        
        # Find corresponding image file
        image_path = None
        image_found = False
        
        if images_folder:
            folder = Path(images_folder)
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                candidates = list(folder.glob(f"{photo_id}{ext}"))
                if candidates:
                    image_path = str(candidates[0])
                    image_found = True
                    break
        
        if not image_found and images_folder:
            print(f"[WARN] Image {photo_id} not found, skipping.")
            missing_images.append(photo_id)
            continue
        
        try:
            pipeline = ObjectProposalPipeline(annotation, image_path)
            pipeline.load_image()
            pipeline.run_gop_simulation()
            pipeline.match_proposals_to_ground_truth(iou_threshold)
            X_batch, Y_labels = pipeline.extract_features_and_labels()
            
            results.append({
                "photo_id": photo_id,
                "pipeline": pipeline,
                "features": X_batch,
                "labels": Y_labels,
                "training_data": pipeline.training_data,
                "relationships": annotation.get('relationships', [])
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process {photo_id}: {str(e)}")
            if photo_id not in missing_images:
                missing_images.append(photo_id)
    
    # Write missing images to log file
    if missing_images:
        with open(missing_images_log, 'w') as f:
            for img_name in missing_images:
                f.write(f"{img_name}\n")
        print(f"\n[INFO] {len(missing_images)} missing images logged to {missing_images_log}")
    
    return results


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Use os.getcwd() instead of os.pwd()
    current = os.getcwd()
    
    # Process JSON file with annotations
    json_file = os.path.join(current, "RCNN",  "test_json", "sg_train_annotations.json")
    images_folder = os.path.join(current, "RCNN",  "test_image")
    
    try:
        results = process_json_annotations(
            json_file, 
            images_folder, 
            iou_threshold=0.7,
            missing_images_log="missing_images.txt"
        )
        
        # Print summary first
        print(f"\n[SUMMARY] Processed {len(results)} images successfully")
        for res in results:
            num_objects = len(res['labels'])
            num_relationships = len(res['relationships'])
            print(f"  - {res['photo_id']}: {num_objects} objects, {num_relationships} relationships")
        
        # Visualize results
        if results:
            last_result = results[-1]
            print(f"\n[INFO] Visualizing last processed image: {last_result['photo_id']}")
            fig = last_result['pipeline'].visualize_results()
            plt.show()
            
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")