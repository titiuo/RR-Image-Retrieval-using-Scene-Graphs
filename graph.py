import json

# 1. The SceneGraph Class (The Data Container)
class SceneGraph:
    def __init__(self, filename=None):
        self.objects = []
        self.relationships = []
        if filename:
            self.filename = filename  # Store the image filename (e.g., "1024.jpg")
        else:
            print("Warning: No filename provided for SceneGraph.")
            self.filename = "unknown.jpg"

    def add_object(self, class_id, attribute_ids, bbox):
        self.objects.append({
            "class_id": class_id,
            "attributes": attribute_ids,
            "bbox": bbox  # Store bbox [x, y, w, h]
        })

    def add_relationship(self, subject_idx, relation_id, object_idx):
        self.relationships.append({
            "subject_idx": subject_idx,
            "relation_id": relation_id,
            "object_idx": object_idx
        })


# 2. The Vocabulary Helper (To map "man" -> 1, "dog" -> 2)
class Vocabulary:
    def __init__(self):
        self.to_id = {}
        self.to_word = {}
    
    def add(self, word):
        if word not in self.to_id:
            idx = len(self.to_id)
            self.to_id[word] = idx
            self.to_word[idx] = word
            return idx
        return self.to_id[word]

    def get_word(self, idx):
        return self.to_word.get(idx, "UNK")

# 3. The Loading Function
# --- Updated Loader to capture Bounding Boxes ---
def load_scene_graphs(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    graphs = []
    obj_vocab = Vocabulary()
    attr_vocab = Vocabulary()
    rel_vocab = Vocabulary()

    for entry in raw_data:
        # Pass filename to the constructor
        sg = SceneGraph(entry['filename'])
        
        # Process Objects
        for obj in entry['objects']:
            name_str = obj['names'][0]
            c_id = obj_vocab.add(name_str)
            
            a_ids = []
            if 'attributes' in obj:
                for attr in obj['attributes']:
                    attr_str = attr['attribute'] if isinstance(attr, dict) else attr
                    a_ids.append(attr_vocab.add(attr_str))
            
            # Capture BBOX: Ensure we get x, y, w, h
            # The JSON usually has it as 'bbox': {'x':..., 'y':..., 'w':..., 'h':...} 
            # or sometimes just a list. We assume dict based on your previous code.
            bbox = obj['bbox'] 
            
            sg.add_object(c_id, a_ids, bbox)

        # Process Relationships
        for rel in entry['relationships']:
            subj_idx = rel['objects'][0]
            obj_idx = rel['objects'][1]
            rel_str = rel['relationship']
            r_id = rel_vocab.add(rel_str)
            sg.add_relationship(subj_idx, r_id, obj_idx)
            
        graphs.append(sg)

    return graphs, obj_vocab, attr_vocab, rel_vocab




class ImageCandidates:
    def __init__(self, boxes, features):
        self.boxes = boxes        # [N, 4] numpy array (x, y, w, h)
        self.features = features  # [N, 4096] numpy array
    
    def num_boxes(self):
        return len(self.boxes)


def build_predicted_scene_graph(boxes, filename="predicted.jpg"):
    """
    Builds a predicted SceneGraph from RCNN bounding boxes.
    Each box becomes an object with a generic name.
    
    Args:
        boxes: List or array of bounding boxes in format [x, y, w, h]
        filename: Image filename (optional)
    
    Returns:
        SceneGraph object with predicted objects
    """
    sg = SceneGraph(filename)
    
    # Create a simple vocabulary for predicted objects
    obj_vocab = Vocabulary()
    obj_vocab.add("predicted_object")  # Generic class ID 0
    
    # Add each detected box as an object
    if boxes is not None and len(boxes) > 0:
        for box_idx, box in enumerate(boxes):
            try:
                # Handle different box formats
                if isinstance(box, (list, tuple)):
                    if len(box) >= 4:
                        bbox = {'x': int(box[0]), 'y': int(box[1]), 'w': int(box[2]), 'h': int(box[3])}
                    else:
                        continue
                else:
                    # numpy array handling
                    bbox = {'x': int(box[0]), 'y': int(box[1]), 'w': int(box[2]), 'h': int(box[3])}
                
                # Add object with class_id 0 and no attributes
                sg.add_object(0, [], bbox)
            except (TypeError, ValueError, IndexError):
                continue
    
    return sg


def _calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes are in format [x, y, w, h].
    """
    try:
        x1_min, y1_min, w1, h1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
        x2_min, y2_min, w2, h2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
        
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    except (TypeError, ValueError):
        return 0.0


def build_predicted_scene_graph_with_gt_labels(predicted_boxes, ground_truth_sg, obj_vocab, iou_threshold=0.3):
    """
    Builds a predicted SceneGraph from RCNN boxes matched to ground truth labels.
    Matches each predicted box to the closest ground truth box using IoU.
    
    Args:
        predicted_boxes: List or array of predicted bounding boxes [x, y, w, h]
        ground_truth_sg: Ground truth SceneGraph object for label lookup
        obj_vocab: Vocabulary for object class names
        iou_threshold: Minimum IoU to consider a match (default 0.3)
    
    Returns:
        SceneGraph object with predicted objects labeled with GT names
    """
    sg = SceneGraph(ground_truth_sg.filename if hasattr(ground_truth_sg, 'filename') else "predicted.jpg")
    
    if not predicted_boxes or len(predicted_boxes) == 0:
        return sg
    
    if not ground_truth_sg.objects or len(ground_truth_sg.objects) == 0:
        # No GT objects matched, use generic labels
        for box in predicted_boxes:
            try:
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    bbox = {'x': int(box[0]), 'y': int(box[1]), 'w': int(box[2]), 'h': int(box[3])}
                else:
                    bbox = {'x': int(box[0]), 'y': int(box[1]), 'w': int(box[2]), 'h': int(box[3])}
                sg.add_object(0, [], bbox)  # Generic class 0
            except (TypeError, ValueError, IndexError):
                continue
        return sg
    
    # Match each predicted box to closest GT box by IoU
    used_gt_indices = set()
    
    for pred_box in predicted_boxes:
        try:
            best_iou = 0
            best_gt_idx = -1
            
            # Find GT box with highest IoU
            for gt_idx, gt_obj in enumerate(ground_truth_sg.objects):
                if gt_idx in used_gt_indices:
                    continue
                
                gt_bbox = gt_obj['bbox']
                gt_box = [gt_bbox['x'], gt_bbox['y'], gt_bbox['w'], gt_bbox['h']]
                
                iou = _calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Use matched GT label if IoU exceeds threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_obj = ground_truth_sg.objects[best_gt_idx]
                class_id = gt_obj['class_id']
                attributes = gt_obj['attributes']
                used_gt_indices.add(best_gt_idx)
            else:
                # No good match, use generic label
                class_id = 0
                attributes = []
            
            # Add predicted box with assigned label
            if isinstance(pred_box, (list, tuple)) and len(pred_box) >= 4:
                bbox = {'x': int(pred_box[0]), 'y': int(pred_box[1]), 'w': int(pred_box[2]), 'h': int(pred_box[3])}
            else:
                bbox = {'x': int(pred_box[0]), 'y': int(pred_box[1]), 'w': int(pred_box[2]), 'h': int(pred_box[3])}
            
            sg.add_object(class_id, attributes, bbox)
            
        except (TypeError, ValueError, IndexError):
            continue
    
    return sg
