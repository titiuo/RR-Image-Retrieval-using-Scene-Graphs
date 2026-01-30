import json

# 1. The SceneGraph Class (The Data Container)
class SceneGraph:
    def __init__(self, filename):
        self.filename = filename  # Store the image filename (e.g., "1024.jpg")
        self.objects = []
        self.relationships = []

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