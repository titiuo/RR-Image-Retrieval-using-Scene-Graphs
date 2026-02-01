import numpy as np
from sklearn.mixture import GaussianMixture

# ==========================================
# PHASE 1: DATA STRUCTURES
# ==========================================

class SceneGraph:
    def __init__(self):
        self.objects = []       # List of {class_id, attributes}
        self.relationships = [] # List of {subject_idx, relation_id, object_idx}

    def add_object(self, class_id, attribute_ids=None):
        if attribute_ids is None: attribute_ids = []
        self.objects.append({"class_id": class_id, "attributes": attribute_ids})
        return len(self.objects) - 1

    def add_relationship(self, subject_idx, relation_id, object_idx):
        self.relationships.append({
            "subject_idx": subject_idx,
            "relation_id": relation_id,
            "object_idx": object_idx
        })

class ImageCandidates:
    def __init__(self, boxes, features):
        self.boxes = boxes        # [N, 4] numpy array (x, y, w, h)
        self.features = features  # [N, 4096] numpy array
    
    def num_boxes(self):
        return len(self.boxes)

# ==========================================
# PHASE 2: SCORING FUNCTIONS (POTENTIALS)
# ==========================================

class UnaryPotential:
    def __init__(self, classifiers, platt_params):
        self.classifiers = classifiers
        self.platt_params = platt_params

    def _sigmoid(self, x, A, B):
        return 1 / (1 + np.exp(A * x + B))

    def compute_matrix(self, scene_graph, image_candidates):
        """
        Computes the unary potential matrix.
        Args:
            scene_graph: SceneGraph object.
            image_candidates: ImageCandidates object.
        Returns:
            unary_matrix_log: (N_objects, M_boxes) numpy array of log-probabilities: log P(object|box)
        """
        num_objs = len(scene_graph.objects)
        num_boxes = image_candidates.num_boxes()
        
        # Initialize with 0.0 log-probability (probability 1.0)
        # We work in LOG SPACE to prevent underflow
        unary_matrix_log = np.zeros((num_objs, num_boxes))
        
        box_features = image_candidates.features

        for i, obj in enumerate(scene_graph.objects):
            # 1. Object Score
            c_id = obj['class_id']
            if c_id in self.classifiers:
                raw_scores = self.classifiers[c_id].predict(box_features)
                A, B = self.platt_params.get(c_id, (1, 0))
                probs = self._sigmoid(raw_scores, A, B)             # probs = P(class|box)
                unary_matrix_log[i, :] += np.log(probs + 1e-9) 

            # 2. Attribute Scores
            for attr_id in obj['attributes']:
                if attr_id in self.classifiers:
                    raw_scores = self.classifiers[attr_id].predict(box_features)
                    A, B = self.platt_params.get(attr_id, (1, 0))
                    probs = self._sigmoid(raw_scores, A, B)        # probs = P(attribute|box)
                    unary_matrix_log[i, :] += np.log(probs + 1e-9)
                
        return unary_matrix_log

class BinaryPotential:
    def __init__(self, generic_gmms, specific_gmms, platt_params):
        self.generic_gmms = generic_gmms
        self.specific_gmms = specific_gmms
        self.platt_params = platt_params

    def extract_features(self, box_sub, box_obj):
        """
        Extracts spatial features for a subject-object box pair.
        Args:
            box_sub: (4,) array [x, y, w, h] for subject
            box_obj: (4,) array [xp, yp, wp, hp] for object
        Returns:
            features: (4,) array of spatial features: f(gamma_o,gamma_op)        
        """
        x, y, w, h = box_sub
        xp, yp, wp, hp = box_obj
        # Avoid div by zero
        w, h = max(w, 1e-6), max(h, 1e-6)
        return np.array([(x-xp)/w, (y-yp)/h, wp/w, hp/h])

    def compute_log_prob(self, box_sub, box_obj, sub_cls, rel_id, obj_cls):
        features = self.extract_features(box_sub, box_obj).reshape(1, -1)
        
        # Select Model
        triplet = (sub_cls, rel_id, obj_cls)
        if triplet in self.specific_gmms:
            model = self.specific_gmms[triplet]
            key = triplet
        elif rel_id in self.generic_gmms:
            model = self.generic_gmms[rel_id]
            key = rel_id
        else:
            return np.log(0.001) # Penalty for unknown relation

        # Get Log Density
        log_density = model.score_samples(features)[0]
        
        # Platt Scale (Sigmoid on log density)
        if key in self.platt_params:
            A, B = self.platt_params[key]
            prob = 1 / (1 + np.exp(A * log_density + B))
        else:
            prob = 0.5
            
        return np.log(prob + 1e-9)

# ==========================================
# PHASE 3: INFERENCE ENGINE
# ==========================================

class CRFInference:
    def __init__(self, unary_model, binary_model):
        self.unary = unary_model
        self.binary = binary_model

    def solve(self, graph, candidates):
        """
        Finds the best grounding gamma* using a simplified Beam Search.
        Real paper uses Belief Propagation, but Beam Search works for mocks.
        """
        # 1. Compute Unary Matrix (N_objects x M_boxes)
        # shape: (N, M)
        unary_scores = self.unary.compute_matrix(graph, candidates)
        
        num_objs = len(graph.objects)
        num_boxes = candidates.num_boxes()
        
        # 2. Beam Search Initialization
        # Start with the first object. Keep top K assignments.
        K = 10 
        # List of paths: (current_score, [list_of_box_indices])
        paths = []
        
        # Initialize paths with the first object's unary scores
        first_obj_scores = unary_scores[0]
        # Get top K boxes for the first object
        top_indices = np.argsort(first_obj_scores)[::-1][:K]
        
        for idx in top_indices:
            paths.append((first_obj_scores[idx], [idx]))

        # 3. Iterate through remaining objects
        for obj_idx in range(1, num_objs):
            new_paths = []
            
            # For each existing path, try extending it with a new box for the current object
            for score, assignment in paths:
                # Calculate Unary for current object
                # Optimization: Only consider top K unary matches for this object too
                current_unary = unary_scores[obj_idx]
                top_k_candidates = np.argsort(current_unary)[::-1][:K]

                for box_idx in top_k_candidates:
                    # Constraint: One box cannot be two objects (optional, usually enforced)
                    if box_idx in assignment:
                        continue
                        
                    new_score = score + current_unary[box_idx]
                    
                    # Calculate Binary Potentials with all previous objects in the graph
                    # Check if there are edges between (0...obj_idx-1) and (obj_idx)
                    for rel in graph.relationships:
                        # Case A: Previous Object -> Current Object
                        if rel['object_idx'] == obj_idx and rel['subject_idx'] < obj_idx:
                            prev_box_idx = assignment[rel['subject_idx']]
                            bin_score = self.binary.compute_log_prob(
                                candidates.boxes[prev_box_idx],
                                candidates.boxes[box_idx],
                                graph.objects[rel['subject_idx']]['class_id'],
                                rel['relation_id'],
                                graph.objects[obj_idx]['class_id']
                            )
                            new_score += bin_score

                        # Case B: Current Object -> Previous Object
                        elif rel['subject_idx'] == obj_idx and rel['object_idx'] < obj_idx:
                            prev_box_idx = assignment[rel['object_idx']]
                            bin_score = self.binary.compute_log_prob(
                                candidates.boxes[box_idx],
                                candidates.boxes[prev_box_idx],
                                graph.objects[obj_idx]['class_id'],
                                rel['relation_id'],
                                graph.objects[rel['object_idx']]['class_id']
                            )
                            new_score += bin_score

                    new_paths.append((new_score, assignment + [box_idx]))
            
            # Prune: Keep only top K paths
            new_paths.sort(key=lambda x: x[0], reverse=True)
            paths = new_paths[:K]

        # 4. Final Result
        best_score, best_assignment = paths[0]
        return best_score, best_assignment

# ==========================================
# PHASE 4: MAIN MOCK EXECUTION
# ==========================================

if __name__ == "__main__":
    # --- 1. MOCK MODELS ---
    class MockClassifier:
        def predict(self, X): return np.random.randn(X.shape[0]) # Random raw scores
    
    class MockGMM:
        def score_samples(self, X): return np.array([-2.0]) # Constant log density

    # Unary dependencies
    classifiers = {0: MockClassifier(), 1: MockClassifier(), 100: MockClassifier()}
    platt_unary = {0: (-1, 0), 1: (-1, 0), 100: (-1, 0)} # Standard params
    
    # Binary dependencies
    generic_gmms = {5: MockGMM()} # Relation 5: "on"
    specific_gmms = {}
    platt_binary = {5: (-1, 0)}

    # Initialize Modules
    unary_module = UnaryPotential(classifiers, platt_unary)
    binary_module = BinaryPotential(generic_gmms, specific_gmms, platt_binary)
    inference_engine = CRFInference(unary_module, binary_module)

    # --- 2. CREATE DUMMY DATA ---
    print("Generating Image Candidates...")
    # 20 random boxes [x,y,w,h]
    dummy_boxes = np.abs(np.random.rand(20, 4) * 100) 
    dummy_feats = np.random.rand(20, 4096)
    candidates = ImageCandidates(dummy_boxes, dummy_feats)

    print("Creating Scene Graph Query...")
    # Graph: "Man(0) [Old(100)] ON(5) Boat(1)"
    graph = SceneGraph()
    idx_man = graph.add_object(class_id=0, attribute_ids=[100]) # Man, Old
    idx_boat = graph.add_object(class_id=1)                     # Boat
    graph.add_relationship(idx_man, 5, idx_boat)                # Man ON Boat

    # --- 3. RUN PIPELINE ---
    print("\nRunning Inference...")
    final_score, assignment = inference_engine.solve(graph, candidates)

    # --- 4. OUTPUT ---
    print("="*30)
    print(f"Final Retrieval Score: {final_score:.4f} (Higher is better)")
    print(f"Best Assignment (Box Indices): {assignment}")
    print("="*30)
    
    # Interpretation
    box_man = candidates.boxes[assignment[0]]
    box_boat = candidates.boxes[assignment[1]]
    print(f"Man grounded to box at: {box_man}")
    print(f"Boat grounded to box at: {box_boat}")