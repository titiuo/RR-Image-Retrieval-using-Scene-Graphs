#!/usr/bin/env python
"""
Interactive Scene Graph Query Retrieval
Allows user to create/modify a scene graph query and find similar images
"""

import os
import sys
import pickle
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from CRF import train_unary
    from graph import load_scene_graphs
    from test_pipeline import (
        CRFInference,
        UNARY_MODEL_PATH,
        BINARY_MODEL_PATH,
        BINARY_PLATT_PATH,
        TEST_ANNOTATIONS_PATH,
        FEATURE_CACHE_PATH,
        find_image_path,
        load_feature_cache,
    )
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


class QueryEditor:
    """Interactive scene graph query editor and retriever"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("SCENE GRAPH QUERY RETRIEVAL - INTERACTIVE MODE")
        print("="*60)
        
        # Load test graphs and vocabularies
        print("\n[Loading] Test annotations...")
        self.graphs, self.obj_vocab, self.attr_vocab, self.rel_vocab = load_scene_graphs(
            TEST_ANNOTATIONS_PATH
        )
        
        # Load feature cache
        print("[Loading] Feature cache...")
        self.feature_cache = load_feature_cache()
        if not self.feature_cache:
            print("WARNING: No feature cache found. Live R-CNN extraction may be slow.")
            self.feature_cache = {}
        
        # Load CRF model
        print("[Loading] CRF inference model...")
        self.pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)
        
        # Initialize with first test graph
        self.query = self._load_default_query(0)
        print(f"\n✓ Ready! Loaded {len(self.graphs)} test images")
        print(f"✓ Feature cache has {len(self.feature_cache)} images")
        
    def _load_default_query(self, graph_idx=0):
        """Load a scene graph as initial query"""
        graph = self.graphs[graph_idx]
        objects = [self.obj_vocab.get_word(obj['class_id']) for obj in graph.objects]
        relationships = []
        for rel in graph.relationships:
            relationships.append([
                rel['subject_idx'],
                self.rel_vocab.get_word(rel['relation_id']),
                rel['object_idx']
            ])
        return {
            'objects': objects,
            'relationships': relationships,
            'source_image': graph.filename
        }
    
    def print_query(self):
        """Display current query"""
        print("\n" + "-"*60)
        print("CURRENT QUERY")
        print("-"*60)
        print(f"Objects: {self.query['objects']}")
        if self.query['relationships']:
            print("Relationships:")
            for s_idx, rel_name, o_idx in self.query['relationships']:
                print(f"  - {self.query['objects'][s_idx]} --[{rel_name}]--> {self.query['objects'][o_idx]}")
        else:
            print("Relationships: (none)")
        if 'source_image' in self.query:
            print(f"Source: {self.query['source_image']}")
        print("-"*60)
    
    def print_available_classes(self):
        """Show all available object/relation classes"""
        print("\n" + "-"*60)
        print("AVAILABLE CLASSES")
        print("-"*60)
        
        # Get all available classes from vocabularies
        obj_classes = sorted(set(self.obj_vocab.to_word.values()))
        rel_classes = sorted(set(self.rel_vocab.to_word.values()))
        
        print(f"\nObject Classes ({len(obj_classes)}):")
        for i, obj in enumerate(obj_classes, 1):
            print(f"  {i:2d}. {obj}")
        
        print(f"\nRelationship Classes ({len(rel_classes)}):")
        for i, rel in enumerate(rel_classes, 1):
            print(f"  {i:2d}. {rel}")
        print("-"*60)
    
    def add_object(self, obj_name):
        """Add an object to query"""
        if obj_name not in self.obj_vocab.to_id:
            print(f"❌ Unknown object: '{obj_name}'")
            return False
        self.query['objects'].append(obj_name)
        print(f"✓ Added object: '{obj_name}'")
        return True
    
    def remove_object(self, idx):
        """Remove an object from query"""
        if idx < 0 or idx >= len(self.query['objects']):
            print(f"❌ Invalid index: {idx}")
            return False
        removed = self.query['objects'].pop(idx)
        # Remove relationships involving this object
        self.query['relationships'] = [
            rel for rel in self.query['relationships']
            if rel[0] != idx and rel[2] != idx
        ]
        # Adjust indices in remaining relationships
        self.query['relationships'] = [
            [s if s < idx else s-1, r, o if o < idx else o-1]
            for s, r, o in self.query['relationships']
            if (s != idx and o != idx)
        ]
        print(f"✓ Removed object at index {idx}: '{removed}'")
        return True
    
    def add_relationship(self, subj_idx, rel_name, obj_idx):
        """Add a relationship to query"""
        if subj_idx < 0 or subj_idx >= len(self.query['objects']):
            print(f"❌ Invalid subject index: {subj_idx}")
            return False
        if obj_idx < 0 or obj_idx >= len(self.query['objects']):
            print(f"❌ Invalid object index: {obj_idx}")
            return False
        if rel_name not in self.rel_vocab.to_id:
            print(f"❌ Unknown relationship: '{rel_name}'")
            return False
        
        self.query['relationships'].append([subj_idx, rel_name, obj_idx])
        subj_name = self.query['objects'][subj_idx]
        obj_name = self.query['objects'][obj_idx]
        print(f"✓ Added relationship: {subj_name} --[{rel_name}]--> {obj_name}")
        return True
    
    def remove_relationship(self, idx):
        """Remove a relationship from query"""
        if idx < 0 or idx >= len(self.query['relationships']):
            print(f"❌ Invalid relationship index: {idx}")
            return False
        removed = self.query['relationships'].pop(idx)
        print(f"✓ Removed relationship: {removed}")
        return True
    
    def load_test_graph(self, graph_idx):
        """Load a different test graph as query"""
        if graph_idx < 0 or graph_idx >= len(self.graphs):
            print(f"❌ Invalid graph index. Available: 0-{len(self.graphs)-1}")
            return False
        self.query = self._load_default_query(graph_idx)
        print(f"✓ Loaded test graph #{graph_idx}: {self.graphs[graph_idx].filename}")
        return True
    
    def retrieve(self, top_k=10):
        """Run retrieval with current query"""
        if not self.query['objects']:
            print("❌ Query is empty! Add at least one object.")
            return None
        
        if not self.feature_cache:
            print("❌ Feature cache is empty! Cannot retrieve images.")
            print("   Make sure data/rcnn_test_features.pkl exists and is loaded.")
            return None
        
        print(f"\n[Retrieving] Searching for {len(self.feature_cache)} images...")
        print(f"[Query] Objects: {self.query['objects']}")
        
        results = []
        errors = 0
        
        try:
            for idx, (img_filename, cache_data) in enumerate(self.feature_cache.items()):
                if (idx + 1) % max(1, len(self.feature_cache) // 10) == 0:
                    print(f"  ✓ Processed {idx+1}/{len(self.feature_cache)}...")
                
                try:
                    boxes = cache_data.get('boxes', [])
                    features = cache_data.get('features', None)
                    
                    if features is None or len(features) == 0:
                        continue
                    
                    # Run inference
                    score, assignment = self.pipeline.beam_search(boxes, features, self.query)
                    if score > -9000:
                        results.append((img_filename, score))
                
                except Exception as e:
                    errors += 1
                    if errors <= 3:  # Only print first 3 errors
                        print(f"  ⚠ Error processing {img_filename}: {e}")
                    continue
            
            if errors > 3:
                print(f"  ⚠ ... and {errors - 3} more errors (suppressed)")
        
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n✓ Retrieved {len(results)} matching images")
        
        if not results:
            print("⚠ No results found. Try a different query.")
            return None
        
        # Display top-k
        print("\n" + "="*60)
        print(f"TOP {min(top_k, len(results))} RESULTS")
        print("="*60)
        for i, (fname, score) in enumerate(results[:top_k], 1):
            print(f"{i:2d}. {fname:30s}  Score: {score:8.4f}")
        print("="*60)
        
        return results[:top_k]
    
    def visualize_results(self, results):
        """Visualize retrieval results"""
        if not results:
            print("No results to visualize")
            return
        
        print("\n[Visualizing] Generating plot...")
        fig = plt.figure(figsize=(20, 10))
        
        # Display top results
        for i, (fname, score) in enumerate(results[:10]):
            ax = plt.subplot(2, 5, i+1)
            img_path = find_image_path(fname)
            
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
            
            ax.set_title(f"Rank {i+1}\nScore: {score:.4f}", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_menu(self):
        """Interactive menu loop"""
        while True:
            print("\n" + "="*60)
            print("OPTIONS")
            print("="*60)
            print("1. View current query")
            print("2. Load test graph (start from existing query)")
            print("3. Add object")
            print("4. Remove object")
            print("5. Add relationship")
            print("6. Remove relationship")
            print("7. View available classes")
            print("8. Run retrieval")
            print("9. Clear query")
            print("0. Exit")
            print("="*60)
            
            choice = input("Enter choice (0-9): ").strip()
            
            if choice == '0':
                print("\nGoodbye!")
                break
            
            elif choice == '1':
                self.print_query()
            
            elif choice == '2':
                try:
                    idx = int(input("Enter test graph index (0-{}): ".format(len(self.graphs)-1)))
                    self.load_test_graph(idx)
                    self.print_query()
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '3':
                obj_name = input("Enter object name: ").strip()
                self.add_object(obj_name)
            
            elif choice == '4':
                try:
                    idx = int(input("Enter object index to remove: "))
                    self.remove_object(idx)
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '5':
                try:
                    self.print_query()
                    s_idx = int(input("Subject index: "))
                    rel_name = input("Relationship name: ").strip()
                    o_idx = int(input("Object index: "))
                    self.add_relationship(s_idx, rel_name, o_idx)
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '6':
                try:
                    idx = int(input("Enter relationship index to remove: "))
                    self.remove_relationship(idx)
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '7':
                self.print_available_classes()
            
            elif choice == '8':
                results = self.retrieve(top_k=10)
                if results:
                    visualize = input("\nVisualize results? (y/n): ").strip().lower()
                    if visualize == 'y':
                        self.visualize_results(results)
            
            elif choice == '9':
                self.query = {'objects': [], 'relationships': []}
                print("✓ Query cleared")
            
            else:
                print("❌ Invalid choice")


if __name__ == "__main__":
    editor = QueryEditor()
    editor.interactive_menu()
