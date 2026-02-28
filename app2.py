#!/usr/bin/env python
"""
Interactive Scene Graph Query Retrieval - Streamlit App
"""

import os
import sys
import streamlit as st
from PIL import Image
import numpy as np

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
    st.error(f"Import Error: {e}")
    sys.exit(1)



# Initialize session state
@st.cache_resource
def load_models():
    """Load models and data once"""
    with st.spinner("Loading models and data..."):
        graphs, obj_vocab, attr_vocab, rel_vocab = load_scene_graphs(TEST_ANNOTATIONS_PATH)
        feature_cache = load_feature_cache()
        pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)
        
        # Extract available classes
        obj_classes = sorted(set(obj_vocab.to_word.values()))
        rel_classes = sorted(set(rel_vocab.to_word.values()))
        
        return {
            'graphs': graphs,
            'obj_vocab': obj_vocab,
            'attr_vocab': attr_vocab,
            'rel_vocab': rel_vocab,
            'feature_cache': feature_cache,
            'pipeline': pipeline,
            'obj_classes': obj_classes,
            'rel_classes': rel_classes
        }


def initialize_session_state():
    """Initialize session state variables"""
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    if 'objects' not in st.session_state:
        st.session_state.objects = []
    
    if 'relationships' not in st.session_state:
        st.session_state.relationships = []
    
    if 'results' not in st.session_state:
        st.session_state.results = None


def build_scene_graph():
    """Build scene graph from selected objects and relationships"""
    return {
        'objects': st.session_state.objects,
        'relationships': st.session_state.relationships
    }


def retrieve_images(query_graph, top_k=10):
    """Run retrieval with current query"""
    if not query_graph['objects']:
        st.error("❌ Query is empty! Add at least one object.")
        return None
    
    models = st.session_state.models
    feature_cache = models['feature_cache']
    pipeline = models['pipeline']
    
    if not feature_cache:
        st.error("❌ Feature cache is empty! Run precompute_100.py or precompute_all.py first.")
        return None
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(feature_cache)
    
    try:
        for idx, (img_filename, cache_data) in enumerate(feature_cache.items()):
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1}/{total}")
            
            try:
                boxes = cache_data.get('boxes', [])
                features = cache_data.get('features', None)
                
                if features is None or len(features) == 0:
                    continue
                
                score, _ = pipeline.beam_search(boxes, features, query_graph)
                if score > -9000:
                    results.append((img_filename, score))
            
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
    
    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        return None
    
    if not results:
        st.warning("⚠ No results found. Try a different query.")
        return None
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    st.set_page_config(page_title="Scene Graph Image Retrieval", layout="wide")
    
    # Title
    st.markdown("# 🔍 Scene Graph Image Retrieval")
    st.markdown("Build a scene graph query by selecting objects and relationships, then find similar images.")
    
    # Initialize session state
    initialize_session_state()
    models = st.session_state.models
    
    # ==================== SIDEBAR: Query Builder ====================
    with st.sidebar:
        st.markdown("## 📝 Build Your Query")
        
        # Objects Section
        st.markdown("### Objects")
        obj_name = st.selectbox(
            "Select an object to add:",
            models['obj_classes'],
            key="obj_select"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")
        with col2:
            if st.button("➕ Add", key="add_obj"):
                if obj_name and obj_name not in st.session_state.objects:
                    st.session_state.objects.append(obj_name)
                    st.success(f"Added: {obj_name}")
        
        # Display current objects
        if st.session_state.objects:
            st.markdown("**Current objects:**")
            for i, obj in enumerate(st.session_state.objects):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i}. {obj}")
                with col2:
                    if st.button("❌", key=f"remove_obj_{i}"):
                        st.session_state.objects.pop(i)
                        st.rerun()
        
        st.divider()
        
        # Relationships Section
        st.markdown("### Relationships")
        if len(st.session_state.objects) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                subj_idx = st.selectbox(
                    "Subject:",
                    range(len(st.session_state.objects)),
                    format_func=lambda i: st.session_state.objects[i],
                    key="subj_select"
                )
            with col2:
                rel_name = st.selectbox(
                    "Relationship:",
                    models['rel_classes'],
                    key="rel_select"
                )
            with col3:
                obj_idx = st.selectbox(
                    "Object:",
                    range(len(st.session_state.objects)),
                    format_func=lambda i: st.session_state.objects[i],
                    key="obj_idx_select"
                )
            
            if st.button("➕ Add Relationship", key="add_rel"):
                if subj_idx != obj_idx:
                    new_rel = [subj_idx, rel_name, obj_idx]
                    if new_rel not in st.session_state.relationships:
                        st.session_state.relationships.append(new_rel)
                        st.success(f"Added: {st.session_state.objects[subj_idx]} --[{rel_name}]--> {st.session_state.objects[obj_idx]}")
                else:
                    st.error("Subject and object must be different!")
        else:
            st.info("Add at least 2 objects to create relationships")
        
        # Display current relationships
        if st.session_state.relationships:
            st.markdown("**Current relationships:**")
            for i, (s_idx, rel, o_idx) in enumerate(st.session_state.relationships):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i}. {st.session_state.objects[s_idx]} --[{rel}]--> {st.session_state.objects[o_idx]}")
                with col2:
                    if st.button("❌", key=f"remove_rel_{i}"):
                        st.session_state.relationships.pop(i)
                        st.rerun()
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear Query"):
            st.session_state.objects = []
            st.session_state.relationships = []
            st.session_state.results = None
            st.rerun()
    
    # ==================== MAIN: Your Phrase & Results ====================
    
    # Your Phrase Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 📌 Your Phrase")
        
        if not st.session_state.objects:
            st.info("👈 Select objects from the sidebar to build your query")
        else:
            # Display as structured text
            phrase_text = "**Objects:** " + ", ".join(st.session_state.objects)
            st.markdown(phrase_text)
            
            if st.session_state.relationships:
                st.markdown("**Relationships:**")
                for s_idx, rel, o_idx in st.session_state.relationships:
                    st.markdown(f"- {st.session_state.objects[s_idx]} --**{rel}**--> {st.session_state.objects[o_idx]}")
    
    with col2:
        st.write("")
        st.write("")
        # Search button
        if st.button("🔍 Search", key="search_btn", use_container_width=True):
            if not st.session_state.objects:
                st.error("❌ Add at least one object first!")
            else:
                query_graph = build_scene_graph()
                st.session_state.results = retrieve_images(query_graph, top_k=10)
    
    st.divider()
    
    # Results Section
    if st.session_state.results:
        st.markdown("## 🎯 Top 10 Results")
        
        # Results table
        st.markdown("### Ranking")
        result_data = []
        for i, (fname, score) in enumerate(st.session_state.results, 1):
            result_data.append({"Rank": i, "Image": fname, "Score": f"{score:.4f}"})
        
        st.dataframe(result_data, use_container_width=True)
        
        # Results gallery
        st.markdown("### Gallery")
        
        cols = st.columns(5)
        for i, (fname, score) in enumerate(st.session_state.results):
            with cols[i % 5]:
                img_path = find_image_path(fname)
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Rank {i+1}\nScore: {score:.4f}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading {fname}: {e}")
                else:
                    st.warning(f"Image not found: {fname}")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **Scene Graph Image Retrieval** | CRF-based visual search
    """)


if __name__ == "__main__":
    main()
