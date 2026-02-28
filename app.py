#!/usr/bin/env python
"""
Interactive Scene Graph Query Retrieval - Streamlit App
"""

import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from plot import visualize_graph, visualize_graph_html, draw_bboxes_on_image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'data'))
SG_DIR   = os.path.abspath(os.path.join(ROOT_DIR, '..', 'sg_dataset'))

OBJECT_CLASSES_CSV    = os.path.join(DATA_DIR, 'object_classes.csv')
ATTRIBUTE_CLASSES_CSV = os.path.join(DATA_DIR, 'attribute_classes.csv')
RELATION_CLASSES_CSV  = os.path.join(DATA_DIR, 'relation_classes.csv')

try:
    from CRF import train_unary
    from graph import load_scene_graphs, build_predicted_scene_graph, build_predicted_scene_graph_with_gt_labels
    from test_pipeline import (
        CRFInference,
        UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH,
        TEST_ANNOTATIONS_PATH, FEATURE_CACHE_PATH,
        find_image_path, load_feature_cache,
    )
except ImportError as e:
    st.error(f"Import Error: {e}")
    sys.exit(1)


# ── Styling ────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #FFFFFF;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1400px;
    }

    /* Header */
    .sg-header {
        display: flex;
        align-items: baseline;
        gap: 16px;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 20px;
        margin-bottom: 32px;
    }
    .sg-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #f7fafc;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .sg-header .sg-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #FFFFFF;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Section labels */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #73FBFD;
        margin-bottom: 10px;
        margin-top: 0;
    }

    /* Panel accent borders */
    [data-testid="stVerticalBlock"] .panel-obj  > div:first-child { border-top: 2px solid #63b3ed !important; }
    [data-testid="stVerticalBlock"] .panel-attr > div:first-child { border-top: 2px solid #68d391 !important; }
    [data-testid="stVerticalBlock"] .panel-rel  > div:first-child { border-top: 2px solid #f6ad55 !important; }

    /* Tag pills */
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 9px;
        border-radius: 4px;
        border: 1px solid;
        white-space: nowrap;
        margin: 2px 0;
    }
    .tag-obj  { color: #90cdf4; border-color: #2c5282; background: #1a2f4a; }
    .tag-attr { color: #9ae6b4; border-color: #22543d; background: #1a3a2a; }
    .tag-rel  { color: #fbd38d; border-color: #7b341e; background: #3a2010; }
    .tag-empty { color:  #73FBFD; border-color: #2d3748; background: transparent; font-style: italic; }

    /* Phrase block */
    .phrase-block {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        line-height: 2;
        color: #a0aec0;
        background: #111827;
        border-radius: 6px;
        padding: 14px 18px;
        min-height: 56px;
        border: 1px solid #1f2937;
    }
    .phrase-block .ph-obj   { color: #90cdf4; font-weight: 500; }
    .phrase-block .ph-attr  { color: #9ae6b4; }
    .phrase-block .ph-rel   { color: #fbd38d; }
    .phrase-block .ph-arrow { color: #FFFFFF; }
    .phrase-empty {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #2d3748;
        padding: 16px 18px;
        font-style: italic;
        background: #111827;
        border-radius: 6px;
        border: 1px solid #1f2937;
    }

    /* Result metadata */
    .result-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 4px 2px 4px;
    }
    .result-rank {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #68d391;
    }
    .result-fname {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding-bottom: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: #1a2535 !important;
        color: #90cdf4 !important;
        border: 1px solid #2b4070 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #243352 !important;
        border-color: #63b3ed !important;
        color: #bee3f8 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #111827 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Progress bar */
    .stProgress > div > div { background-color: #63b3ed !important; }

    /* Divider */
    hr { border-color: #1f2937 !important; margin: 24px 0 !important; }

    /* Obj-name label above attr list */
    .attr-group-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #FFFFFF;
        margin: 10px 0 4px 0;
        letter-spacing: 0.06em;
    }

    /* Remove-button column — visually subtle */
    div[data-testid="column"]:last-child .stButton > button {
        background: transparent !important;
        border: 1px solid #2d3748 !important;
        color: #FFFFFF !important;
        padding: 2px 6px !important;
        font-size: 0.7rem !important;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        border-color: #e53e3e !important;
        color: #fc8181 !important;
        background: #2d1515 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_classes_from_csv():
    obj_classes, attr_classes, rel_classes = [], [], []

    if os.path.exists(OBJECT_CLASSES_CSV):
        df = pd.read_csv(OBJECT_CLASSES_CSV)
        obj_classes = sorted(df.iloc[:, 0].astype(str).tolist())
    else:
        st.warning(f"Object classes CSV not found: {OBJECT_CLASSES_CSV}")

    if os.path.exists(ATTRIBUTE_CLASSES_CSV):
        df = pd.read_csv(ATTRIBUTE_CLASSES_CSV)
        attr_classes = sorted(df.iloc[:, 0].astype(str).tolist())
    else:
        st.warning(f"Attribute classes CSV not found: {ATTRIBUTE_CLASSES_CSV}")

    if os.path.exists(RELATION_CLASSES_CSV):
        df = pd.read_csv(RELATION_CLASSES_CSV)
        rel_classes = sorted(df.iloc[:, 0].astype(str).tolist())
    else:
        st.warning(f"Relation classes CSV not found: {RELATION_CLASSES_CSV}")

    return obj_classes, attr_classes, rel_classes


@st.cache_resource
def load_models():
    obj_classes, attr_classes, rel_classes = load_classes_from_csv()
    feature_cache = load_feature_cache()
    pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)
    
    # Load scene graphs with vocabularies
    try:
        scene_graphs, obj_vocab, attr_vocab, rel_vocab = load_scene_graphs(TEST_ANNOTATIONS_PATH)
        # Create a mapping from filename to scene graph for quick lookup
        scene_graph_dict = {sg.filename: sg for sg in scene_graphs}
    except Exception as e:
        st.warning(f"Could not load scene graphs: {e}")
        scene_graph_dict = {}
        obj_vocab = attr_vocab = rel_vocab = None
    
    return {
        'feature_cache': feature_cache,
        'pipeline':      pipeline,
        'obj_classes':   obj_classes,
        'attr_classes':  attr_classes,
        'rel_classes':   rel_classes,
        'scene_graph_dict': scene_graph_dict,
        'obj_vocab': obj_vocab,
        'attr_vocab': attr_vocab,
        'rel_vocab': rel_vocab,
    }


# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    defaults = {
        'objects':           [],
        'object_attributes': {},
        'relationships':     [],
        'results':           None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Core logic ─────────────────────────────────────────────────────────────────

def build_scene_graph():
    return {
        'objects':       st.session_state.objects,
        'relationships': st.session_state.relationships,
    }


def retrieve_images(query_graph, top_k=4):
    """Run retrieval — identical logic to original."""
    if not query_graph['objects']:
        st.error("❌ Add at least one object first.")
        return None

    feature_cache = st.session_state.models['feature_cache']
    pipeline      = st.session_state.models['pipeline']

    if not feature_cache:
        st.error("❌ Feature cache is empty. Run the precompute script first.")
        return None

    results      = []
    progress_bar = st.progress(0)
    status_text  = st.empty()
    total        = len(feature_cache)

    try:
        for idx, (img_filename, cache_data) in enumerate(feature_cache.items()):
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Scoring {idx + 1} / {total} …")
            try:
                boxes    = cache_data.get('boxes', [])
                features = cache_data.get('features', None)
                if features is None or len(features) == 0:
                    continue
                score, _ = pipeline.beam_search(boxes, features, query_graph)
                if score > -9000:
                    # Store filename, score, and boxes for visualization
                    results.append((img_filename, score, boxes))
            except Exception:
                continue
    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

    if not results:
        st.warning("No results found. Try a different query.")
        return None

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Phrase renderer ────────────────────────────────────────────────────────────

def render_phrase_html():
    objects = st.session_state.objects
    attrs   = st.session_state.object_attributes
    rels    = st.session_state.relationships

    if not objects:
        return '<div class="phrase-empty">← Add objects to build your query</div>'

    # Objects row
    obj_parts = []
    for i, obj in enumerate(objects):
        obj_html  = f'<span class="ph-obj">{obj}</span>'
        obj_attrs = attrs.get(i, [])
        if obj_attrs:
            attr_html  = ' '.join(f'<span class="ph-attr">[{a}]</span>' for a in obj_attrs)
            obj_html  += f' {attr_html}'
        obj_parts.append(obj_html)

    objects_line = ' <span class="ph-arrow">·</span> '.join(obj_parts)

    # Relationships rows
    rel_lines = ''
    for s_idx, rel, o_idx in rels:
        if s_idx < len(objects) and o_idx < len(objects):
            rel_lines += (
                f'<br><span class="ph-arrow">  ↳ </span>'
                f'<span class="ph-obj">{objects[s_idx]}</span>'
                f'<span class="ph-arrow"> ── </span>'
                f'<span class="ph-rel">[{rel}]</span>'
                f'<span class="ph-arrow"> ──▶ </span>'
                f'<span class="ph-obj">{objects[o_idx]}</span>'
            )

    return f'<div class="phrase-block">{objects_line}{rel_lines}</div>'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Scene Graph Retrieval",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    init_state()
    models = st.session_state.models

    # Header
    st.markdown("""
    <div class="sg-header">
        <h1>⬡ Scene Graph Retrieval</h1>
        <span class="sg-subtitle">CRF · Visual Search · Image Retrieval</span>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ROW 1 — Three builder panels
    # ══════════════════════════════════════════════════════════
    col_obj, col_attr, col_rel = st.columns(3, gap="medium")

    # ── ① Objects ────────────────────────────────────────────
    with col_obj:
        st.markdown('<p class="section-label">① Objects</p>', unsafe_allow_html=True)
        with st.container(border=True):

            sel_obj = st.selectbox(
                "Object class", models['obj_classes'],
                label_visibility="collapsed", key="sel_obj"
            )
            if st.button("➕  Add object", key="add_obj", use_container_width=True):
                if sel_obj and sel_obj not in st.session_state.objects:
                    st.session_state.objects.append(sel_obj)
                    st.rerun()

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if not st.session_state.objects:
                st.markdown('<span class="tag tag-empty">none yet</span>', unsafe_allow_html=True)
            else:
                for i, obj in enumerate(st.session_state.objects):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f'<span class="tag tag-obj">⬡ {i}: {obj}</span>', unsafe_allow_html=True)
                    with c2:
                        if st.button("✕", key=f"rm_obj_{i}", help=f"Remove {obj}"):
                            st.session_state.objects.pop(i)
                            # Clean up linked attributes
                            if i in st.session_state.object_attributes:
                                del st.session_state.object_attributes[i]
                            new_attrs = {}
                            for k, v in st.session_state.object_attributes.items():
                                new_attrs[k - 1 if k > i else k] = v
                            st.session_state.object_attributes = new_attrs
                            # Clean up linked relationships
                            st.session_state.relationships = [
                                [s if s < i else s - 1, r, o if o < i else o - 1]
                                for s, r, o in st.session_state.relationships
                                if s != i and o != i
                            ]
                            st.rerun()

    # ── ② Attributes ─────────────────────────────────────────
    with col_attr:
        st.markdown('<p class="section-label">② Attributes</p>', unsafe_allow_html=True)
        with st.container(border=True):

            if st.session_state.objects:
                attr_target = st.selectbox(
                    "Apply to",
                    range(len(st.session_state.objects)),
                    format_func=lambda i: f"{i}: {st.session_state.objects[i]}",
                    label_visibility="collapsed", key="attr_target"
                )
                sel_attr = st.selectbox(
                    "Attribute class", models['attr_classes'],
                    label_visibility="collapsed", key="sel_attr"
                )
                if st.button("➕  Add attribute", key="add_attr", use_container_width=True):
                    bucket = st.session_state.object_attributes.setdefault(attr_target, [])
                    if sel_attr and sel_attr not in bucket:
                        bucket.append(sel_attr)
                        st.rerun()

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                has_any = any(
                    v for v in st.session_state.object_attributes.values()
                )
                if not has_any:
                    st.markdown('<span class="tag tag-empty">none yet</span>', unsafe_allow_html=True)
                else:
                    for obj_i in sorted(st.session_state.object_attributes.keys()):
                        att_list = st.session_state.object_attributes[obj_i]
                        if not att_list or obj_i >= len(st.session_state.objects):
                            continue
                        obj_name = st.session_state.objects[obj_i]
                        st.markdown(
                            f'<p class="attr-group-label">{obj_name}</p>',
                            unsafe_allow_html=True
                        )
                        for j, attr in enumerate(att_list):
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f'<span class="tag tag-attr">◈ {attr}</span>', unsafe_allow_html=True)
                            with c2:
                                if st.button("✕", key=f"rm_attr_{obj_i}_{j}", help=f"Remove {attr}"):
                                    st.session_state.object_attributes[obj_i].pop(j)
                                    st.rerun()
            else:
                st.markdown('<span class="tag tag-empty">add objects first</span>', unsafe_allow_html=True)

    # ── ③ Relationships ───────────────────────────────────────
    with col_rel:
        st.markdown('<p class="section-label">③ Relationships</p>', unsafe_allow_html=True)
        with st.container(border=True):

            if len(st.session_state.objects) >= 2:
                r1, r2, r3 = st.columns(3)
                with r1:
                    subj_idx = st.selectbox(
                        "From",
                        range(len(st.session_state.objects)),
                        format_func=lambda i: st.session_state.objects[i],
                        key="subj_sel"
                    )
                with r2:
                    sel_rel = st.selectbox(
                        "Relation", models['rel_classes'],
                        key="sel_rel"
                    )
                with r3:
                    obj_idx = st.selectbox(
                        "To",
                        range(len(st.session_state.objects)),
                        format_func=lambda i: st.session_state.objects[i],
                        key="obj_sel"
                    )

                if st.button("➕  Add relationship", key="add_rel", use_container_width=True):
                    if subj_idx == obj_idx:
                        st.warning("Subject and object must differ.")
                    else:
                        new_rel = [subj_idx, sel_rel, obj_idx]
                        if new_rel not in st.session_state.relationships:
                            st.session_state.relationships.append(new_rel)
                            st.rerun()

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                if not st.session_state.relationships:
                    st.markdown('<span class="tag tag-empty">none yet</span>', unsafe_allow_html=True)
                else:
                    for k, (s, r, o) in enumerate(st.session_state.relationships):
                        if s >= len(st.session_state.objects) or o >= len(st.session_state.objects):
                            continue
                        label = f"{st.session_state.objects[s]} [{r}]▶ {st.session_state.objects[o]}"
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f'<span class="tag tag-rel">↔ {label}</span>', unsafe_allow_html=True)
                        with c2:
                            if st.button("✕", key=f"rm_rel_{k}", help="Remove"):
                                st.session_state.relationships.pop(k)
                                st.rerun()
            else:
                st.markdown('<span class="tag tag-empty">add ≥ 2 objects first</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ROW 2 — Phrase display + action buttons
    # ══════════════════════════════════════════════════════════
    st.markdown('<p class="section-label">Your phrase</p>', unsafe_allow_html=True)

    phrase_col, btn_col = st.columns([4, 1], gap="medium")

    with phrase_col:
        st.markdown(render_phrase_html(), unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🔍  Search", key="search_btn", use_container_width=True):
            if not st.session_state.objects:
                st.error("Add at least one object first.")
            else:
                st.session_state.results = retrieve_images(build_scene_graph(), top_k=4)
                st.rerun()
        if st.button("⟳  Clear", key="clear_btn", use_container_width=True):
            st.session_state.objects           = []
            st.session_state.object_attributes = {}
            st.session_state.relationships     = []
            st.session_state.results           = None
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # ROW 3 — Results grid
    # ══════════════════════════════════════════════════════════
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)

        results = st.session_state.results
        st.markdown(
            f'<p class="section-label">Top {len(results)} results</p>',
            unsafe_allow_html=True
        )

        # Display each result as one row: predicted image on left, scene graph on right
        for rank, result in enumerate(results, 1):
            # Unpack result - can be 2-tuple (legacy) or 3-tuple (with boxes)
            if len(result) == 3:
                fname, score, boxes = result
            else:
                fname, score = result
                boxes = []
            
            col_img, col_sg = st.columns(2, gap="medium")
            
            # Left column: Predicted image with RCNN bounding boxes
            with col_img:
                st.markdown(f'<p style="font-size:0.8rem;color:#a0aec0;margin-bottom:8px;">'
                           f'<span style="color:#FFFFFF;font-weight:bold;">#{rank}</span> {fname} '
                           f'<span style="color:#68d391;">{score:.4f}</span></p>', 
                           unsafe_allow_html=True)
                img_path = find_image_path(fname)
                if img_path and os.path.exists(img_path):
                    try:
                        # Create predicted scene graph with GT labels matched by IoU
                        scene_graph_dict = models['scene_graph_dict']
                        obj_vocab = models['obj_vocab']
                        
                        if fname in scene_graph_dict and obj_vocab and boxes:
                            # Match predicted boxes to ground truth labels
                            gt_sg = scene_graph_dict[fname]
                            predicted_sg = build_predicted_scene_graph_with_gt_labels(
                                boxes, gt_sg, obj_vocab, iou_threshold=0.3
                            )
                        else:
                            # Fallback: use predicted boxes without labels
                            from graph import Vocabulary
                            fallback_vocab = Vocabulary()
                            fallback_vocab.add("detected_object")
                            predicted_sg_temp = build_predicted_scene_graph(boxes, filename=fname)
                            predicted_sg = predicted_sg_temp
                            obj_vocab = fallback_vocab
                        
                        if predicted_sg.objects:
                            # Draw predicted bounding boxes on the image with proper labels
                            img = draw_bboxes_on_image(img_path, predicted_sg, obj_vocab)
                        else:
                            img = Image.open(img_path)
                        
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.markdown(
                            '<div style="height:300px;background:#111827;border-radius:6px;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:#e53e3e;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
                            f'Load error: {str(e)[:40]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div style="height:300px;background:#111827;border-radius:6px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#2d3748;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
                        'Image not found</div>',
                        unsafe_allow_html=True
                    )
            
            # Right column: Ground truth scene graph visualization (for comparison)
            with col_sg:
                st.markdown(f'<p style="font-size:0.8rem;color:#a0aec0;margin-bottom:8px;visibility:hidden;">placeholder</p>', 
                           unsafe_allow_html=True)
                
                scene_graph_dict = models['scene_graph_dict']
                if fname in scene_graph_dict:
                    try:
                        # Use ground truth scene graph for comparison
                        gt_sg = scene_graph_dict[fname]
                        obj_vocab = models['obj_vocab']
                        attr_vocab = models['attr_vocab']
                        rel_vocab = models['rel_vocab']
                        
                        if all([obj_vocab, attr_vocab, rel_vocab, gt_sg.objects]):
                            html_content = visualize_graph_html(gt_sg, obj_vocab, attr_vocab, rel_vocab, height="500px")
                            st.components.v1.html(html_content, height=500, scrolling=True)
                        else:
                            st.markdown(
                                '<div style="height:400px;background:#111827;border-radius:6px;'
                                'display:flex;align-items:center;justify-content:center;'
                                'color:#2d3748;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
                                'Ground truth data incomplete</div>',
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.markdown(
                            '<div style="height:400px;background:#111827;border-radius:6px;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:#e53e3e;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
                            f'Error: {str(e)[:50]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div style="height:400px;background:#111827;border-radius:6px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#2d3748;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
                        'Ground truth not found</div>',
                        unsafe_allow_html=True
                    )
            
            st.markdown("---")


if __name__ == "__main__":
    main()