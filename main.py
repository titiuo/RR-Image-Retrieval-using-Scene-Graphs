#!/usr/bin/env python3
"""
Scene Graph Image Retrieval - IPOL Demo Entry Point

Usage:
    python3 main.py '<objects>' '<relationships>'

Arguments:
    objects       Comma-separated object names:   "person, horse, tree"
    relationships Comma-separated predicate words: "on, near"
                  Each predicate is applied to the consecutive object pair:
                    "on"       → [[0, "on",  1]]
                    "on, near" → [[0, "on",  1], [1, "near", 2]]
                  Leave empty "" for object-only search.
"""

import os
import sys
import base64
import pickle
import traceback
from io import BytesIO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIN_DIR    = os.environ.get("bin", os.path.dirname(os.path.abspath(__file__))).rstrip("/")
ASSETS_DIR = os.environ.get("ASSETS_DIR", "/assets")

sys.path.insert(0, BIN_DIR)
sys.path.insert(0, os.path.join(BIN_DIR, "CRF"))

UNARY_MODEL_PATH   = os.path.join(BIN_DIR,    "CRF", "trained_models", "unary_potentials.pkl")
BINARY_MODEL_PATH  = os.path.join(BIN_DIR,    "CRF", "trained_models", "binary_potentials.pkl")
BINARY_PLATT_PATH  = os.path.join(BIN_DIR,    "CRF", "trained_models", "platt_params_binary_potentials.pkl")
FEATURE_CACHE_PATH = os.path.join(ASSETS_DIR, "data", "rcnn_test_features.pkl")
IMAGE_ROOTS = [
    os.path.join(ASSETS_DIR, "sg_dataset", "sg_test_images"),
    os.path.join(ASSETS_DIR, "sg_dataset", "sg_train_images"),
    os.path.join(ASSETS_DIR, "sg_dataset", "images"),
]

TOP_K = 4

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
try:
    from CRF.test_pipeline import CRFInference
except ImportError:
    try:
        from test_pipeline import CRFInference
    except ImportError as e:
        print(f"[ERROR] Could not import CRFInference: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image_path(image_name):
    for root in IMAGE_ROOTS:
        path = os.path.join(root, image_name)
        if os.path.exists(path):
            return path
    return None


def load_feature_cache():
    if not os.path.exists(FEATURE_CACHE_PATH):
        print(f"[ERROR] Feature cache not found: {FEATURE_CACHE_PATH}", file=sys.stderr)
        return None
    print(f"[INFO] Loading feature cache: {FEATURE_CACHE_PATH}", file=sys.stderr)
    with open(FEATURE_CACHE_PATH, "rb") as f:
        try:
            header = pickle.load(f)
        except EOFError:
            return None
        if isinstance(header, dict) and header.get("_stream") is True:
            cache = {}
            while True:
                try:
                    entry = pickle.load(f)
                except EOFError:
                    break
                if isinstance(entry, dict) and entry.get("image"):
                    cache[entry["image"]] = {
                        "boxes":    entry.get("boxes", []),
                        "features": entry.get("features"),
                    }
            print(f"[INFO] Stream cache: {len(cache)} images", file=sys.stderr)
            return cache
        if isinstance(header, dict):
            print(f"[INFO] Dict cache: {len(header)} images", file=sys.stderr)
            return header
    return None

# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

def parse_objects(raw: str) -> list:
    """'person, horse, tree' → ['person', 'horse', 'tree']"""
    return [o.strip() for o in raw.split(",") if o.strip()]


def parse_relationships(raw: str, objects: list) -> list:
    """
    Takes comma-separated predicate words and pairs them with consecutive
    object indices.

    Examples:
      objects = ['person', 'horse', 'tree']
      raw = 'on'        → [[0, 'on',   1]]
      raw = 'on, near'  → [[0, 'on',   1], [1, 'near', 2]]
      raw = ''          → []

    If more predicates than consecutive pairs, extras are ignored.
    If fewer predicates than pairs, only the given ones are used.
    """
    if not raw or not raw.strip():
        return []

    predicates = [p.strip() for p in raw.split(",") if p.strip()]
    n_objects  = len(objects)

    if n_objects < 2:
        print("[WARN] Need at least 2 objects to define a relationship.", file=sys.stderr)
        return []

    rels = []
    for k, pred in enumerate(predicates):
        subj = k
        obj  = k + 1
        if obj >= n_objects:
            print(f"[WARN] Predicate '{pred}' ignored — no object at index {obj}.", file=sys.stderr)
            break
        rels.append([subj, pred, obj])
        print(f"[INFO]   {objects[subj]} --[{pred}]--> {objects[obj]}", file=sys.stderr)

    return rels

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(query_graph: dict, feature_cache: dict) -> list:
    for label, path in [("Unary model",  UNARY_MODEL_PATH),
                        ("Binary model", BINARY_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("[INFO] Loading CRF models...", file=sys.stderr)
    pipeline = CRFInference(UNARY_MODEL_PATH, BINARY_MODEL_PATH, BINARY_PLATT_PATH)

    results = []
    total = len(feature_cache)
    report_every = max(1, total // 20)

    print(f"[INFO] Scoring {total} images...", file=sys.stderr)
    for idx, (fname, cache_data) in enumerate(feature_cache.items()):
        if idx % report_every == 0:
            print(f"[INFO]   {idx}/{total}", file=sys.stderr)
        boxes    = cache_data.get("boxes", [])
        features = cache_data.get("features", None)
        if features is None or len(features) == 0:
            continue
        try:
            score, _ = pipeline.beam_search(boxes, features, query_graph)
            if score > -9000:
                results.append((fname, score, boxes))
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}", file=sys.stderr)

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:TOP_K]

    print(f"[INFO] Top {len(top)} results:", file=sys.stderr)
    for rank, (fname, score, _) in enumerate(top, 1):
        print(f"[INFO]   #{rank}  score={score:.4f}  {fname}", file=sys.stderr)

    return top

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def image_to_data_uri(img_path: str):
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((300, 300))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[WARN] Could not encode {img_path}: {e}", file=sys.stderr)
        return None


def write_html(query_graph: dict, results: list, out_path: str):
    objects = query_graph["objects"]
    rels    = query_graph["relationships"]

    obj_tags = " ".join(f'<span class="tag">{o}</span>' for o in objects)

    if rels:
        rel_rows = "".join(
            f"<tr><td>{objects[s]}</td><td class='pred'>{r}</td><td>{objects[o]}</td></tr>"
            for s, r, o in rels
        )
        rel_block = (
            '<table class="rel-table">'
            '<thead><tr><th>Subject</th><th>Predicate</th><th>Object</th></tr></thead>'
            f'<tbody>{rel_rows}</tbody></table>'
        )
    else:
        rel_block = "<p class='muted'>No relationships — object-only search.</p>"

    cards = ""
    for rank, (fname, score, _) in enumerate(results, 1):
        img_path = find_image_path(fname)
        src = image_to_data_uri(img_path) if (img_path and os.path.exists(img_path)) else None
        img_tag = (
            f'<img src="{src}" alt="{fname}" loading="lazy">'
            if src else '<div class="no-img">Image not found</div>'
        )
        cards += (
            f'<div class="card">{img_tag}'
            f'<div class="caption">'
            f'<span class="rank">#{rank}</span>'
            f'<span class="fname" title="{fname}">{fname}</span><br>'
            f'<span class="score">score: {score:.4f}</span>'
            f'</div></div>\n'
        )

    gallery = f'<div class="gallery">{cards}</div>' if cards else "<p class='muted'>No results.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scene Graph Retrieval — Results</title>
<style>
  *,*::before,*::after{{box-sizing:border-box}}
  body{{font-family:"Segoe UI",Arial,sans-serif;margin:0;padding:24px 32px;background:#f0f2f5;color:#222}}
  h1{{font-size:1.5rem;margin-bottom:4px}}
  h2{{font-size:1rem;margin:16px 0 8px;color:#555;text-transform:uppercase;letter-spacing:.05em}}
  .query-box{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px 20px;
              margin-bottom:28px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .tag{{display:inline-block;background:#e8f0fe;color:#1a56db;border-radius:4px;
        padding:3px 11px;margin:2px 3px 2px 0;font-size:.9rem;font-weight:600}}
  .rel-table{{border-collapse:collapse;margin-top:8px;font-size:.9rem}}
  .rel-table th,.rel-table td{{border:1px solid #e0e0e0;padding:5px 16px}}
  .rel-table thead{{background:#f7f7f7;font-weight:600}}
  .pred{{font-style:italic;color:#666;text-align:center}}
  .muted{{color:#888;font-style:italic;margin-top:4px}}
  .gallery{{display:flex;flex-wrap:wrap;gap:16px}}
  .card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;width:220px;padding:8px;
         box-shadow:0 2px 6px rgba(0,0,0,.08);transition:transform .15s ease}}
  .card:hover{{transform:translateY(-4px)}}
  .card img{{width:100%;height:165px;object-fit:cover;border-radius:5px;display:block}}
  .no-img{{width:100%;height:165px;background:#eee;border-radius:5px;display:flex;
           align-items:center;justify-content:center;color:#aaa;font-size:.8rem}}
  .caption{{font-size:.78rem;margin-top:7px;color:#555;line-height:1.4}}
  .rank{{font-weight:700;color:#1a56db;margin-right:5px;font-size:.85rem}}
  .fname{{display:inline-block;max-width:175px;white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis;vertical-align:bottom;font-size:.78rem}}
  .score{{color:#888;font-size:.75rem}}
</style>
</head>
<body>
<h1>&#128269; Scene Graph Image Retrieval &mdash; Results</h1>
<div class="query-box">
  <h2>Objects</h2>{obj_tags}
  <h2>Relationships</h2>{rel_block}
</div>
<h2>Top {len(results)} Retrieved Image(s)</h2>
{gallery}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML written → {out_path}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"[INFO] sys.argv = {sys.argv}", file=sys.stderr)
    print(f"[INFO] BIN_DIR = {BIN_DIR}", file=sys.stderr)
    print(f"[INFO] FEATURE_CACHE_PATH = {FEATURE_CACHE_PATH}", file=sys.stderr)

    if len(sys.argv) < 3:
        print("Usage: python3 main.py '<objects>' '<relationships>'", file=sys.stderr)
        sys.exit(1)

    raw_objects = sys.argv[1]
    raw_rels    = sys.argv[2]

    if raw_objects.startswith("$"):
        print(f"[ERROR] Parameter substitution failed — got '{raw_objects}' literally.", file=sys.stderr)
        sys.exit(1)

    objects = parse_objects(raw_objects)
    if not objects:
        print("[ERROR] No valid objects.", file=sys.stderr)
        sys.exit(1)

    relationships = parse_relationships(raw_rels, objects)
    query_graph   = {"objects": objects, "relationships": relationships}

    print(f"[INFO] Query → objects={objects}", file=sys.stderr)
    print(f"[INFO] Query → relationships={relationships}", file=sys.stderr)

    feature_cache = load_feature_cache()
    if not feature_cache:
        print("[ERROR] Feature cache empty or missing.", file=sys.stderr)
        sys.exit(1)

    results  = run_retrieval(query_graph, feature_cache)
    out_path = os.path.join(os.getcwd(), "results.html")
    write_html(query_graph, results, out_path)
    print(f"[INFO] Done — {len(results)} result(s).", file=sys.stderr)


if __name__ == "__main__":
    main()