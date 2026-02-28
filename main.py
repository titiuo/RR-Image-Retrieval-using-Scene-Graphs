#!/usr/bin/env python3
"""
Scene Graph Image Retrieval - IPOL Demo Entry Point

Usage:
    python3 main.py '<objects>' '<relationships>'

Arguments:
    objects       Comma-separated list, e.g. "person, horse, tree"
    relationships Comma-separated triplets, e.g. "0 on 1, 0 next_to 2"
                  Format: <subj_idx> <predicate_words> <obj_idx>
                  Pass empty string "" for no relationships.

Output (written to cwd = IPOL work directory):
    results.html  — self-contained gallery displayed inline by IPOL
"""

import os
import sys
import base64
import traceback
from io import BytesIO

# ---------------------------------------------------------------------------
# Path resolution
#
# IPOL clones the repo into $bin and sets $bin as the working directory when
# building, but sets a *separate* work dir (cwd) at runtime for outputs.
#
# test_pipeline.py uses paths like:
#   ROOT_DIR  = dirname(abspath(__file__))          → .../CRF/
#   DATA_DIR  = ROOT_DIR/../data/
#   SG_DIR    = ROOT_DIR/../sg_dataset/
#   MODEL_DIR = ROOT_DIR/../model/
#
# Since IPOL calls  python3 $bin/main.py  (main.py is at repo root, not CRF/),
# we must point sys.path at the repo root AND patch the DATA_DIR / SG_DIR /
# MODEL_DIR constants BEFORE importing test_pipeline, so it resolves the
# large assets from /assets (downloaded at Docker build time).
# ---------------------------------------------------------------------------

# $bin is the repo root (IPOL sets this env var)
BIN_DIR    = os.environ.get("bin", os.path.dirname(os.path.abspath(__file__)))
# Large assets were extracted here at Docker build time (see Dockerfile)
ASSETS_DIR = os.environ.get("ASSETS_DIR", "/assets")

# Add repo root and CRF sub-package to path
sys.path.insert(0, BIN_DIR)
sys.path.insert(0, os.path.join(BIN_DIR, "CRF"))

# ---------------------------------------------------------------------------
# Patch asset paths BEFORE importing test_pipeline.
# test_pipeline reads these as module-level constants, so we override them
# via environment variables that it already respects (FEATURE_CACHE_PATH),
# and by monkey-patching after import for the rest.
# ---------------------------------------------------------------------------
FEATURE_CACHE_PATH = os.path.join(ASSETS_DIR, "data", "rcnn_test_features.pkl")
os.environ["FEATURE_CACHE_PATH"] = FEATURE_CACHE_PATH

try:
    import test_pipeline as tp
except ImportError as e:
    print(f"[ERROR] Could not import test_pipeline: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# Patch path constants so helpers (find_image_path, load_feature_cache, etc.)
# look inside /assets instead of relative to the repo.
tp.DATA_DIR  = os.path.join(ASSETS_DIR, "data")
tp.SG_DIR    = os.path.join(ASSETS_DIR, "sg_dataset")
tp.MODEL_DIR = os.path.join(ASSETS_DIR, "model")

# trained_models/ IS in the repo (committed) — keep pointing there
tp.TRAINED_DIR = os.path.join(BIN_DIR, "CRF", "trained_models")

# Re-derive file paths that depend on the dirs above
tp.FEATURE_CACHE_PATH        = FEATURE_CACHE_PATH
tp.TEST_ANNOTATIONS_PATH     = os.path.join(tp.SG_DIR, "sg_test_annotations.json")
tp.UNARY_MODEL_PATH          = os.path.join(tp.TRAINED_DIR, "unary_potentials.pkl")
tp.BINARY_MODEL_PATH         = os.path.join(tp.TRAINED_DIR, "binary_potentials.pkl")
tp.BINARY_PLATT_PATH         = os.path.join(tp.TRAINED_DIR, "platt_params_binary_potentials.pkl")
tp.IMAGE_ROOT_CANDIDATES     = [
    os.path.join(tp.SG_DIR, "sg_test_images"),
    os.path.join(tp.SG_DIR, "sg_train_images"),
    os.path.join(tp.SG_DIR, "images"),
]

TOP_K = 10


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_objects(raw: str) -> list:
    return [o.strip() for o in raw.split(",") if o.strip()]


def parse_relationships(raw: str, n_objects: int) -> list:
    """
    '0 on 1, 0 next_to 2'  →  [[0, 'on', 1], [0, 'next_to', 2]]
    Each entry: <int> <predicate words…> <int>
    """
    if not raw or not raw.strip():
        return []
    rels = []
    for entry in raw.split(","):
        tokens = entry.strip().split()
        if len(tokens) < 3:
            print(f"[WARN] Skipping malformed relationship: '{entry.strip()}' "
                  f"(need: subj_idx predicate obj_idx)", file=sys.stderr)
            continue
        try:
            subj = int(tokens[0])
            obj  = int(tokens[-1])
            pred = " ".join(tokens[1:-1])
        except ValueError:
            print(f"[WARN] Indices must be integers in: '{entry.strip()}'", file=sys.stderr)
            continue
        if not (0 <= subj < n_objects and 0 <= obj < n_objects):
            print(f"[WARN] Index out of range in: '{entry.strip()}' "
                  f"(valid range: 0–{n_objects - 1})", file=sys.stderr)
            continue
        if subj == obj:
            print(f"[WARN] Subject == object in: '{entry.strip()}', skipping.", file=sys.stderr)
            continue
        rels.append([subj, pred, obj])
    return rels


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(query_graph: dict) -> list:
    print("[INFO] Loading feature cache...", file=sys.stderr)
    feature_cache = tp.load_feature_cache()
    if not feature_cache:
        print(f"[ERROR] Feature cache not found or empty at: {tp.FEATURE_CACHE_PATH}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(feature_cache)} images from cache.", file=sys.stderr)

    print("[INFO] Loading CRF models...", file=sys.stderr)
    pipeline = tp.CRFInference(tp.UNARY_MODEL_PATH, tp.BINARY_MODEL_PATH, tp.BINARY_PLATT_PATH)

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
                results.append((fname, score))
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}", file=sys.stderr)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:TOP_K]


# ---------------------------------------------------------------------------
# HTML output (self-contained, base64-embedded thumbnails)
# ---------------------------------------------------------------------------

def image_to_data_uri(img_path: str):
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((300, 300))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[WARN] Could not encode {img_path}: {e}", file=sys.stderr)
        return None


def write_html(query_graph: dict, results: list, out_path: str):
    objects = query_graph["objects"]
    rels    = query_graph["relationships"]

    # --- Query summary ---
    obj_tags = " ".join(f'<span class="tag">{o}</span>' for o in objects)

    if rels:
        rel_rows = "".join(
            f"<tr><td>{objects[s]}</td>"
            f"<td class='pred'>{r}</td>"
            f"<td>{objects[o]}</td></tr>"
            for s, r, o in rels
        )
        rel_block = (
            '<table class="rel-table">'
            '<thead><tr><th>Subject</th><th>Predicate</th><th>Object</th></tr></thead>'
            f'<tbody>{rel_rows}</tbody></table>'
        )
    else:
        rel_block = "<p class='muted'>No relationships specified — object-only search.</p>"

    # --- Gallery cards ---
    if results:
        cards = ""
        for rank, (fname, score) in enumerate(results, 1):
            img_path = tp.find_image_path(fname)
            src = None
            if img_path and os.path.exists(img_path):
                src = image_to_data_uri(img_path)

            if src:
                img_tag = f'<img src="{src}" alt="{fname}" loading="lazy">'
            else:
                img_tag = (
                    f'<div class="no-img">'
                    f'<span>Image not found</span>'
                    f'<small>{fname}</small>'
                    f'</div>'
                )

            cards += (
                f'<div class="card">'
                f'{img_tag}'
                f'<div class="caption">'
                f'<span class="rank">#{rank}</span>'
                f'<span class="fname" title="{fname}">{fname}</span><br>'
                f'<span class="score">score: {score:.4f}</span>'
                f'</div></div>\n'
            )
        gallery_html = f'<div class="gallery">\n{cards}</div>'
    else:
        gallery_html = "<p class='muted'>No results found for this query.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scene Graph Retrieval — Results</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 0; padding: 24px 32px;
    background: #f0f2f5; color: #222;
  }}
  h1  {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2  {{ font-size: 1rem; margin: 16px 0 8px; color: #555; text-transform: uppercase;
         letter-spacing: .05em; }}
  .query-box {{
    background: #fff; border: 1px solid #ddd; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.07);
  }}
  .tag {{
    display: inline-block; background: #e8f0fe; color: #1a56db;
    border-radius: 4px; padding: 3px 11px; margin: 2px 3px 2px 0;
    font-size: .9rem; font-weight: 600;
  }}
  .rel-table {{
    border-collapse: collapse; margin-top: 8px; font-size: .9rem;
  }}
  .rel-table th, .rel-table td {{
    border: 1px solid #e0e0e0; padding: 5px 16px;
  }}
  .rel-table thead {{ background: #f7f7f7; font-weight: 600; }}
  .pred {{ font-style: italic; color: #666; text-align: center; }}
  .muted {{ color: #888; font-style: italic; margin-top: 4px; }}
  .gallery {{
    display: flex; flex-wrap: wrap; gap: 16px;
  }}
  .card {{
    background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
    width: 200px; padding: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,.08);
    transition: transform .15s ease;
  }}
  .card:hover {{ transform: translateY(-4px); box-shadow: 0 6px 16px rgba(0,0,0,.12); }}
  .card img {{
    width: 100%; height: 155px; object-fit: cover; border-radius: 5px; display: block;
  }}
  .no-img {{
    width: 100%; height: 155px; background: #f0f0f0; border-radius: 5px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; color: #aaa; font-size: .8rem; gap: 4px;
    text-align: center; padding: 8px;
  }}
  .caption {{ font-size: .78rem; margin-top: 7px; color: #555; line-height: 1.4; }}
  .rank  {{ font-weight: 700; color: #1a56db; margin-right: 5px; font-size: .85rem; }}
  .fname {{
    display: inline-block; max-width: 155px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    vertical-align: bottom; font-size: .78rem;
  }}
  .score {{ color: #888; font-size: .75rem; }}
</style>
</head>
<body>

<h1>&#128269; Scene Graph Image Retrieval &mdash; Results</h1>

<div class="query-box">
  <h2>Objects</h2>
  {obj_tags}
  <h2>Relationships</h2>
  {rel_block}
</div>

<h2>Top {len(results)} Retrieved Image(s)</h2>
{gallery_html}

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML written → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 main.py '<objects>' '<relationships>'\n"
            "  objects:       'person, horse, tree'\n"
            "  relationships: '0 on 1, 0 next_to 2'  (or empty string '')",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate required model files before doing any heavy work
    missing = []
    for label, path in [
        ("Unary model",   tp.UNARY_MODEL_PATH),
        ("Binary model",  tp.BINARY_MODEL_PATH),
        ("Feature cache", tp.FEATURE_CACHE_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("[ERROR] The following required files are missing:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)

    objects = parse_objects(sys.argv[1])
    if not objects:
        print("[ERROR] No valid objects found in first argument.", file=sys.stderr)
        sys.exit(1)

    relationships = parse_relationships(sys.argv[2], len(objects))
    query_graph   = {"objects": objects, "relationships": relationships}

    print(f"[INFO] Query → objects={objects}, relationships={relationships}", file=sys.stderr)

    results  = run_retrieval(query_graph)
    out_path = os.path.join(os.getcwd(), "results.html")
    write_html(query_graph, results, out_path)

    print(f"[INFO] Done — {len(results)} result(s) written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()