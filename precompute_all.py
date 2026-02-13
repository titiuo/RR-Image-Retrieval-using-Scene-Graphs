import os
import pickle
import sys

try:
    from test_pipeline import (
        load_scene_graphs,
        extract_features,
        find_image_path,
        TEST_ANNOTATIONS_PATH,
    )
except ImportError:
    print("Error: Could not import from test_pipeline.")
    print("Run this script from the project root: python precompute_all.py")
    sys.exit(1)

OUTPUT_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "rcnn_test_features.pkl")
)


def precompute_all():
    print("Loading test annotations...")
    graphs, _, _, _ = load_scene_graphs(TEST_ANNOTATIONS_PATH)
    if not graphs:
        print("No graphs found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CACHE_PATH), exist_ok=True)

    print(f"Writing cache to {OUTPUT_CACHE_PATH}")
    with open(OUTPUT_CACHE_PATH, "wb") as f:
        pickle.dump({"_stream": True}, f)

        for idx, graph in enumerate(graphs, start=1):
            fname = graph.filename
            img_path = find_image_path(fname)
            if not img_path:
                print(f"[{idx}/{len(graphs)}] Missing image: {fname}")
                continue

            gt_boxes = []
            for obj in graph.objects:
                bbox = obj["bbox"]
                gt_boxes.append([bbox["x"], bbox["y"], bbox["w"], bbox["h"]])

            if not gt_boxes:
                print(f"[{idx}/{len(graphs)}] No boxes for {fname}")
                continue

            try:
                feats = extract_features(img_path, gt_boxes)
            except Exception as exc:
                print(f"[{idx}/{len(graphs)}] Failed {fname}: {exc}")
                continue

            entry = {
                "image": fname,
                "boxes": gt_boxes,
                "features": feats,
            }
            pickle.dump(entry, f)

            if idx == 1 or idx % 50 == 0 or idx == len(graphs):
                print(f"Processed {idx}/{len(graphs)} images")

    print("Done.")


if __name__ == "__main__":
    precompute_all()
