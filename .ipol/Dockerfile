# .ipol/Dockerfile
# ============================================================
# Scene Graph Image Retrieval — IPOL Docker Image
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    wget unzip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. Python dependencies
#    Adjust versions if your code requires something specific.
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    Pillow \
    matplotlib \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------
# 3. Download large assets from GitHub Release v1.0.0
#
#    Replace GITHUB_RELEASE_URL with the actual base URL of
#    your release, e.g.:
#    https://github.com/titiuo/RR-Image-Retrieval-using-Scene-Graphs/releases/download/v1.0.0
#
#    Directory layout expected by test_pipeline.py (paths are
#    relative to the repo root, which IPOL clones to $bin):
#
#      $bin/data/                  ← from data.zip
#      $bin/model/                 ← from model.zip
#      $bin/sg_dataset/
#          sg_test_images/         ← from sg_test_images.zip
#          sg_test_annotations.json  (inside data.zip or repo)
#
#    trained_models/ is already committed to the repo → no download needed.
# ------------------------------------------------------------

ENV GITHUB_RELEASE_URL=https://github.com/titiuo/RR-Image-Retrieval-using-Scene-Graphs/releases/download/v1.0.0
ENV ASSETS_DIR=/assets

RUN mkdir -p ${ASSETS_DIR}

# --- data.zip → $ASSETS_DIR/data/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/data.zip" \
    -O /tmp/data.zip \
    && unzip -q /tmp/data.zip -d ${ASSETS_DIR} \
    && rm /tmp/data.zip \
    && echo "data.zip extracted OK"

# --- model.zip → $ASSETS_DIR/model/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/model.zip" \
    -O /tmp/model.zip \
    && unzip -q /tmp/model.zip -d ${ASSETS_DIR} \
    && rm /tmp/model.zip \
    && echo "model.zip extracted OK"

# --- sg_test_images.zip → $ASSETS_DIR/sg_dataset/sg_test_images/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/sg_test_images.zip" \
    -O /tmp/sg_test_images.zip \
    && mkdir -p ${ASSETS_DIR}/sg_dataset \
    && unzip -q /tmp/sg_test_images.zip -d ${ASSETS_DIR}/sg_dataset \
    && rm /tmp/sg_test_images.zip \
    && echo "sg_test_images.zip extracted OK"

# Verify key files exist
RUN echo "=== Asset verification ===" \
    && ls ${ASSETS_DIR}/data/ \
    && ls ${ASSETS_DIR}/model/ \
    && ls ${ASSETS_DIR}/sg_dataset/

# ------------------------------------------------------------
# 4. Smoke-test Python imports
# ------------------------------------------------------------
RUN python3 -c "import numpy, PIL, sklearn, torch; print('Python imports OK')"