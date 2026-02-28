# .ipol/Dockerfile
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
#    Install PyPI packages first, then torch separately
#    with its own index (mixing --index-url with other packages
#    breaks pip because it replaces PyPI entirely).
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    Pillow \
    matplotlib

RUN pip3 install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------
# 3. Download large assets from GitHub Release v1.0.0
# ------------------------------------------------------------
ENV GITHUB_RELEASE_URL=https://github.com/titiuo/RR-Image-Retrieval-using-Scene-Graphs/releases/download/v1.0.0
ENV ASSETS_DIR=/assets

RUN mkdir -p ${ASSETS_DIR}

# --- data.zip → /assets/data/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/data.zip" \
    -O /tmp/data.zip \
    && unzip -q /tmp/data.zip -d ${ASSETS_DIR} \
    && rm /tmp/data.zip \
    && echo "data.zip extracted OK" \
    && ls ${ASSETS_DIR}/data/

# --- model.zip → /assets/model/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/model.zip" \
    -O /tmp/model.zip \
    && unzip -q /tmp/model.zip -d ${ASSETS_DIR} \
    && rm /tmp/model.zip \
    && echo "model.zip extracted OK" \
    && ls ${ASSETS_DIR}/model/

# --- sg_test_images.zip → /assets/sg_dataset/sg_test_images/ ---
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/sg_test_images.zip" \
    -O /tmp/sg_test_images.zip \
    && mkdir -p ${ASSETS_DIR}/sg_dataset \
    && unzip -q /tmp/sg_test_images.zip -d ${ASSETS_DIR}/sg_dataset \
    && rm /tmp/sg_test_images.zip \
    && echo "sg_test_images.zip extracted OK" \
    && ls ${ASSETS_DIR}/sg_dataset/

# ------------------------------------------------------------
# 4. Smoke-test
# ------------------------------------------------------------
RUN python3 -c "import numpy, PIL, sklearn, torch, scipy, pandas; print('All imports OK')"