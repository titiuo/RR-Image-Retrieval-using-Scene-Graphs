# .ipol/Dockerfile
# Use the official IPOL base image — this sets up $bin, $HOME,
# the ipol user, and all IPOL-specific conventions automatically.
FROM registry.ipol.im/ipol:v2-py3.11

# ------------------------------------------------------------
# 1. Install Python dependencies from requirements.txt
# ------------------------------------------------------------
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

# ------------------------------------------------------------
# 2. Copy repo code into $bin
# ------------------------------------------------------------
ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

# ------------------------------------------------------------
# 3. Download large assets from GitHub Release v1.0.0
#    into /assets/ so test_pipeline.py can find them.
#    (trained_models/ is already in the repo → copied above)
# ------------------------------------------------------------
ENV GITHUB_RELEASE_URL=https://github.com/titiuo/RR-Image-Retrieval-using-Scene-Graphs/releases/download/v1.0.0
ENV ASSETS_DIR=/assets

RUN mkdir -p ${ASSETS_DIR}

# data.zip → /assets/data/
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/data.zip" \
    -O /tmp/data.zip \
    && unzip -q /tmp/data.zip -d ${ASSETS_DIR} \
    && rm /tmp/data.zip \
    && echo "data.zip OK" && ls ${ASSETS_DIR}/data/

# model.zip → /assets/model/
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/model.zip" \
    -O /tmp/model.zip \
    && unzip -q /tmp/model.zip -d ${ASSETS_DIR} \
    && rm /tmp/model.zip \
    && echo "model.zip OK" && ls ${ASSETS_DIR}/model/

# sg_test_images.zip → /assets/sg_dataset/sg_test_images/
RUN wget -q --show-progress \
    "${GITHUB_RELEASE_URL}/sg_test_images.zip" \
    -O /tmp/sg_test_images.zip \
    && mkdir -p ${ASSETS_DIR}/sg_dataset \
    && unzip -q /tmp/sg_test_images.zip -d ${ASSETS_DIR}/sg_dataset \
    && rm /tmp/sg_test_images.zip \
    && echo "sg_test_images.zip OK" && ls ${ASSETS_DIR}/sg_dataset/

# ------------------------------------------------------------
# 4. IPOL conventions (mirrors the original Dockerfile)
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH
ENV HOME /home/ipol

RUN groupadd -g 1001 ipol \
    && useradd -m -u 1001 -g 1001 ipol -d $HOME \
    && chmod -R 777 $HOME \
    && chmod -R 777 /assets

USER ipol