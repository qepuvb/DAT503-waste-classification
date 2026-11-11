#!/bin/bash
# TrashNet, resized. Commit 6fa2b878 (2023-06-02).
cd "$(dirname "$0")/.."
mkdir -p data/raw

if [ ! -f data/dataset-resized.zip ]; then
  curl -L -o data/dataset-resized.zip \
    https://github.com/garythung/trashnet/raw/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip
fi

echo "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2  data/dataset-resized.zip" | sha256sum -c

unzip -oq data/dataset-resized.zip -d data/raw
rm -rf data/raw/__MACOSX
find data/raw -name ".DS_Store" -delete
