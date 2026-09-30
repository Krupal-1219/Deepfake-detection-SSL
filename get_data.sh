#!/bin/bash

# 1. Kaggle API token: set it in your shell first, never commit it.
#    export KAGGLE_API_TOKEN=...
: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN before running this script}"

# 2. Install required tools quietly
echo "Installing libraries..."
pip install kaggle opencv-python pandas -q

# 3. Create the folder
DATA_DIR="./dfdc_sample_data"
mkdir -p $DATA_DIR

# 4. Download the community clone of the DFDC dataset
echo "Downloading 4GB DFDC Dataset..."
kaggle datasets download -d ashifurrahman34/dfdc-dataset -p $DATA_DIR

# 5. Unzip and clean up
echo "Unzipping the dataset (This takes a minute)..."
unzip -q $DATA_DIR/dfdc-dataset.zip -d $DATA_DIR
rm $DATA_DIR/dfdc-dataset.zip

echo "Success! Dataset organized in: $DATA_DIR"
