#!/bin/bash

# 1. Set your Kaggle API Token (bypasses the need for kaggle.json)
export KAGGLE_API_TOKEN=KGAT_c704c4e24751ebc6242116f9518973a2

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
