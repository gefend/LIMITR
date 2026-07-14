#!/bin/bash

# ─── Required arguments ───────────────────────────────────────────────────────
# --ckpt_path        : path to model checkpoint
# --features_base_dir: base directory where feature files will be saved
# --feature_folder   : sub-folder name inside features_base_dir
# --evaldata_csv_path: path to the evaldata CSV file
# ──────────────────────────────────────────────────────────────────────────────

CKPT_PATH=""
FEATURES_BASE_DIR=""
FEATURE_FOLDER=""
EVALDATA_CSV_PATH=""
CONFIG="./configs/mimic_config.yaml"

usage() {
    echo "Usage: $0 --ckpt_path <path> --features_base_dir <path> --feature_folder <name> --evaldata_csv_path <path> [--config <path>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt_path)          CKPT_PATH="$2";          shift 2 ;;
        --features_base_dir)  FEATURES_BASE_DIR="$2";  shift 2 ;;
        --feature_folder)     FEATURE_FOLDER="$2";     shift 2 ;;
        --evaldata_csv_path)  EVALDATA_CSV_PATH="$2";  shift 2 ;;
        --config)             CONFIG="$2";             shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$CKPT_PATH" || -z "$FEATURES_BASE_DIR" || -z "$FEATURE_FOLDER" || -z "$EVALDATA_CSV_PATH" ]]; then
    echo "Error: missing required arguments."
    usage
fi

mkdir -p "$FEATURES_BASE_DIR/$FEATURE_FOLDER"

echo "============================================================"
echo " Extracting features for: gt"
echo "============================================================"
python run.py \
    -c "$CONFIG" \
    --test \
    --ckpt_path "$CKPT_PATH" \
    --save_features \
    --features_base_dir "$FEATURES_BASE_DIR" \
    --feature_folder "$FEATURE_FOLDER" \
    --evaldata_csv_path "$EVALDATA_CSV_PATH" \
    --evaldata_report gt \
    --data_type evaldata

if [[ $? -ne 0 ]]; then
    echo "ERROR: feature extraction failed for 'gt'. Aborting."
    exit 1
fi

echo ""
echo "============================================================"
echo " Extracting features for: output"
echo "============================================================"
python run.py \
    -c "$CONFIG" \
    --test \
    --ckpt_path "$CKPT_PATH" \
    --save_features \
    --features_base_dir "$FEATURES_BASE_DIR" \
    --feature_folder "$FEATURE_FOLDER" \
    --evaldata_csv_path "$EVALDATA_CSV_PATH" \
    --evaldata_report output \
    --data_type evaldata

if [[ $? -ne 0 ]]; then
    echo "ERROR: feature extraction failed for 'output'. Aborting."
    exit 1
fi

echo ""
echo "============================================================"
echo " Merging feature files"
echo "============================================================"
python merge_sims_features.py \
    --features_base_dir "$FEATURES_BASE_DIR" \
    --feature_folder "$FEATURE_FOLDER" \
    --merge_features

if [[ $? -ne 0 ]]; then
    echo "ERROR: merging features failed. Aborting."
    exit 1
fi

echo ""
echo "Done. Merged features saved to: $FEATURES_BASE_DIR/$FEATURE_FOLDER"
