from pathlib import Path
# #############################################
# MIMIC-CXR-JPG constants
# #############################################
DATA_BASE_DIR = Path("/home/gefen/Documents")
SPLITS_DIR = Path('mimic_csv')
# Created csv
MIMIC_CXR_TRAIN_CSV = SPLITS_DIR / "train_split.csv"
MIMIC_CXR_VALID_CSV = SPLITS_DIR / "valid_split.csv"
MIMIC_CXR_TEST_CSV = SPLITS_DIR / "test_split.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL_F = "Path_frontal"
MIMIC_CXR_PATH_COL_L = "Path_lateral"
MIMIC_CXR_SPLIT_COL = "split"
MIMIC_CXR_REPORT_COL = "Report"
MIMIC_CXR_PATH_COL = "Path"



