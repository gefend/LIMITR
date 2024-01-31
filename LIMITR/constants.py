from pathlib import Path
# #############################################
# MIMIC-CXR-JPG constants
# #############################################
DATA_BASE_DIR = Path("/home/gefen/Documents")
MIMIC_CXR_DATA_DIR = Path('/home/gefen/Documents/physionet.org/files/mimic-cxr-jpg/2.0.0')
# Created csv
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR / "splits/train_split.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR / "splits/valid_split.csv"
MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR / "splits/test_split.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL_F = "Path_frontal"
MIMIC_CXR_PATH_COL_L = "Path_lateral"
MIMIC_CXR_SPLIT_COL = "split"
MIMIC_CXR_REPORT_COL = "Report"
MIMIC_CXR_PATH_COL = "Path"



