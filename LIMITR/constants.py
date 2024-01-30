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


CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

# SIIM Pneumothorax
PNEUMOTHORAX_DATA_DIR = Path("/data4/siim")
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7

# RSNA Pneumonia
PNEUMONIA_DATA_DIR = Path("/home/gefen/Downloads/RSNA")  # aimi0/rsna_pneumonia')
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR / "stage_2_train_labels.csv"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_DATA_DIR / "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_DATA_DIR / "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_DATA_DIR / "test.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR / "stage_2_train_images"
PNEUMONIA_TRAIN_PCT = 0.7

MS_CXR_PATHOLOGY = {
    "Atelectasis": [
        "atelectasis",
    ],
    "Pneumonia": [
        'pneumonia',
        'opacity',
        'opacities',
        'consolidation'
    ],
    "Pneumothorax": [
        'pneumothorax',
        'hydropneumothorax'
    ],
    "Consolidation": [
        'consolidation'
        'patchy'
    ],
    "Edema": [
        'edema',
        'opacities'
    ],
    "Cardiomegaly": [
        'enlarged',
        'cardiomegaly'
    ],
    'Lung Opacity': [
        'opacities',
        'patchy',
        'opacity',
        'ground-glass'
    ],
    'Pleural Effusion': [
        'effusions',
        'effusion'
    ]
}
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}
