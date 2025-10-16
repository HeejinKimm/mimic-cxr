# config.py
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # ★ local_train 상위(mimic-cxr/mimic-cxr)

@dataclass
class Cfg:
    # paths
    CLIENT_CSV_DIR: Path = BASE / "client_splits"
    TEST_CSV_PATH:  Path = BASE / "client_splits" / "test.csv"
    METADATA_CSV:   Path = Path(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\mimic-cxr-2.0.0-metadata.csv")
    IMG_ROOT:       Path = Path(r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files")
    TXT_ROOT:       Path = Path(r"D:\mimic-cxr\physionet.org\files\mimic-cxr\2.1.0\files")
    LABEL_CSV_NEGBIO:   Path = Path(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\mimic-cxr-2.0.0-negbio.csv")
    LABEL_CSV_CHEXPERT: Path = Path(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\mimic-cxr-2.0.0-chexpert.csv")
    USE_LABEL: str = "negbio"

    # labels
    LABEL_COLUMNS = [
        "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
        "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
        "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
    ]

    # model/data
    TEXT_MODEL_NAME: str = "prajjwal1/bert-mini"
    MAX_LEN: int = 256

    # train
    BATCH_SIZE: int = 32
    EPOCHS: int = 5
    LR: float = 1e-4
    NUM_WORKERS: int = 2
