# data.py
import os, csv, torch, pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms

def img_transform(): # 이미지 데이터 정규화 
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


# 역할: 'p10001234', 's50414267' 같은 문자열에서 앞글자(p/s)를 떼고 정수로 변환.
# return: (subject_id_int, study_id_int)
def id_to_int(subject_id: str, study_id: str):
    return int(subject_id[1:]), int(study_id[1:])

#역할: 메타데이터 CSV(열: dicom_id, subject_id, study_id, ViewPosition)를 읽어 
# 각 (subject, study)에 대해 대표 이미지 1장 경로를 매핑(dict)으로 생성.
def build_image_picker_from_metadata(
    metadata_csv: str,
    img_root: str,
    prefer_views=("PA", "AP"), # 튜플의 앞에 있을수록 더 우선됨. 우선순위 : PA > AP
    use_sharded=True,          # 디렉토리 구조 형태 여부 
    check_exists=True,         # 파일 존재 검사 활성화 
    fallback_glob=True         
):
    
    """
    MIMIC-CXR metadata(csv)에 'path'가 없어도 (subject_id, study_id, dicom_id)로
    JPG 파일 절대경로를 구성해, 각 (subject_id, study_id)에 대해 '대표 이미지' 1장을 매핑
    대표 선택은 prefer_views 우선순위(PA/AP) → 그 외 정렬순

    Parameters
    ----------
    metadata_csv : str
        mimic-cxr-2.0.0-metadata.csv 경로 (열: dicom_id, subject_id, study_id, ViewPosition 등)
    img_root : str
        MIMIC-CXR-JPG 루트 디렉터리 (예: D:/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files)
    prefer_views : tuple
        우선시할 뷰 이름들(앞에 있는 게 더 높은 우선순위). 기본 ("PA","AP")
    use_sharded : bool
        True면 p{subject_id//10000:02d}/p{subject_id}/s{study_id}/{dicom_id}.jpg 구조를 사용.
        False면 p{subject_id}/s{study_id}/{dicom_id}.jpg 구조를 사용.
    check_exists : bool
        생성한 경로가 실제 존재하는지 검사. 존재하지 않으면 fallback_glob이 True일 경우 글로빙 시도.
    fallback_glob : bool
        경로가 없을 때 study 폴더 아래에서 {dicom_id}.jpg를 글로빙으로 찾아보는 폴백 사용.

    Returns
    -------
    dict[(int,int) -> str]
        {(subject_id, study_id) : abs_path_to_representative_jpg}
        파일을 못 찾으면 해당 pair는 누락될 수 있습니다.
    """
    if not metadata_csv or not os.path.exists(metadata_csv):
        return {}


    df = pd.read_csv(metadata_csv)
    required_cols = {"dicom_id", "subject_id", "study_id"} # 필수로 확인해야하는 열 
    missing = [c for c in required_cols if c not in df.columns] 
    if missing:
        raise ValueError(f"metadata CSV에 필요한 컬럼이 없습니다: {missing}")


    # 타입 정리 (정수형으로 통일)
    df["subject_id"] = df["subject_id"].astype(int)
    df["study_id"]   = df["study_id"].astype(int)


    # 뷰 컬럼 찾기
    view_col = "ViewPosition" if "ViewPosition" in df.columns else ("view" if "view" in df.columns else None)


    # 우선순위 점수 만들기 (작을수록 우선)
    if view_col:
        prio_map = {v: i for i, v in enumerate(prefer_views)}  # "PA":0, "AP":1, 그 외 2 = len(prefer_views)
        def _prio(v):
            return prio_map.get(str(v), len(prefer_views))
        df["_prio"] = df[view_col].fillna("").map(_prio)
    else:
        df["_prio"] = len(prefer_views)  # 전부 동일 우선순위


    # (subject_id, study_id, _prio)로 정렬 후, 각 그룹 첫 행만 추출 (대표 이미지)
    df = df.sort_values(["subject_id", "study_id", "_prio"])
    rep = df.drop_duplicates(subset=["subject_id", "study_id"], keep="first").copy()

    img_root = Path(img_root)
    picks = {} # 결과 담을 딕셔너리 

    for r in rep.itertuples(index=False):
        # 튜플 레코드 r에서 subject_id, study_id, dicom_id를 꺼내 정수/문자열로 캐스팅.
        subj = int(r.subject_id)
        study = int(r.study_id)
        dicom = str(r.dicom_id)

        # 폴더 구조 생성 (현재 가진 데이터로는 use_sharded로 작동될 것)
        if use_sharded:
            # 예: p10/p10001234/s50414267/{dicom_id}.jpg
            shard = f"p{subj // 10000:02d}"
            study_dir = img_root / shard / f"p{subj}" / f"s{study}"
        else:
            # 예: p10001234/s50414267/{dicom_id}.jpg
            study_dir = img_root / f"p{subj}" / f"s{study}"

        cand = study_dir / f"{dicom}.jpg"  # 후보 파일 경로 

        chosen = None
        if not check_exists:
            chosen = cand
        else:
            if cand.exists():
                chosen = cand
            elif fallback_glob and study_dir.exists():
                # 파일명이 소문자/대문자 혼재하거나 확장자가 다를 수 있으니 글로빙 시도
                matches = list(study_dir.glob(f"{dicom}.*"))
                if matches:
                    chosen = matches[0]

        if chosen is not None:
            picks[(subj, study)] = str(chosen)

    return picks # 대표 이미지 정해진 정보 리턴

# =========================
# 라벨 로딩/매핑
# 라벨 CSV를 읽어서 각 (subject_id, study_id)에 대응하는 멀티라벨 벡터(0/1 리스트)를 빠르게 조회할 수 있도록 딕셔너리로 만들어 반환
# =========================
def load_label_table(label_csv_path: str, label_cols):
    import pandas as pd, time, shutil
    from pathlib import Path

    # 라벨 csv 파일 (negbio or chexpert) 읽기 시도 
    last_err = None
    for attempt in range(3):
        try:
            df = pd.read_csv(label_csv_path)
            break
        except PermissionError as e:
            last_err = e
            print(f"[WARN] PermissionError on '{label_csv_path}' (attempt {attempt+1}/3). Retrying...")
            time.sleep(1.0)
        except Exception as e:
            last_err = e
            break
    else:
        # 임시 복사 → 읽기
        tmp_dir = Path("./_tmp"); tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / Path(label_csv_path).name
        print(f"[INFO] Trying temp copy -> {tmp_path}")
        shutil.copy2(label_csv_path, tmp_path)
        df = pd.read_csv(tmp_path)

    required = {"subject_id","study_id", *label_cols}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"라벨 CSV에 칼럼이 없습니다: {missing}")


    for c in label_cols:
        df[c] = df[c].fillna(0)  # 결측치는 0으로 처리 
        df[c] = (df[c] >= 1).astype(int)  # 1 이상이면 1, 그 외 0 (현재는 -1 그냥 0이라고 고려)


    # table 구조
    # {
    #   (subject_id:int, study_id:int): [label0, label1, ..., labelK]  # LABEL_COLUMNS 순서!
    # }
    table = {}
    for _, row in df.iterrows():  # row: pandas Series
        key = (int(row["subject_id"]), int(row["study_id"]))
        vec = [int(row[c]) for c in label_cols]  # 공백/특수문자 포함 컬럼도 OK
        table[key] = vec
    return table


# 역할: 클라이언트 분할 CSV 한 장을 읽어 학습 배치를 생성.
class ClientDataset(Dataset):
    def __init__(self, csv_path, label_table, mode, meta_picker, text_model_name, max_len, transform):
        # label_table: (subject,study) → 라벨 벡터 dict
        # mode: 'multimodal' | 'image_only' | 'text_only' | 'test_mix'
        self.mode = mode
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if mode == "test_mix":
                    if row["modality"] == "test_mix":
                        self.rows.append(row)
                elif row["modality"] == mode:
                    self.rows.append(row)

        self.label_table = label_table
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_len = max_len
        self.meta_picker = meta_picker
        self.tf = transform or img_transform()

    def __len__(self): return len(self.rows)

    def _load_image(self, image_dir, subject_id, study_id):
        if self.meta_picker and subject_id and study_id: # 대표 이미지 존재하면 그걸 먼저 사용.
            key = (int(subject_id[1:]), int(study_id[1:]))
            path = self.meta_picker.get(key)
            if path and os.path.exists(path):
                return self.tf(Image.open(path).convert("RGB"))
        p = Path(image_dir)
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")])
        if not imgs: return torch.zeros(3,224,224) # 없으면 제로 텐서(3×224×224) 반환(결손 대응).
        return self.tf(Image.open(imgs[0]).convert("RGB"))

    def _load_text(self, text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f: text = f.read()
        except: text = ""
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = id_to_int(r["subject_id"], r["study_id"])
        labels = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(self.label_table[next(iter(self.label_table))])),
                              dtype=torch.float)

        pixel_values = torch.zeros(3,224,224)
        input_ids = attention_mask = torch.zeros(1, dtype=torch.long).repeat(256)
        img_mask = torch.tensor([0.], dtype=torch.float32); txt_mask = torch.tensor([0.], dtype=torch.float32)

        if self.mode in ["multimodal","image_only","test_mix"]:
            pixel_values = self._load_image(r["image_dir"], r["subject_id"], r["study_id"])
            img_mask = torch.tensor([1.], dtype=torch.float32)
        if self.mode in ["multimodal","text_only","test_mix"]:
            t = self._load_text(r["text_path"])
            input_ids, attention_mask = t["input_ids"], t["attention_mask"]
            txt_mask = torch.tensor([1.], dtype=torch.float32)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_mask=img_mask, txt_mask=txt_mask,
            labels=labels,
            subject_id=r["subject_id"], study_id=r["study_id"]
        )
