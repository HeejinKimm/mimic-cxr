# zero_shot_h_plus_alphaZ.py
# --------------------------
# Zero-shot inference by injecting global Z: h <- h + alpha * Z
# + Temperature scaling (logits/T), grid-search over (alpha, T), report AUROC.
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from PIL import Image

# ====== 프로젝트 의존 모듈 ======
from local_train.config import Cfg
from local_train.models import MultiModalLateFusion
from local_train.data import build_image_picker_from_metadata, img_transform

# ---------- 전역 ----------
NUM_CLASSES = 13
LABEL_COLUMNS = getattr(Cfg, "LABEL_COLUMNS", None)
if LABEL_COLUMNS is None:
    LABEL_COLUMNS = [
        "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
        "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
        "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
    ]

# ---------- 유틸 ----------
def pad2(n: int) -> str:
    return f"{n:02d}"

def parse_clients(spec: str) -> List[int]:
    parts = re.split(r"[,\s]+", spec.strip())
    out = []
    for p in parts:
        if not p: continue
        if "-" in p:
            a, b = p.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def normalize_label_columns(df):
    """Pleural_Effusion → Pleural Effusion 등 언더스코어→공백 정규화"""
    rename_map = {}
    for c in df.columns:
        if "_" in c:
            renamed = c.replace("_", " ")
            if renamed in LABEL_COLUMNS:
                rename_map[c] = renamed
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _try_cast_int_series(s):
    import pandas as pd
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except Exception:
        return s

def filter_by_client_robust(df, cid: int):
    """
    다양한 표기 형태를 지원하여 client별 행을 필터링.
    - 정수형: client_id==1, client==1, cid==1
    - 문자열형: tag=='client_01', client=='client_01' 등
    - 문자열 숫자 추출: '01' → 1, 'client_1' → 1
    실패 시 경고 후 필터링 없이 반환.
    """
    import pandas as pd
    original_len = len(df)
    cand_cols = ["client_id", "client", "cid", "tag"]
    cols_present = [c for c in cand_cols if c in df.columns]
    if not cols_present:
        print("[WARN] No client column found in test_csv; proceeding without client filtering.")
        return df

    df2 = df.copy()
    cid_str = f"client_{cid:02d}"
    candidates = []

    for c in cols_present:
        s = df2[c]

        # 1) 정확 문자열 매칭 (client_01)
        if s.dtype == object:
            hit = df2[s.astype(str).str.strip().str.lower() == cid_str]
            if len(hit) > 0:
                candidates.append(hit)

        # 2) 문자열에서 숫자만 추출해서 비교
        s_str = s.astype(str).str.extract(r"(\d+)", expand=False)
        with np.errstate(all="ignore"):
            s_num = pd.to_numeric(s_str, errors="coerce")
        hit = df2[s_num == cid]
        if len(hit) > 0:
            candidates.append(hit)

        # 3) 정수형 강제 캐스팅 후 비교
        s_int = _try_cast_int_series(s)
        if str(s_int.dtype).startswith("Int"):
            hit = df2[s_int == cid]
            if len(hit) > 0:
                candidates.append(hit)

    if candidates:
        best = max(candidates, key=lambda x: len(x))
        print(f"[INFO] Client filter used -> column among {cols_present}, kept {len(best)}/{original_len} rows.")
        return best.reset_index(drop=True)
    else:
        print(f"[WARN] Client filter produced 0 rows for cid={cid}. Proceeding WITHOUT filtering.")
        return df.reset_index(drop=True)

def _digits_as_int_series(s):
    """'p12345' / 's0000123' / ' 12345 ' / 12345 -> 숫자만 추출해 Int64로."""
    import pandas as pd
    s = s.astype(str).str.strip()
    digits = s.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(digits, errors="coerce").astype("Int64")
    return out

def unify_join_key_types(left_df, right_df, keys):
    """
    left_df/right_df의 join 키 컬럼들을 모두 Int64로 통일.
    접두사(p/s 등)는 제거되어 숫자만 남도록.
    """
    left_df = left_df.copy()
    right_df = right_df.copy()
    for k in keys:
        if k not in left_df.columns or k not in right_df.columns:
            raise ValueError(f"join key '{k}' not found in both dataframes")
        left_df[k] = _digits_as_int_series(left_df[k])
        right_df[k] = _digits_as_int_series(right_df[k])
    before_l = len(left_df); before_r = len(right_df)
    left_df = left_df.dropna(subset=keys)
    right_df = right_df.dropna(subset=keys)
    if len(left_df) < before_l or len(right_df) < before_r:
        print(f"[WARN] Dropped rows with missing join keys "
              f"(left: {before_l}->{len(left_df)}, right: {before_r}->{len(right_df)})")
    print("[DEBUG] dtypes after unify:",
          "left:", {k: str(left_df[k].dtype) for k in keys},
          "right:", {k: str(right_df[k].dtype) for k in keys})
    return left_df, right_df

def read_label_csv_direct(label_csv: str):
    import pandas as pd
    df = pd.read_csv(label_csv)
    df = normalize_label_columns(df)
    for c in LABEL_COLUMNS:
        if c in df.columns:
            df[c] = df[c].replace(-1, 0).fillna(0).astype(float)
    return df

# ---- PICKLE-SAFE PICKER ----
class DictPicker:
    """dict 기반 메타 인덱스를 call 형태로 감싸는 전역 클래스(피클 가능).
       - dicom_id가 비어있거나 후보가 여러 개일 때도 첫 번째를 안정적으로 선택
       - 메타데이터 CSV를 폴백 소스로 사용(해당 (subject, study)의 첫 JPG)
       - 메타 CSV가 없어도 MIMIC-CXR-JPG 폴더 구조에서 직접 탐색
    """
    def __init__(self, index_dict, img_root: str, metadata_csv: Optional[str] = None):
        self.index = index_dict
        self.img_root = img_root
        self.meta_csv = metadata_csv
        self._meta_df = None
        self._path_candidates = []

    @staticmethod
    def _digits_or_none(x):
        if x is None: return None
        s = str(x).strip()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def _join_path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.img_root, p)

    @staticmethod
    def _first_value(d: dict):
        """dict에서 키를 문자열 기준 정렬 후 첫 값을 리턴. 중첩이면 한 단계 더."""
        if not isinstance(d, dict) or len(d) == 0:
            return None
        first_key = sorted(d.keys(), key=lambda k: str(k))[0]
        v = d[first_key]
        if isinstance(v, dict):
            if len(v) == 0:
                return None
            inner_key = sorted(v.keys(), key=lambda k: str(k))[0]
            v = v[inner_key]
        return v

    def _lookup_tuple(self, subj_i, study_i, dicom_i):
        d = self.index
        key = (subj_i, study_i, dicom_i)
        if key in d: return d[key]
        key2 = (subj_i, study_i)
        if key2 in d: return d[key2]
        key3 = (str(subj_i), str(study_i), str(dicom_i))
        if key3 in d: return d[key3]
        key4 = (str(subj_i), str(study_i))
        if key4 in d: return d[key4]
        return None

    def _lookup_nested(self, subj_i, study_i, dicom_i):
        """중첩 구조 지원 + 후보가 여러 개면 첫 번째 반환"""
        d = self.index
        x = d.get(subj_i, d.get(str(subj_i), None))
        if isinstance(x, dict):
            y = x.get(study_i, x.get(str(study_i), None))
            if isinstance(y, dict):
                # dicom 정확 매칭 시도
                if dicom_i is not None:
                    if dicom_i in y: return y[dicom_i]
                    if str(dicom_i) in y: return y[str(dicom_i)]
                # 후보가 여러 개인 경우: 첫 번째 선택
                v = self._first_value(y)
                if v is not None:
                    return v
            elif isinstance(y, (str, os.PathLike)):
                return y
        return None

    # ---- 메타 CSV 폴백 ----
    def _load_meta(self):
        if self.meta_csv is None or self._meta_df is not None:
            return
        import pandas as pd
        try:
            df = pd.read_csv(self.meta_csv)
        except FileNotFoundError:
            print(f"[WARN] metadata_csv not found: {self.meta_csv} (will fallback to folder structure)")
            self.meta_csv = None
            self._meta_df = None
            self._path_candidates = []
            return

        def _dig_series(series):
            return pd.to_numeric(
                pd.Series(series).astype(str).str.extract(r'(\d+)', expand=False),
                errors="coerce"
            ).astype("Int64")

        for k in ("subject_id", "study_id", "dicom_id"):
            if k in df.columns:
                df[f"{k}__norm"] = _dig_series(df[k])

        self._path_candidates = [c for c in df.columns if any(x in c.lower() for x in ["jpg", "path", "file"])]
        self._meta_df = df

    def _fallback_from_metadata(self, sid: Optional[int], stid: Optional[int]):
        """(sid, stid)로 메타CSV를 검색하여 첫 번째 jpg 경로를 반환."""
        self._load_meta()
        if self._meta_df is None:
            return None
        df = self._meta_df
        if sid is None or stid is None:
            return None

        q = df
        if "subject_id__norm" in q.columns:
            q = q[q["subject_id__norm"] == sid]
        else:
            return None
        if "study_id__norm" in q.columns:
            q = q[q["study_id__norm"] == stid]
        else:
            return None
        if len(q) == 0:
            return None

        # 1) 경로 후보 컬럼에서 .jpg 포함된 값 우선 사용
        for col in self._path_candidates:
            vals = q[col].dropna().astype(str)
            jpgs = vals[vals.str.contains(r"\.jpg$", case=False, regex=True)]
            if len(jpgs) > 0:
                p = jpgs.iloc[0]
                return p if os.path.isabs(p) else os.path.join(self.img_root, p)

        # 2) 값들 중 .jpg로 끝나는 문자열이 존재하면 사용
        for _, row in q.iterrows():
            for v in row.values:
                if isinstance(v, str) and re.search(r"\.jpg$", v, flags=re.I):
                    p = v
                    return p if os.path.isabs(p) else os.path.join(self.img_root, p)

        return None

    # ---- 폴더 구조 폴백 (메타 CSV 없이도 동작) ----
    def _guess_base_files_dir(self) -> Optional[str]:
        # 후보 1: <img_root>/2.1.0/files
        cand1 = os.path.join(self.img_root, "2.1.0", "files")
        # 후보 2: <img_root>/mimic-cxr-jpg/2.1.0/files  (사용자 폴더 예시 대응)
        cand2 = os.path.join(self.img_root, "mimic-cxr-jpg", "2.1.0", "files")
        if os.path.isdir(cand1): return cand1
        if os.path.isdir(cand2): return cand2
        # 그래도 없으면 img_root 자체로 시도 (사용자가 이미 files까지 줬을 수도)
        return self.img_root if os.path.isdir(self.img_root) else None

    def _fallback_from_folder(self, sid: Optional[int], stid: Optional[int]) -> Optional[str]:
        if sid is None or stid is None:
            return None
        base = self._guess_base_files_dir()
        if base is None:
            return None
        # p{앞2자리}/p{sid}/s{stid}
        sid_str = str(sid)
        group = f"p{sid_str[:2]}" if len(sid_str) >= 2 else f"p{sid_str}"
        subj_dir = f"p{sid}"
        study_dir = f"s{stid}"
        d = os.path.join(base, group, subj_dir, study_dir)
        if not os.path.isdir(d):
            return None
        # 스터디 폴더 안 첫 번째 JPG 선택
        files = [f for f in os.listdir(d) if re.search(r"\.jpg$", f, flags=re.I)]
        if not files:
            return None
        files.sort()
        return os.path.join(d, files[0])

    def __call__(self, subj: str, study: str, dicom: str):
        sid = self._digits_or_none(subj)
        stid = self._digits_or_none(study)
        did = self._digits_or_none(dicom)

        # 1) 튜플 인덱스 기반 탐색
        val = self._lookup_tuple(sid, stid, did)
        if val is None and did is not None:
            val = self._lookup_tuple(sid, stid, None)
        # 2) 중첩 dict 탐색 (여러 후보면 첫 번째)
        if val is None:
            val = self._lookup_nested(sid, stid, did)
        # 3) 메타 CSV 폴백
        if val is None:
            val = self._fallback_from_metadata(sid, stid)
        # 4) 폴더 구조 폴백
        if val is None:
            val = self._fallback_from_folder(sid, stid)

        if val is None:
            raise FileNotFoundError(
                f"[picker] path not found for keys subj={subj}, study={study}, dicom={dicom}"
            )

        # dict로 남아있다면 키 정렬 기준 첫 값을 뽑아 사용
        if isinstance(val, dict):
            for k in ("path", "img_path", "jpg_path", "file"):
                if k in val:
                    val = val[k]; break
            if isinstance(val, dict):
                chosen = self._first_value(val)
                if chosen is not None:
                    val = chosen

        if not isinstance(val, (str, os.PathLike)):
            raise TypeError(f"[picker] unexpected value type: {type(val)}")

        path = self._join_path(str(val))
        return path

def create_picker_in_worker(metadata_csv: str, img_root: str):
    """
    워커 프로세스 안에서 호출되어, 항상 피클 안전한 콜러블을 반환한다.
    - build_image_picker_from_metadata가 callable이면 그대로 반환
    - dict면 DictPicker로 감싼다(메타CSV/폴더 폴백 포함)
    - 그 외에는 폴더 폴백 전용 픽커 생성
    """
    obj = build_image_picker_from_metadata(metadata_csv=metadata_csv, img_root=img_root)
    if callable(obj):
        return obj
    if isinstance(obj, dict):
        return DictPicker(obj, img_root=img_root, metadata_csv=metadata_csv)
    return DictPicker({}, img_root=img_root, metadata_csv=metadata_csv)

def load_Z_for_client(z_root: str, cid: int, device: torch.device) -> torch.Tensor:
    d1 = os.path.join(z_root, f"client_{pad2(cid)}", "Z.npy")
    d2 = os.path.join(z_root, f"client_{cid}", "Z.npy")
    z_path = d1 if os.path.isfile(d1) else d2
    if not os.path.isfile(z_path):
        raise FileNotFoundError(f"[Z] Not found for client {cid}: {d1} or {d2}")
    z = np.load(z_path)  # (256,) or (1,256)
    z = torch.tensor(z, dtype=torch.float32, device=device)
    if z.ndim == 1:
        z = z[None, :]
    return z  # (1, D)

def find_final_classifier_linear(model: nn.Module) -> nn.Linear:
    target = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == NUM_CLASSES:
            target = m
    if target is None:
        raise RuntimeError("Final classifier Linear(out_features=NUM_CLASSES) not found.")
    return target

def register_h_plus_alphaZ_hook(model: nn.Module, Z: torch.Tensor, alpha: float):
    clf = find_final_classifier_linear(model)
    def pre_hook(mod, inputs):
        (h,) = inputs
        return (h + alpha * Z, )
    handle = clf.register_forward_pre_hook(pre_hook, with_kwargs=False)
    return handle

def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits if (T is None or T == 1.0) else (logits / T)

# ---------- Forward helper: be robust to different signatures ----------
def _safe_forward_logits(model: nn.Module, imgs: torch.Tensor, device: torch.device, txt_len: int = 8):
    """
    다양한 forward 시그니처에 대응:
      1) 이미지 전용 헬퍼 메서드가 있으면 사용 (encode_image/forward_image 등)
      2) forward에 여러 인자 패턴을 시도 (더미 text/mask 포함)
    반환: logits (B, NUM_CLASSES)
    """
    B = imgs.size(0)

    # (1) 전용 이미지 인코딩 경로가 있으면 이용
    cand_img_funcs = ["encode_image", "forward_image", "image_forward", "img_forward"]
    for fn in cand_img_funcs:
        if hasattr(model, fn):
            h = getattr(model, fn)(imgs)
            clf = find_final_classifier_linear(model)
            if hasattr(model, "proj") and isinstance(model.proj, nn.Linear) and model.proj.out_features == clf.in_features:
                h = model.proj(h)
            return clf(h)

    # (2) forward 시그니처 추측해가며 시도
    input_ids      = torch.zeros((B, txt_len), dtype=torch.long,  device=device)
    attention_mask = torch.zeros((B, txt_len), dtype=torch.long,  device=device)
    img_mask       = torch.ones((B,),         dtype=torch.bool,   device=device)
    txt_mask       = torch.zeros((B,),        dtype=torch.bool,   device=device)

    trials = [
        lambda: model(imgs, input_ids, attention_mask, img_mask, txt_mask),
        lambda: model(input_ids, attention_mask, imgs, img_mask, txt_mask),
        lambda: model(imgs, input_ids, attention_mask),
        lambda: model(imgs),
        lambda: model(x=imgs, input_ids=input_ids, attention_mask=attention_mask, img_mask=img_mask, txt_mask=txt_mask),
        lambda: model(vision=imgs, input_ids=input_ids, attention_mask=attention_mask, img_mask=img_mask, txt_mask=txt_mask),
    ]
    for t in trials:
        try:
            out = t()
            return out
        except (TypeError, RuntimeError):
            continue

    raise TypeError(
        "MultiModalLateFusion.forward 시그니처를 자동으로 파악하지 못했습니다. "
        "필요시 모델의 forward 정의를 확인해 (image, input_ids, attention_mask, img_mask, txt_mask) 형태로 맞춰주세요."
    )

# ---------- Dataset / Loader ----------
class ClientTestDataset(Dataset):
    """
    test.csv(샘플 목록) + label_csv(라벨 테이블)를 key로 merge하여 라벨 확보.
    - 조인 키 우선순위: (subject_id, study_id, dicom_id) -> (subject_id, study_id)
    - 이미지 경로는 metadata+picker로 해석 (dict/callable 모두 지원, 워커 내 지연 초기화)
    """
    def __init__(self, test_csv: str, label_csv: str, client_id: int,
                 img_root: str, metadata_csv: str, transform=None):
        import pandas as pd

        self.test_df = pd.read_csv(test_csv)

        before_n = len(self.test_df)
        self.test_df = filter_by_client_robust(self.test_df, client_id)
        after_n = len(self.test_df)
        print(f"[DEBUG] test rows before/after client filter: {before_n} -> {after_n}")

        label_df = read_label_csv_direct(label_csv)

        join_candidates = [
            ["subject_id", "study_id", "dicom_id"],
            ["subject_id", "study_id"]
        ]
        join_keys = None
        for keys in join_candidates:
            if all(k in self.test_df.columns for k in keys) and all(k in label_df.columns for k in keys):
                join_keys = keys
                break
        if join_keys is None:
            raise ValueError("조인 키를 찾지 못했습니다. test_csv와 label_csv에 공통 키가 필요합니다 "
                             "(가능 후보: (subject_id,study_id,dicom_id) 또는 (subject_id,study_id)).")

        print(
            f"[DEBUG] join key candidates present? "
            f"(subj,study,dicom)={all(k in self.test_df.columns for k in ['subject_id','study_id','dicom_id'])} & "
            f"(label has same)={all(k in label_df.columns for k in ['subject_id','study_id','dicom_id'])}"
        )
        print(
            f"[DEBUG] join fallback (subj,study)={all(k in self.test_df.columns for k in ['subject_id','study_id'])} & "
            f"(label has same)={all(k in label_df.columns for k in ['subject_id','study_id'])}"
        )
        print(f"[INFO] Using join keys: {join_keys}")

        self.test_df, label_df = unify_join_key_types(self.test_df, label_df, join_keys)
        merged = self.test_df.merge(label_df, on=join_keys, how="left", suffixes=("", "_lbl"))

        missing = [c for c in LABEL_COLUMNS if c not in merged.columns]
        if missing:
            raise ValueError(f"라벨 테이블에서 필요한 라벨 컬럼을 찾지 못했습니다: {missing}")

        if merged[LABEL_COLUMNS].isna().all(axis=1).any():
            print("[WARN] 일부 샘플에서 라벨이 비어있습니다 (NaN). 해당 항목은 평가에서 NaN으로 처리될 수 있습니다.")

        self.df = merged.reset_index(drop=True)
        print(f"[INFO] Final test rows for client {client_id}: {len(self.df)}")
        if len(self.df) == 0:
            raise ValueError(
                "No test rows found for this client after filtering/merge.\n"
                "- test.csv의 client 표기를 확인 (client_id/client/tag/cid)\n"
                "- label/test의 조인 키(subject_id, study_id, (optional) dicom_id) 일치 여부 확인"
            )

        self.transform = transform or img_transform()
        self.img_root = img_root
        self.meta_csv = metadata_csv
        self._picker = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self._picker is None:
            self._picker = create_picker_in_worker(self.meta_csv, self.img_root)

        row = self.df.iloc[i]
        subj = str(row.get("subject_id", ""))
        study = str(row.get("study_id", ""))
        dicom = str(row.get("dicom_id", ""))

        img_path = self._picker(subj, study, dicom)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        labels = torch.tensor([float(row[c]) for c in LABEL_COLUMNS], dtype=torch.float32)
        return img, labels

def build_loader(test_csv: str, label_csv: str, client_id: int, batch: int, num_workers: int,
                 device: torch.device, img_root: str, metadata_csv: str) -> DataLoader:
    ds = ClientTestDataset(
        test_csv=test_csv,
        label_csv=label_csv,
        client_id=client_id,
        img_root=img_root,
        metadata_csv=metadata_csv,
        transform=img_transform()
    )
    print(f"[INFO] Dataset size (client {client_id:02d}): {len(ds)}")
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                    pin_memory=(device.type == "cuda"))
    return dl

# ---------- 모델 로딩 (호환 키만 로드) ----------
def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> nn.Module:
    model = MultiModalLateFusion(num_classes=NUM_CLASSES).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model_sd = model.state_dict()
    compatible, skipped = {}, []
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    model_sd.update(compatible)
    model.load_state_dict(model_sd, strict=False)
    if skipped:
        print(f"[INFO] skipped {len(skipped)} keys due to shape mismatch (e.g., text encoder).")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

# ---------- 평가 ----------
@torch.no_grad()
def evaluate_one_client(model: nn.Module, loader: DataLoader, T: float, device: torch.device) -> Tuple[float, Dict[str, float]]:
    all_probs, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = _safe_forward_logits(model, imgs, device=device, txt_len=8)
        logits = apply_temperature(logits, T)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if not all_probs:
        raise ValueError("Empty DataLoader: no batches yielded. Check filtering/merge.")

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    per_class = {}
    for i, name in enumerate(LABEL_COLUMNS):
        try:
            auc = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        per_class[name] = auc
    valid_aucs = [v for v in per_class.values() if (v == v)]
    macro = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro, per_class

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--label_csv", type=str, required=True, help="CheXpert/NegBio 라벨 테이블 CSV")
    ap.add_argument("--local_dir", type=str, required=True, help=".../local_train_outputs")
    ap.add_argument("--z_root", type=str, required=True, help=".../final_output/preparation (client_XX/Z.npy)")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--img_root", type=str, required=True, help="이미지 루트")
    ap.add_argument("--metadata_csv", type=str, required=True, help="메타데이터 CSV (picker용)")

    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alpha_grid", type=str, default="0.0,0.3,0.5,0.7,1.0")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--temp_grid", type=str, default="1.0,0.7,0.5,1.5")

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    alpha_list = [args.alpha] if args.alpha is not None else parse_float_list(args.alpha_grid)
    temp_list  = [args.temp]  if args.temp  is not None else parse_float_list(args.temp_grid)
    clients = parse_clients(args.clients)

    summary_rows, best_rows = [], []

    for cid in clients:
        ckpt = os.path.join(args.local_dir, f"client_{pad2(cid)}", "best.pt")
        if not os.path.isfile(ckpt):
            ckpt2 = os.path.join(args.local_dir, f"client_{cid}", "best.pt")
            if os.path.isfile(ckpt2):
                ckpt = ckpt2
            else:
                print(f"[WARN] ckpt not found for client {cid}: {ckpt}")
                continue

        print(f"\n[client {cid:02d}] Loading model: {ckpt}")
        model = build_model_from_ckpt(ckpt, device)
        Z = load_Z_for_client(args.z_root, cid, device)

        loader = build_loader(
            test_csv=args.test_csv,
            label_csv=args.label_csv,
            client_id=cid,
            batch=args.batch,
            num_workers=args.num_workers,
            device=device,
            img_root=args.img_root,
            metadata_csv=args.metadata_csv
        )
        if len(loader.dataset) == 0:
            print(f"[WARN] Skipping client {cid:02d} because dataset is empty.")
            continue

        best_macro, best_combo = -1.0, (None, None)
        for alpha in alpha_list:
            handle = register_h_plus_alphaZ_hook(model, Z, alpha)
            try:
                for T in temp_list:
                    macro, _ = evaluate_one_client(model, loader, T=T, device=device)
                    summary_rows.append({"client": cid, "alpha": alpha, "temp": T, "macro_auroc": macro})
                    print(f"[client {cid:02d}] alpha={alpha:.3g}, T={T:.3g} -> macro AUROC={macro:.4f}")
                    if macro == macro and macro > best_macro:
                        best_macro, best_combo = macro, (alpha, T)
            finally:
                handle.remove()

        best_rows.append({
            "client": cid, "best_alpha": best_combo[0], "best_temp": best_combo[1], "best_macro_auroc": best_macro
        })
        print(f"[client {cid:02d}] BEST  alpha={best_combo[0]}  T={best_combo[1]}  macro={best_macro:.4f}")

    # 저장
    sum_csv = out_dir / "zero_shot_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["client","alpha","temp","macro_auroc"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n[Saved] {sum_csv}")

    best_csv = out_dir / "best_combo_per_client.csv"
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["client","best_alpha","best_temp","best_macro_auroc"])
        w.writeheader(); w.writerows(best_rows)
    print(f"[Saved] {best_csv}")

if __name__ == "__main__":
    main()
