# client_split.py 
# MIMIC-CXR 디렉터리(p10~p17)를 훑어서 (subject_id, study_id) 단위의 스터디 레코드를 모은 뒤, 
# 20개 클라이언트에 **균등 개수(PER_CLIENT=8,400)**로 잘라 CSV로 내보내는 유틸
# 전체 데이터 개수 세고 -> 각 클라이언트(20명)한테 적절히 배분 진행

# 사용되는 파일 : Mimic-cxr dataset 
# output file
# 1. client_splits/all_clients_master.csv = 모든 클라이언트 레코드 합본
# 2. client_splits/client_id.csv - 각 클라이언트별 샘플 목록 
# **레코드 컬럼 : client_id, modality, p_bucket, subject_id, study_id, image_dir, text_path, has_image, has_text

import os
import csv
from pathlib import Path
from typing import Iterator, Dict, List

# ==============================
# 경로 설정(필요 시 수정)
# ==============================
IMG_ROOT = r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files"   # ...\p10\p10000032\s50414267
TXT_ROOT = r"D:\mimic-cxr\physionet.org\files\mimic-cxr\2.1.0\files"  # ...\p10\p10000032\s50414267.txt
P_BUCKETS = [f"p{n}" for n in range(10, 18)]  # p10~p17

NUM_CLIENTS = 20
PER_CLIENT = 8400
OUT_DIR = r".\client_splits"
os.makedirs(OUT_DIR, exist_ok=True)

# 클라이언트 구성
MULTIMODAL_CLIENTS = list(range(1, 17))  # 1..16
IMG_ONLY_CLIENTS   = [17, 18]
TXT_ONLY_CLIENTS   = [19, 20]

# ==============================
# 스터디 수집 (모두 페어 존재 가정)
# ==============================
def iter_studies_paired(img_root: str, txt_root: str, p_buckets) -> Iterator[Dict]:
    """
    pXX/pYYYYYYYY/sZZZZZZZZ 를 정렬 순서대로 yield.
    모든 스터디가 이미지·텍스트 페어가 있다고 가정하므로 존재 여부 체크는 생략.
    """
    for p in sorted(p_buckets): # p10~p17
        p_dir = Path(img_root) / p
        if not p_dir.exists():
            continue
        # subject 디렉토리 정렬
        for subj_dir in sorted([d for d in p_dir.iterdir() if d.is_dir() and d.name.startswith("p")]): # pYYYYYYYY
            subject_id = subj_dir.name
            # study 디렉토리 정렬
            for study_dir in sorted([d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("s")]): # sZZZZZZZZ
                study_id = study_dir.name
                yield {
                    "p_bucket": p,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "image_dir": str(study_dir),
                    "text_path": str(Path(txt_root) / p / subject_id / f"{study_id}.txt"),
                }

def take_first_n(it: Iterator[Dict], n: int) -> List[Dict]:  # 168,000개 만큼만 앞에서 자라 리스트로 만드는 것 (나머지 데이터는 test로 사용할 것)
    buf = []
    for i, rec in enumerate(it):
        if i >= n:
            break
        buf.append(rec)
    return buf

# ==============================
# CSV 저장
# ==============================
def write_csv(rows, out_path):
    headers = [
        "client_id","modality","p_bucket","subject_id","study_id",
        "image_dir","text_path","has_image","has_text"
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ==============================
# 메인
# ==============================
def main():
    total_needed = NUM_CLIENTS * PER_CLIENT  # 168,000
    print(f"[INFO] 총 필요 수량: {total_needed:,}개 (클라이언트 {NUM_CLIENTS}명 × {PER_CLIENT}개)")

    # 1) p10~p17에서 앞에서부터 total_needed개만 추출해서 리스트에 담음. 
    all_recs = take_first_n(iter_studies_paired(IMG_ROOT, TXT_ROOT, P_BUCKETS), total_needed)
    if len(all_recs) < total_needed:
        print(f"[경고] 스캔 수량 부족: {len(all_recs):,}/{total_needed:,}")
    else:
        print(f"[INFO] 스캔 및 사용 예정 수량: {len(all_recs):,}")

    # 2) 균등 슬라이싱으로 20명 × 6500개 배분
    client_rows = {cid: [] for cid in range(1, NUM_CLIENTS+1)}
    idx = 0

    def make_row(rec, client_id, modality): 
        if modality == "multimodal":
            return {
                "client_id": client_id,
                "modality": "multimodal",
                "p_bucket": rec["p_bucket"],
                "subject_id": rec["subject_id"],
                "study_id": rec["study_id"],
                "image_dir": rec["image_dir"],
                "text_path": rec["text_path"],
                "has_image": True,
                "has_text": True,
            }
        elif modality == "image_only":
            return {
                "client_id": client_id,
                "modality": "image_only",
                "p_bucket": rec["p_bucket"],
                "subject_id": rec["subject_id"],
                "study_id": rec["study_id"],
                "image_dir": rec["image_dir"],
                "text_path": "",      # 텍스트 제공 안 함
                "has_image": True,
                "has_text": False,
            }
        elif modality == "text_only":
            return {
                "client_id": client_id,
                "modality": "text_only",
                "p_bucket": rec["p_bucket"],
                "subject_id": rec["subject_id"],
                "study_id": rec["study_id"],
                "image_dir": "",      # 이미지 제공 안 함
                "text_path": rec["text_path"],
                "has_image": False,
                "has_text": True,
            }
        else:
            raise ValueError("Unknown modality")

    # 2-A) 1~16: 멀티모달
    for cid in MULTIMODAL_CLIENTS:
        slice_rec = all_recs[idx: idx + PER_CLIENT]
        for rec in slice_rec:
            client_rows[cid].append(make_row(rec, cid, "multimodal"))
        idx += len(slice_rec)

    # 2-B) 17~18: 이미지 전용
    for cid in IMG_ONLY_CLIENTS:
        slice_rec = all_recs[idx: idx + PER_CLIENT]
        for rec in slice_rec:
            client_rows[cid].append(make_row(rec, cid, "image_only"))
        idx += len(slice_rec)

    # 2-C) 19~20: 텍스트 전용
    for cid in TXT_ONLY_CLIENTS:
        slice_rec = all_recs[idx: idx + PER_CLIENT]
        for rec in slice_rec:
            client_rows[cid].append(make_row(rec, cid, "text_only"))
        idx += len(slice_rec)

    # 3) 저장
    master = []
    for cid in range(1, NUM_CLIENTS+1):
        rows = client_rows[cid]
        out_csv = os.path.join(OUT_DIR, f"client_{cid:02d}.csv")
        write_csv(rows, out_csv)
        master.extend(rows)

    write_csv(master, os.path.join(OUT_DIR, "all_clients_master.csv"))

    # 4) 요약
    print("\n[배분 요약]")
    print(f" - 실제 배분: {len(master):,}개 (클라이언트당 {PER_CLIENT}개 × {NUM_CLIENTS}명)")
    for cid in range(1, NUM_CLIENTS+1):
        mm = sum(r["modality"] == "multimodal" for r in client_rows[cid])
        io = sum(r["modality"] == "image_only"  for r in client_rows[cid])
        to = sum(r["modality"] == "text_only"   for r in client_rows[cid])
        print(f"   · Client {cid:02d}: total={len(client_rows[cid]):5d} (MM={mm:5d}, IMG={io:5d}, TXT={to:5d})")

if __name__ == "__main__":
    main()
