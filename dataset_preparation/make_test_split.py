# make_test_split.py
# all_clients_master.csv를 읽고, 배분에서 빠진 나머지 스터디를 찾아 test.csv로 저장

import csv
from pathlib import Path

IMG_ROOT = r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files"
TXT_ROOT = r"D:\mimic-cxr\physionet.org\files\mimic-cxr\2.1.0\files"
P_BUCKETS = [f"p{n}" for n in range(10, 18)] # p10~p17

MASTER_CSV = r".\client_splits\all_clients_master.csv"
OUT_TEST = r".\client_splits\test.csv"

def iter_all_studies():
    from pathlib import Path
    for p in sorted(P_BUCKETS):
        pdir = Path(IMG_ROOT)/p
        if not pdir.exists(): 
            continue
        for subj in sorted([d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("p")]):
            for sdir in sorted([d for d in subj.iterdir() if d.is_dir() and d.name.startswith("s")]):
                study_id = sdir.name
                yield {
                    "p_bucket": p,
                    "subject_id": subj.name,
                    "study_id": study_id,
                    "image_dir": str(sdir),
                    "text_path": str(Path(TXT_ROOT)/p/subj.name/f"{study_id}.txt"),
                }

def load_assigned_master():
    assigned = set()
    with open(MASTER_CSV, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["p_bucket"], row["subject_id"], row["study_id"])
            assigned.add(key)
    return assigned

def write_csv(rows, path):
    headers = ["client_id","modality","p_bucket","subject_id","study_id","image_dir","text_path","has_image","has_text"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for rec in rows:
            w.writerow(rec)

def main():
    assigned = load_assigned_master()
    test_rows = []
    for rec in iter_all_studies():
        key = (rec["p_bucket"], rec["subject_id"], rec["study_id"])
        if key in assigned: 
            continue
        test_rows.append({
            "client_id": 0,                   # 테스트셋 표시용
            "modality": "test_mix",           # 필요시 "image_only"/"text_only"로 바꿔도 됨
            "p_bucket": rec["p_bucket"],
            "subject_id": rec["subject_id"],
            "study_id": rec["study_id"],
            "image_dir": rec["image_dir"],
            "text_path": rec["text_path"],
            "has_image": True,
            "has_text": True,
        })
    Path(OUT_TEST).parent.mkdir(parents=True, exist_ok=True)
    write_csv(test_rows, OUT_TEST)
    print(f"[INFO] test.csv saved: {len(test_rows):,} rows at {OUT_TEST}")

if __name__ == "__main__":
    main()
