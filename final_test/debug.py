# quick_join_probe.py
import pandas as pd, os

BASE = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr"

def norm(s: pd.Series, kind: str):
    s = s.astype(str).str.strip()
    if kind == "sub": s = s.str.replace(r'^[pP]', '', regex=True)
    if kind == "sty": s = s.str.replace(r'^[sS]', '', regex=True)
    s = s.str.replace(r'[^0-9]', '', regex=True).replace('', pd.NA)
    return s.astype('Int64')

test = pd.read_csv(os.path.join(BASE, "client_splits", "test.csv"))[["subject_id","study_id","image_dir","text_path"]]
test["subject_id_n"] = norm(test["subject_id"], "sub")
test["study_id_n"]   = norm(test["study_id"], "sty")

for cid in [1]:
    c = pd.read_csv(os.path.join(BASE, "client_splits", f"client_{cid:02d}.csv"))[["subject_id","study_id","image_dir","text_path"]]
    c["subject_id_n"] = norm(c["subject_id"], "sub")
    c["study_id_n"]   = norm(c["study_id"], "sty")

    # 1) 키 교집합 크기
    key_test = test[["subject_id_n","study_id_n"]].dropna().drop_duplicates()
    key_cli  = c[["subject_id_n","study_id_n"]].dropna().drop_duplicates()
    inter = key_test.merge(key_cli, on=["subject_id_n","study_id_n"], how="inner")
    print(f"[client_{cid:02d}] test keys={len(key_test)}, client keys={len(key_cli)}, INTERSECTION={len(inter)}")

    # 2) 앞에서 5개만 보여줘 (비교용)
    print("  sample test keys:", key_test.head().to_dict("records"))
    print("  sample client keys:", key_cli.head().to_dict("records"))

    # 3) 경로 기반 fallback도 간단 체크(부분문자열)
    tpaths = (test["image_dir"].astype(str).tolist() + test["text_path"].astype(str).tolist())
    cpaths = (c["image_dir"].astype(str).tolist() + c["text_path"].astype(str).tolist())
    hit = 0
    for cp in cpaths[:50]:  # 일부만 검사
        if not cp or cp == "nan": continue
        for tp in tpaths[:200]:
            if cp in tp:
                hit += 1; break
    print(f"  path-based (subset) hits ~ {hit}")
