#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prepare_local_update.py  (2025-10-17)

목적:
1) 클러스터링 결과(csv) 읽기
   - global_output/clustering/img_client_assignments.csv
   - global_output/clustering/txt_client_assignments.csv
   - global_output/clustering/img_to_txt_mapping_optimal.csv

2) 각 client별 Z 선택 규칙
   - (멀티모달/이미지 온리) img_cluster → img2txt 매핑으로 txt_cluster를 확정
   - (텍스트 온리: 예 19, 20) txt_cluster를 고정으로 사용.
       img2txt를 역참조(txt→img 후보 리스트)하여
       (a) Z 파일이 실제 존재하고, (b) 해당 img_cluster에 할당된 클라이언트 수가 가장 많은
       후보를 고른다. (동률이면 img_cluster 번호가 작은 쪽)

3) final_output/preparation/client_XX 생성 후
   - global_output/final_output_z/에서 Z_img{img}_txt{txt}.npy (0-padding 변형 포함)을 찾아
     client_XX/Z.npy 로 복사 (+원본명 백업, mapping_info.txt 기록)

사용 예:
python prepare_local_update.py ^
  --root "C:\\HJHJ0808\\김희진\\연구\\졸업프로젝트\\mimic-cxr\\mimic-cxr"
"""

import argparse
from pathlib import Path
import shutil
import sys
import re
from typing import Dict, Tuple, Optional, List

import pandas as pd


# ---------------------------
# CSV 로딩/칼럼 탐지 유틸
# ---------------------------
def find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """DataFrame에서 key(대소문자무시)가 이름에 포함된 첫 번째 칼럼명 반환."""
    key_l = key.lower()
    for c in df.columns:
        if key_l in c.lower():
            return c
    return None


def read_client_assignments(path: Path) -> Dict[int, int]:
    """
    client -> cluster 매핑 읽기.
    칼럼명이 다를 수 있어 'client', 'cluster' 키워드로 자동 탐지.
    실패 시 숫자형 2개 칼럼을 client/cluster로 간주.
    """
    df = pd.read_csv(path)
    c_client = find_col(df, "client")
    c_cluster = find_col(df, "cluster")

    if c_client is None or c_cluster is None:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            c_client = num_df.columns[0]
            c_cluster = num_df.columns[1]
        else:
            raise ValueError(
                f"[ERROR] '{path}'에서 client/cluster 칼럼을 찾지 못했습니다. "
                f"칼럼들: {list(df.columns)}"
            )

    mapping = {}
    for _, row in df.iterrows():
        client_id = int(row[c_client])
        cluster_id = int(row[c_cluster])
        mapping[client_id] = cluster_id
    return mapping


def read_img_to_txt_mapping(path: Path) -> Dict[int, int]:
    """
    이미지 클러스터 -> 텍스트 클러스터 매핑 로드.
    칼럼명에서 'img', 'txt' 키워드를 우선 탐지, 실패 시 숫자형 2개 칼럼을 사용.
    """
    df = pd.read_csv(path)
    c_img = None
    c_txt = None
    for c in df.columns:
        cl = c.lower()
        if "img" in cl and c_img is None:
            c_img = c
        if "txt" in cl and c_txt is None:
            c_txt = c

    if c_img is None or c_txt is None:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            c_img, c_txt = num_df.columns[0], num_df.columns[1]
        else:
            raise ValueError(
                f"[ERROR] '{path}'에서 img/txt 칼럼을 찾지 못했습니다. "
                f"칼럼들: {list(df.columns)}"
            )

    mapping = {}
    for _, row in df.iterrows():
        img_k = int(row[c_img])
        txt_k = int(row[c_txt])
        mapping[img_k] = txt_k
    return mapping


def invert_img2txt(img2txt: Dict[int, int]) -> Dict[int, List[int]]:
    """
    img→txt 매핑을 txt→[img 후보들]로 뒤집기.
    """
    inv: Dict[int, List[int]] = {}
    for img_k, txt_k in img2txt.items():
        inv.setdefault(txt_k, []).append(img_k)
    for k in inv:
        inv[k] = sorted(inv[k])
    return inv


# ---------------------------
# Z 파일 탐색
# ---------------------------
def glob_z_file(z_dir: Path, img_k: int, txt_k: int) -> Optional[Path]:
    """
    global_output/final_output_z/ 안에서 (img_k, txt_k) 조합의 Z 파일을 유연 탐색.
    지원 패턴:
      Z_img{img}_txt{txt}.npy
      Z_img{img:02d}_txt{txt:02d}.npy
      Z_img{img}_txt0{txt}.npy
      Z_img0{img}_txt{txt}.npy
      Z_img{img}-txt{txt}.npy
    마지막으로 정규식 스캔(r"Z_img0*(\d+)_txt0*(\d+)\.npy$")
    """
    img_keys = {f"{img_k}", f"{img_k:02d}", f"0{img_k}"}
    txt_keys = {f"{txt_k}", f"{txt_k:02d}", f"0{txt_k}"}

    for ik in img_keys:
        for tk in txt_keys:
            for pat in [
                f"Z_img{ik}_txt{tk}.npy",
                f"Z_img{ik}_txt0{tk}.npy",
                f"Z_img0{ik}_txt{tk}.npy",
                f"Z_img{ik}-txt{tk}.npy",
            ]:
                p = z_dir / pat
                if p.exists():
                    return p

    rx = re.compile(r"Z_img0*(\d+)_txt0*(\d+)\.npy$", re.IGNORECASE)
    if z_dir.exists():
        for p in sorted(z_dir.glob("Z_*.npy")):
            m = rx.search(p.name)
            if not m:
                continue
            try:
                ik = int(m.group(1))
                tk = int(m.group(2))
                if ik == img_k and tk == txt_k:
                    return p
            except Exception:
                pass
    return None


def find_z_for_txt_only(
    z_dir: Path,
    txt_k: int,
    img2txt_inv: Dict[int, List[int]],
    img_assign: Dict[int, int]
) -> Tuple[Optional[int], Optional[Path]]:
    """
    텍스트 온리 클라이언트용:
    - txt_k(텍스트 클러스터 번호)는 고정.
    - txt_k로 연결되는 img_k 후보들 중
        1) (img_k, txt_k) Z 파일이 실제 존재
        2) img_k에 할당된 클라이언트 수가 가장 많은
      후보를 선택 (동률이면 번호가 작은 img_k).
    """
    candidates = img2txt_inv.get(txt_k, [])
    if not candidates:
        return None, None

    # img_k별 할당된 클라이언트 수
    img_counts: Dict[int, int] = {}
    for _, k in img_assign.items():
        img_counts[k] = img_counts.get(k, 0) + 1

    viable: List[Tuple[int, Path, int]] = []  # (img_k, z_path, count)
    for img_k in candidates:
        z_path = glob_z_file(z_dir, img_k=img_k, txt_k=txt_k)
        if z_path is None:
            continue
        count = img_counts.get(img_k, 0)
        viable.append((img_k, z_path, count))

    if not viable:
        return None, None

    viable.sort(key=lambda x: (-x[2], x[0]))  # count 내림차순, img_k 오름차순
    chosen_img_k, chosen_z, _ = viable[0]
    return chosen_img_k, chosen_z


# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr",
        help="프로젝트 루트 (mimic-cxr/mimic-cxr)"
    )
    ap.add_argument(
        "--dryrun",
        action="store_true",
        help="복사하지 않고 예정 작업만 출력"
    )
    args = ap.parse_args()

    ROOT = Path(args.root)

    # 입력 경로
    d_cluster = ROOT / "global_output" / "clustering"
    p_img_assign = d_cluster / "img_client_assignments.csv"
    p_txt_assign = d_cluster / "txt_client_assignments.csv"
    p_map = d_cluster / "img_to_txt_mapping_optimal.csv"

    d_final_z = ROOT / "global_output" / "final_output_z"

    # 출력 경로
    d_prep = ROOT / "final_output" / "preparation"
    d_prep.mkdir(parents=True, exist_ok=True)

    # 존재 확인
    for p in [p_img_assign, p_txt_assign, p_map]:
        if not p.exists():
            raise FileNotFoundError(f"입력 누락: {p}")
    if not d_final_z.exists():
        raise FileNotFoundError(f"Z 폴더 누락: {d_final_z}")

    # 로드
    img_assign = read_client_assignments(p_img_assign)  # client -> img_cluster
    txt_assign = read_client_assignments(p_txt_assign)  # client -> txt_cluster
    img2txt = read_img_to_txt_mapping(p_map)            # img_cluster -> txt_cluster
    txt2img_inv = invert_img2txt(img2txt)               # txt_cluster -> [img_cluster 후보]

    # 전체 클라이언트 집합
    all_clients = sorted(set(img_assign.keys()) | set(txt_assign.keys()))
    print(f"[INFO] 총 클라이언트 수: {len(all_clients)}  -> 첫 8명: {all_clients[:8]}")

    failures = []

    for cid in all_clients:
        img_k = img_assign.get(cid, None)
        txt_k_from_txt_assign = txt_assign.get(cid, None)

        # Case A) 이미지가 있는 경우 (멀티모달/이미지 온리)
        if img_k is not None:
            txt_k = img2txt.get(img_k, None)
            if txt_k is None:
                print(f"[WARN] client_{cid:02d}: img_cluster={img_k} → txt 매핑 없음. 스킵.")
                continue

            z_path = glob_z_file(d_final_z, img_k=img_k, txt_k=txt_k)
            if z_path is None:
                print(f"[ERROR] client_{cid:02d}: (img={img_k}, txt={txt_k}) Z 파일 없음.")
                failures.append((cid, img_k, txt_k, "img→txt 매핑 기반"))
                continue

            out_dir = d_prep / f"client_{cid:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[OK] client_{cid:02d}: {z_path.name} -> {out_dir}")

            if not args.dryrun:
                try:
                    shutil.copy2(z_path, out_dir / "Z.npy")
                    if z_path.name != "Z.npy":
                        shutil.copy2(z_path, out_dir / z_path.name)
                    (out_dir / "mapping_info.txt").write_text(
                        f"client_id={cid}\nimg_cluster={img_k}\n"
                        f"mapped_txt_cluster={txt_k}\nZ_file={z_path.name}\nmode=img_to_txt\n",
                        encoding="utf-8"
                    )
                except Exception as e:
                    print(f"[ERROR] 복사 실패 client_{cid:02d}: {e}")
                    failures.append((cid, img_k, txt_k, "copy_error"))
            continue

        # Case B) 텍스트 온리: txt_k를 고정으로 사용
        if txt_k_from_txt_assign is not None:
            chosen_img_k, z_path = find_z_for_txt_only(
                z_dir=d_final_z,
                txt_k=txt_k_from_txt_assign,
                img2txt_inv=txt2img_inv,
                img_assign=img_assign,
            )
            if z_path is None or chosen_img_k is None:
                print(f"[ERROR] client_{cid:02d}(txt-only): txt_k={txt_k_from_txt_assign}에 대응하는 Z 없음.")
                failures.append((cid, None, txt_k_from_txt_assign, "txt-only 매칭 실패"))
                continue

            out_dir = d_prep / f"client_{cid:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[OK] client_{cid:02d}(txt-only): {z_path.name} -> {out_dir}  (img_k={chosen_img_k}, txt_k={txt_k_from_txt_assign})")

            if not args.dryrun:
                try:
                    shutil.copy2(z_path, out_dir / "Z.npy")
                    if z_path.name != "Z.npy":
                        shutil.copy2(z_path, out_dir / z_path.name)
                    (out_dir / "mapping_info.txt").write_text(
                        f"client_id={cid}\nimg_cluster={chosen_img_k}\n"
                        f"txt_cluster={txt_k_from_txt_assign}\nZ_file={z_path.name}\nmode=txt_only_fixed_txt\n",
                        encoding="utf-8"
                    )
                except Exception as e:
                    print(f"[ERROR] 복사 실패 client_{cid:02d}(txt-only): {e}")
                    failures.append((cid, chosen_img_k, txt_k_from_txt_assign, "copy_error"))
            continue

        # Case C) 둘 다 없음
        print(f"[WARN] client_{cid:02d}: img/txt 클러스터 정보 모두 없음. 스킵.")
        failures.append((cid, None, None, "no_clusters"))

    # 요약
    if failures:
        print("\n[SUMMARY] 실패 목록 (client, img_k, txt_k, note):")
        for item in failures:
            print("  -", item)
        sys.exit(1)
    else:
        print("\n[SUMMARY] 모든 클라이언트에 대해 준비 완료.")
        sys.exit(0)


if __name__ == "__main__":
    main()
