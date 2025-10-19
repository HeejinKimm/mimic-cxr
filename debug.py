import numpy as np, torch
from pathlib import Path

cid = 1
npz_path = Path(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\outputs\client_01\updated_heads.npz")

print("[파일 존재 여부]", npz_path.exists())
data = np.load(npz_path)
print("[키 목록]", data.files)

for k in data.files:
    print(k, data[k].shape)