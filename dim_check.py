# import numpy as np

# path = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\global_output\final_output_z\Z_img0_txt0.npy"  # 확인할 파일 경로
# arr = np.load(path, allow_pickle=False)
# print("type:", type(arr))
# print("shape:", getattr(arr, "shape", None))
# print("dtype:", getattr(arr, "dtype", None))

# import numpy as np
# a = np.load(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\global_output\final_output_z\Z_img1_txt1.npy", allow_pickle=False)

# print(a.shape)   # (256,)
# print(a.ndim)    # 1  -> 1차원
# print(a.size)    # 256 -> 전체 원소 개수
# print(a.dtype)   # float32
# print(a.nbytes)  # 바이트 크기(= 256 * 4 = 1024 bytes)

import torch

ckpt_path = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\local_train_outputs\client_01\best.pt"

# 모델 로드 (경고 무시 가능)
ckpt = torch.load(ckpt_path, map_location="cpu")

# 모델 state_dict 안쪽에 들어 있을 수도 있음
if "model" in ckpt:
    sd = ckpt["model"]
else:
    sd = ckpt

print(f"총 tensor 개수: {len(sd)}")
print("="*60)

for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:50s} | shape={tuple(v.shape)} | dtype={v.dtype}")
