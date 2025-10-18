import numpy as np

path = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\global_output\final_output_z\Z_img0_txt0.npy"  # 확인할 파일 경로
arr = np.load(path, allow_pickle=False)
print("type:", type(arr))
print("shape:", getattr(arr, "shape", None))
print("dtype:", getattr(arr, "dtype", None))

import numpy as np
a = np.load(r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\global_output\final_output_z\Z_img1_txt1.npy", allow_pickle=False)

print(a.shape)   # (256,)
print(a.ndim)    # 1  -> 1차원
print(a.size)    # 256 -> 전체 원소 개수
print(a.dtype)   # float32
print(a.nbytes)  # 바이트 크기(= 256 * 4 = 1024 bytes)