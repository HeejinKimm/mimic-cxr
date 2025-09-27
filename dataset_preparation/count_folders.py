# 데이터 전체 개수 세기 
# s로 시작하는 하위폴더

import os

def count_s_folders(root_folder):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for d in dirnames:
            if d.startswith("s"):   # s로 시작하는 하위폴더
                count += 1
    return count

# 사용 예시
root = r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files\p17"
num_s = count_s_folders(root)
print(f"{root} 아래에 's'로 시작하는 폴더 개수: {num_s}")