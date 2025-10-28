# mimic-cxr
2025 이화여자대학교 컴퓨터공학과 졸업 프로젝트 - 연구트랙 

## Local 실행순서 
1. mimic-cxr\dataset_preparation\
    1) count_folders.py : 데이터 개수 확인
    2) client_split.py : 20명의 클라이언트에게 데이터 분배 -> mimic-cxr\client_splits\client_XX.csv 저장 
    3) make_test_split.py : test용 데이터 분배 -> mimic-cxr\client_splits\test.csv 저장

2. mimic-cxr\local_train
    1) train_local.py : 클라이언트 각자 train 이후 best.pt 저장 -> mimic-cxr\local_train_outputs\client_XX\best.pt, client_01\client_XX_metrics.json에 성능 저장 
    2) local_update.py : prep_clients.py가 만든 임베딩(repr_img.npy, repr_txt.npy 등)을 이용해 각 클라이언트의 분류 헤드(Linear)를 Z 기반으로 소규모 업데이트
  
   
3. mimic-cxr\local_test
    1) test_local.py : best.pt로 성능 확인 -> mimic-cxr\local_test_results\summary.csv 저장
   


## Global 실행순서 
1. global_train\
   1)  exract_embedding.py
        - MIMIC-CXR용 클라이언트 준비 스크립트 (로컬 파이프라인 연동판)
          (A) summary.csv에서 macro_auroc(및 loss) 읽어 client_{cid}_metrics.json 저장
          (B) client_{cid}.csv 전체에서 이미지/텍스트 임베딩 추출 → train_img_reps.npy / train_txt_reps.npy (경로 예시 : \mimic-cxr\local_train_outputs\client_01\train_img_reps.npy)
   2)  repr_kd.py : kd 실행 -> local_train_outputs\client_03\repr_txt_kd.npy 이런식으로 저장 
   ~3)  orchestrator: 글로벌 Z 계산 및 저장~
   3)  cluster_and_group_z.py : Clustering 진행 후 그룹별 Z 산출 -> global_output\clustering 디렉토리 아래 N개의 glabal Z 생성
   4)  map_img_txt_clusters.py : 클러스터링 단계에서 저장한 그룹별 서브센트로이드(.npy)를 읽어서 "이미지 클러스터 → 텍스트 클러스터" 매핑을 cross-attention 기반 코사인 유사도로 계산
