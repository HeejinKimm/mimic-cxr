# mimic-cxr
2025 이화여자대학교 컴퓨터공학과 졸업 프로젝트 - 연구트랙 
Local 실행순서 
1. mimic-cxr\dataset_preparation\
    1) count_folders.py : 데이터 개수 확인
    2) client_split.py : 20명의 클라이언트에게 데이터 분배 -> mimic-cxr\client_splits\client_XX.csv 저장 
    3) make_test_split.py : test용 데이터 분배 -> mimic-cxr\client_splits\test.csv 저장 
   
Global 실행순서 
1. global_train/
   1)  exract_embedding.py : representation vector 추출
   2)  repr_kd.py : kd 실행
   3)  orchestrator: Z 추출
   4)  local_gating
2. local_gating/
   1) train_local_kd.py : local gating 알고리즘으로 z 업데이트 => 해당 output : global_output/client_XX/global_payload.pt
