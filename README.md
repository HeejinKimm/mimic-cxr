# mimic-cxr
2025 졸업 프로젝트

Global 실행순서 
1. global_train/
   1)  exract_embedding.py : representation vector 추출
   2)  repr_kd.py : kd 실행
   3)  orchestrator: Z 추출
   4)  local_gating
2. local_gating/
   1) train_local_kd.py : local gating 알고리즘으로 z 업데이트 => 해당 output : global_output/client_XX/global_payload.pt
