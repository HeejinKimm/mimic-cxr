from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
     # ------- 경로 -------
    # 각 client_XX 폴더가 있는 루트(= train_local이 저장하는 곳)
    BASE_DIR: str = r".\outputs"
    # 글로벌 산출물(Z, 페이로드, 리포트 등)
    OUT_GLOBAL_DIR: str = r".\outputs"

    # ------- 클라이언트 그룹 -------
    # 1~16 멀티모달, 17~18 이미지 온리, 19~20 텍스트 온리
    FUSION_CLIENTS: Tuple[int, ...] = tuple(range(1, 17))
    IMAGE_ONLY: Tuple[int, ...] = (17, 18)
    TEXT_ONLY:  Tuple[int, ...] = (19, 20)

    # ------- 라벨/클래스 수 -------
    # train_local.py 의 LABEL_COLUMNS 길이 = 13
    NUM_CLASSES: int = 13


    # ------- 체크포인트 파일명 -------
    # outputs/client_XX/best.pt 로 저장하므로 이 이름을 기본값으로
    CKPT_NAME: str = "best.pt"


    # ------- 임베딩/모델 차원 -------
    # ImageHead(backbone)는 MobileNetV3-Small → 특징 차원 576
    IMG_DIM: int = 576
    # TextHead는 prajjwal1/bert-mini → hidden_size 256
    TXT_DIM: int = 256
    # 로컬 결합 표현 기대 차원(필요시 사용). 보통 576+256 = 832
    FUSED_DIM: int = 832
    # 글로벌 Z 차원(오케스트레이터가 PCA/투영해 만드는 공통 차원)
    D_MODEL: int = 256


    # ------- 클러스터링/샘플링 하이퍼 -------
    # 오케스트레이터가 각 클라에서 최대 몇 개 임베딩을 모을지
    SAMPLE_PER_CLIENT: int = 10000
    # 이미지/텍스트 클러스터 개수(설계상 기본 4)
    K_IMG: int = 4
    K_TXT: int = 4
    # KMeans n_init
    KMEANS_N_INIT: str | int = "auto"


    # ------- KD/그룹핑 관련 -------
    # summary.csv에서 그룹 나눌 때 쓸 메트릭(높을수록 좋음)
    METRIC_NAME: str = "macro_auroc"
    # (벡터 레벨 KD 등을 쓸 때) 표현 정합 가중치 / 온도
    KD_REP_WEIGHT: float = 0.2
    KD_TEMP: float = 2.0


    # ------- 기타 -------
    METRICS_SNAPSHOT: str = "metrics_snapshot.json"
    KD_PLAN_JSON: str = "kd_plan.json"
    SEED: int = 42

cfg = Config()

