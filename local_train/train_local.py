# train_local.py
import math, torch, random, numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from .config import Cfg
from .data import ClientDataset, build_image_picker_from_metadata, load_label_table, img_transform
from .models import MultiModalLateFusion

class Trainer:
    def __init__(self, model, device, lr):
        self.model = model.to(device)
        self.device = device
        self.crit = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, batch, train=True):
        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train):
            logits = self.model(
                pixel_values=batch["pixel_values"].to(self.device),
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                img_mask=batch["img_mask"].to(self.device).unsqueeze(1),
                txt_mask=batch["txt_mask"].to(self.device).unsqueeze(1),
            )
            loss = self.crit(logits, batch["labels"].to(self.device))
            if train:
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit_epoch(self, loader):
        tot = n = 0
        for b in loader:
            tot += self.step(b, train=True) * b["labels"].size(0); n += b["labels"].size(0)
        return tot / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        tot = n = 0
        for b in loader:
            tot += self.step(b, train=False) * b["labels"].size(0); n += b["labels"].size(0)
        return tot / max(n, 1)


# ---------------------------
# 유틸: client_splits 경로 처리
# ---------------------------
def _default_split_dir(cfg: Cfg) -> Path:
    """
    client_splits를 가장 그럴듯한 위치에서 자동 탐색:
    1) cfg.CLIENT_CSV_DIR (있다면)
    2) 프로젝트 루트(mimic-cxr/mimic-cxr) 기준 client_splits
    3) 현재 작업폴더의 client_splits
    """
    candidates: List[Path] = []
    try:
        if getattr(cfg, "CLIENT_CSV_DIR", None):
            candidates.append(Path(cfg.CLIENT_CSV_DIR))
    except Exception:
        pass

    here = Path(__file__).resolve()
    # .../mimic-cxr/mimic-cxr/local_train/train_local.py → parents[1] = .../mimic-cxr/mimic-cxr
    candidates.append(here.parents[1] / "client_splits")
    candidates.append(Path.cwd() / "client_splits")

    for p in candidates:
        if p.exists():
            return p
    # 못 찾았을 때는 가장 유력한 경로를 반환(후속에서 친절한 메시지)
    return candidates[0]


def _resolve_client_csv(split_dir: Path, cid: int) -> Path:
    """
    client_{cid:02d}.csv 와 client_{cid}.csv 둘 다 지원.
    """
    cand = [split_dir / f"client_{cid:02d}.csv",
            split_dir / f"client_{cid}.csv"]
    for p in cand:
        if p.exists():
            return p
    have = sorted([p.name for p in split_dir.glob("client_*.csv")])
    raise FileNotFoundError(
        f"[client_splits] cid={cid}에 해당하는 CSV를 찾을 수 없습니다.\n"
        f"검사한 경로: {cand[0]}  또는  {cand[1]}\n"
        f"split_dir='{split_dir}' 내 보유 파일: {have}"
    )


def _mode_for_client(cid: int) -> str:
    if cid == 0:
        return "test_mix"
    if 1 <= cid <= 16:
        return "multimodal"
    if cid in (17, 18):
        return "image_only"
    if cid in (19, 20):
        return "text_only"
    raise ValueError(f"invalid client_id: {cid}")


def run_client(client_id: int, cfg: Cfg = Cfg(), split_dir: Optional[Path] = None):
    # 재현성
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    # 라벨/메타 준비
    label_csv = cfg.LABEL_CSV_NEGBIO if cfg.USE_LABEL == "negbio" else cfg.LABEL_CSV_CHEXPERT
    label_table = load_label_table(str(label_csv), cfg.LABEL_COLUMNS)
    meta_picker = build_image_picker_from_metadata(str(cfg.METADATA_CSV), str(cfg.IMG_ROOT))

    mode = _mode_for_client(client_id)

    if client_id == 0:
        csv_path = cfg.TEST_CSV_PATH
    else:
        # split_dir 우선순위: 인자로 받은 경로 > 자동탐색 > cfg.CLIENT_CSV_DIR
        base_dir = Path(split_dir) if split_dir else _default_split_dir(cfg)
        if not base_dir.exists():
            raise FileNotFoundError(
                f"[client_splits] 기본 경로를 찾을 수 없습니다: '{base_dir}'. "
                f"--split_dir 로 명시적으로 지정해 주세요."
            )
        csv_path = _resolve_client_csv(base_dir, client_id)

    # Dataset/DataLoader
    ds = ClientDataset(str(csv_path), label_table, mode, meta_picker,
                       cfg.TEXT_MODEL_NAME, cfg.MAX_LEN, img_transform())

    n = len(ds); n_tr = int(n * 0.9); n_val = n - n_tr
    tr_set, val_set = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_set, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=cfg.NUM_WORKERS)
    val_loader= DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    num_classes = len(cfg.LABEL_COLUMNS)
    model = MultiModalLateFusion(num_classes, text_model_name=cfg.TEXT_MODEL_NAME)
    trainer = Trainer(model, device=("cuda" if torch.cuda.is_available() else "cpu"), lr=cfg.LR)

    best = math.inf
    save_dir = Path("./local_outputs") / f"client_{client_id:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.EPOCHS + 1):
        tr = trainer.fit_epoch(tr_loader)
        val = trainer.evaluate(val_loader)
        print(f"[Client {client_id:02d}] Epoch {epoch}/{cfg.EPOCHS} | train {tr:.4f} | val {val:.4f}")
        if val < best:
            best = val
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val,
                "label_columns": cfg.LABEL_COLUMNS,
                "mode": mode,
            }, save_dir / "best.pt")
    print(f"[Client {client_id:02d}] Done. Best val loss: {best:.4f}  (saved at {save_dir/'best.pt'})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # 여러 개의 client_id를 공백으로 나열해 순차 실행 가능 (ex: --client_id 1 2 3)
    ap.add_argument("--client_id", type=int, nargs="+", required=True,
                    help="클라이언트 ID (여러 개 입력하면 순차 실행). 0은 test_mix.")
    ap.add_argument("--split_dir", type=str, default=None,
                    help="client_splits 디렉토리 경로를 직접 지정(옵션).")
    args = ap.parse_args()

    # split_dir 해석
    split_dir_path = Path(args.split_dir) if args.split_dir else None

    # 여러 client 순차 실행
    for cid in args.client_id:
        run_client(cid, cfg=Cfg(), split_dir=split_dir_path)
