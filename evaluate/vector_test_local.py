# -*- coding: utf-8 -*-
"""
update_and_eval_dirs.py
- (A) 임베딩 + Z 정렬 게이팅으로 분류 헤드(선형) 소규모 업데이트 → updated_heads.npz
- (B) test.csv 없이, 사용자 지정 스터디 폴더들을 원본 파일로 로드해
      baseline(best.pt) vs updated(updated_heads 주입) 성능 비교
      (멀티모달은 모달 마스크 기반 late fusion; 텍스트 없으면 이미지만 사용)

예)
  python -u ./evaluate/vector_test_local.py --clients 1-20 --label_csv .\\mimic-cxr-2.0.0-negbio.csv --test_root .\\data
"""
import os, csv, json, argparse, re, glob
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score
from PIL import Image, ImageFile

# --------------------------
# 공통 설정
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]
TEXT_MODEL_NAME = "prajjwal1/bert-mini"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary_compare_raw.csv"

# --------------------------
# 경로/유틸
# --------------------------
def cdir(cid: int) -> Path:
    p = Path(f"./outputs/client_{cid:02d}")
    p.mkdir(parents=True, exist_ok=True)
    return p

def id_to_int_from_path(study_dir: str) -> Tuple[int,int]:
    """
    .../p10/p100032/s50414267 -> (100032, 50414267)
    """
    m_p = re.search(r"[\\/]p(\d+)[\\/]", study_dir)
    m_s = re.search(r"[\\/]s(\d+)[\\/]?$", study_dir)
    if not m_p or not m_s:
        raise ValueError(f"경로에서 subject/study를 파싱할 수 없습니다: {study_dir}")
    return int(m_p.group(1)), int(m_s.group(1))

def load_label_table(label_csv: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd
    df = pd.read_csv(label_csv)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"라벨 CSV에 필요한 칼럼이 없습니다: {miss}")
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)
    table = {}
    for _, r in df.iterrows():
        key = (int(r["subject_id"]), int(r["study_id"]))
        table[key] = [int(r[c]) for c in LABEL_COLUMNS]
    return table

def parse_test_dirs(args) -> List[str]:
    paths: List[str] = []
    if args.test_dirs:
        for p in args.test_dirs.split(","):
            p = p.strip()
            if p: paths.append(p)
    if args.test_list and os.path.exists(args.test_list):
        with open(args.test_list, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p: paths.append(p)
    if args.test_root and os.path.exists(args.test_root):
        # p*/s* 형태의 스터디 폴더를 재귀 검색
        # 예: data/p10/**/s12345678
        cand = glob.glob(os.path.join(args.test_root, "**", "s*"), recursive=True)
        paths.extend(cand)
    # 폴더만 유지, 중복 제거
    paths = [p for p in paths if os.path.isdir(p)]
    paths = sorted(list(dict.fromkeys(paths)))
    if not paths:
        raise RuntimeError("테스트 스터디 폴더가 비었습니다. --test_dirs / --test_list / --test_root 중 하나를 지정하세요.")
    return paths

# --------------------------
# (A) 임베딩 로더 + Z 정렬 (업데이트 단계)
# --------------------------
def load_embeddings_and_index(cid: int):
    base = cdir(cid)
    Xi = Xt = rows = None
    p_img = base/"repr_img.npy"
    p_txt = base/"repr_txt.npy"
    p_idx = base/"index.csv"
    if p_img.exists(): Xi = np.load(p_img)
    if p_txt.exists(): Xt = np.load(p_txt)
    if p_idx.exists():
        import pandas as pd
        rows = pd.read_csv(p_idx).to_dict("records")
    return Xi, Xt, rows

def build_embed_dataset(cid: int, label_csv: str):
    Xi, Xt, rows = load_embeddings_and_index(cid)
    if rows is None or (Xi is None and Xt is None):
        raise FileNotFoundError(f"[client_{cid:02d}] 임베딩/인덱스 누락")
    i_ptr = 0; t_ptr = 0
    Ximg_list, Xtxt_list, Y_list, mask_img, mask_txt = [], [], [], [], []
    table = load_label_table(label_csv)
    for r in rows:
        # index.csv에는 "p123...","s456..." 문자열이 들어있다고 가정
        sid = int(str(r["subject_id"])[1:]); stid = int(str(r["study_id"])[1:])
        y = table.get((sid, stid), [0]*len(LABEL_COLUMNS))
        has_img = int(r.get("has_img", 0)) == 1 and Xi is not None
        has_txt = int(r.get("has_txt", 0)) == 1 and Xt is not None
        if has_img and i_ptr < (0 if Xi is None else Xi.shape[0]):
            Ximg_list.append(Xi[i_ptr]); i_ptr += 1; mask_img.append(1)
        else:
            Ximg_list.append(np.zeros((0,), dtype=np.float32)); mask_img.append(0)
        if has_txt and t_ptr < (0 if Xt is None else Xt.shape[0]):
            Xtxt_list.append(Xt[t_ptr]); t_ptr += 1; mask_txt.append(1)
        else:
            Xtxt_list.append(np.zeros((0,), dtype=np.float32)); mask_txt.append(0)
        Y_list.append(y)
    N = len(rows)
    Di = 0 if Xi is None else Xi.shape[1]
    Dt = 0 if Xt is None else Xt.shape[1]
    def pad_stack(lst, D):
        if D == 0: return torch.zeros((N, 0), device=DEVICE)
        X = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            v = lst[i]
            if v.shape == (0,): continue
            X[i] = v
        return torch.tensor(X, device=DEVICE)
    Ximg = pad_stack(Ximg_list, Di)
    Xtxt = pad_stack(Xtxt_list, Dt)
    Y = torch.tensor(np.array(Y_list, dtype=np.float32), device=DEVICE)
    m_img = torch.tensor(np.array(mask_img, dtype=np.float32), device=DEVICE)
    m_txt = torch.tensor(np.array(mask_txt, dtype=np.float32), device=DEVICE)
    return Ximg, Xtxt, Y, m_img, m_txt

def load_Z_for_client(cid: int) -> Optional[torch.Tensor]:
    p = cdir(cid) / "global_payload.json"
    if not p.exists(): return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    z_path = payload.get("Z_path") or payload.get("Z_proxy_text_path") or payload.get("Z_proxy_image_path")
    if not z_path or not os.path.exists(z_path): return None
    Z = torch.from_numpy(np.load(z_path)).float().to(DEVICE)
    Z = torch.nn.functional.normalize(Z, dim=1)
    return Z

def _pad_to_dim(X: torch.Tensor, D: int) -> torch.Tensor:
    Dx = X.shape[1]
    if Dx == D: return X
    pad = torch.zeros((X.size(0), D - Dx), device=X.device, dtype=X.dtype)
    return torch.cat([X, pad], dim=1)

def _fused_feature(Fi, Ft, mi, mt):
    if Fi is None and Ft is None: return None
    if Fi is None: return Ft
    if Ft is None: return Fi
    D = max(Fi.shape[1], Ft.shape[1])
    Fi_p = torch.nn.functional.normalize(_pad_to_dim(Fi, D), dim=1)
    Ft_p = torch.nn.functional.normalize(_pad_to_dim(Ft, D), dim=1)
    denom = (mi + mt).clamp(min=1.0)
    return (Fi_p * mi + Ft_p * mt) / denom

def alignment_loss_with_gating(F, Z, base=0.1, tau=0.1):
    if (Z is None) or (F is None) or (F.numel() == 0):
        dev = F.device if isinstance(F, torch.Tensor) else ("cpu")
        return torch.tensor(0.0, device=dev)
    D = max(F.shape[1], Z.shape[1])
    Fp = torch.nn.functional.normalize(_pad_to_dim(F, D), dim=1)
    Zp = torch.nn.functional.normalize(_pad_to_dim(Z, D), dim=1)
    sim = Fp @ Zp.t()
    sim_max, _ = sim.max(dim=1)
    gate = torch.sigmoid((1.0 - sim_max) / max(tau, 1e-6))
    return base * ((1.0 - sim_max) * gate).mean()

class LinearHead(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)
    def forward(self, X): return self.fc(X)

def make_heads(Di, Dt, C, init_from=None):
    W_img0, b_img0, W_txt0, b_txt0 = init_from or (None,None,None,None)
    head_img = LinearHead(Di, C).to(DEVICE) if Di>0 else None
    head_txt = LinearHead(Dt, C).to(DEVICE) if Dt>0 else None
    with torch.no_grad():
        if head_img and W_img0 is not None and b_img0 is not None and list(head_img.fc.weight.shape)==list(W_img0.shape):
            head_img.fc.weight.copy_(W_img0.to(DEVICE)); head_img.fc.bias.copy_(b_img0.to(DEVICE))
        if head_txt and W_txt0 is not None and b_txt0 is not None and list(head_txt.fc.weight.shape)==list(W_txt0.shape):
            head_txt.fc.weight.copy_(W_txt0.to(DEVICE)); head_txt.fc.bias.copy_(b_txt0.to(DEVICE))
    return head_img, head_txt

def forward_heads(Xi, Xt, mi, mt, head_img, head_txt):
    N = Xi.shape[0] if Xi is not None else Xt.shape[0]
    C = len(LABEL_COLUMNS)
    logits = torch.zeros((N, C), device=DEVICE)
    li = head_img(Xi) if head_img is not None else None
    lt = head_txt(Xt) if head_txt is not None else None
    mi = mi.unsqueeze(1) if mi is not None else torch.zeros((N,1), device=DEVICE)
    mt = mt.unsqueeze(1) if mt is not None else torch.zeros((N,1), device=DEVICE)
    if li is not None: logits += li * mi
    if lt is not None: logits += lt * mt
    return logits / (mi+mt).clamp(min=1.0)

def evaluate_heads(Xi, Xt, Y, mi, mt, ids, head_img, head_txt):
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = forward_heads(Xi[ids], Xt[ids], mi[ids], mt[ids], head_img, head_txt)
        y = Y[ids]
        loss = crit(logits, y).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y.cpu().numpy().astype(np.int32)
        y_pred = (probs>=0.5).astype(np.int32)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:,c]
            if len(np.unique(col))<2: continue
            try: aucs.append(roc_auc_score(col, probs[:,c]))
            except: pass
        auc_macro = float(np.mean(aucs)) if aucs else float("nan")
        return {"loss": loss, "f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

def update_heads_with_Z(cid, label_csv, epochs, lr, z_weight, tau):
    Xi, Xt, Y, mi, mt = build_embed_dataset(cid, label_csv)
    Di = Xi.shape[1]; Dt = Xt.shape[1]

    # best.pt의 헤드 가져와 초기화
    W_img0 = b_img0 = W_txt0 = b_txt0 = None
    ckpt_path = cdir(cid)/"best.pt"
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        if "img.head.weight" in sd and Di>0 and sd["img.head.weight"].shape[1]==Di:
            W_img0, b_img0 = sd["img.head.weight"], sd["img.head.bias"]
        elif "head.weight" in sd and Di>0 and sd["head.weight"].shape[1]==Di:
            W_img0, b_img0 = sd["head.weight"], sd["head.bias"]
        if "txt.cls_head.weight" in sd and Dt>0 and sd["txt.cls_head.weight"].shape[1]==Dt:
            W_txt0, b_txt0 = sd["txt.cls_head.weight"], sd["txt.cls_head.bias"]
        elif "cls_head.weight" in sd and Dt>0 and sd["cls_head.weight"].shape[1]==Dt:
            W_txt0, b_txt0 = sd["cls_head.weight"], sd["cls_head.bias"]

    head_img, head_txt = make_heads(Di, Dt, len(LABEL_COLUMNS), (W_img0, b_img0, W_txt0, b_txt0))

    # 분할
    N = Y.shape[0]
    idx = np.arange(N); rng = np.random.RandomState(SEED); rng.shuffle(idx)
    va_idx = torch.tensor(idx[:max(1, int(0.1*N))], device=DEVICE)
    tr_idx = torch.tensor(idx[max(1, int(0.1*N)):], device=DEVICE)

    params = []
    if head_img: params += list(head_img.parameters())
    if head_txt: params += list(head_txt.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()
    Z = load_Z_for_client(cid)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        y = Y[tr_idx]
        logits = forward_heads(Xi[tr_idx], Xt[tr_idx], mi[tr_idx], mt[tr_idx], head_img, head_txt)
        loss = crit(logits, y)
        Fi = Xi[tr_idx] if Di>0 else None
        Ft = Xt[tr_idx] if Dt>0 else None
        gate_F = _fused_feature(Fi, Ft, mi[tr_idx].unsqueeze(1), mt[tr_idx].unsqueeze(1))
        if Z is not None and gate_F is not None:
            loss = loss + alignment_loss_with_gating(gate_F, Z, base=z_weight, tau=tau)
        loss.backward(); opt.step()
        va = evaluate_heads(Xi, Xt, Y, mi, mt, va_idx, head_img, head_txt)
        print(f"[client {cid:02d}] ep {ep}/{epochs} | train_loss={loss.item():.4f} | val {va}")

    # 저장
    out = {}
    if head_img:
        out["W_img"] = head_img.fc.weight.detach().cpu().numpy()
        out["b_img"] = head_img.fc.bias.detach().cpu().numpy()
    if head_txt:
        out["W_txt"] = head_txt.fc.weight.detach().cpu().numpy()
        out["b_txt"] = head_txt.fc.bias.detach().cpu().numpy()
    if out:
        np.savez(cdir(cid)/"updated_heads.npz", **out)
        print(f"[client {cid:02d}] updated heads saved → {cdir(cid)/'updated_heads.npz'}")

# --------------------------
# (B) 원본 테스트셋(디렉터리) 로더 + 모델 로딩/평가
# --------------------------
class TestDirsDataset(Dataset):
    def __init__(self, study_dirs: List[str], label_table: Dict[Tuple[int,int], List[int]]):
        self.items = []
        for d in study_dirs:
            try:
                sid, stid = id_to_int_from_path(d)
            except Exception:
                continue
            y = label_table.get((sid, stid), [0]*len(LABEL_COLUMNS))
            self.items.append({"dir": d, "sid": sid, "stid": stid, "y": y})
        if not self.items:
            raise RuntimeError("유효한 스터디가 없습니다(라벨 매칭 실패 가능).")
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self): return len(self.items)

    def _load_image(self, d: str):
        imgs = sorted([*Path(d).glob("*.jpg"), *Path(d).glob("*.jpeg"), *Path(d).glob("*.png")])
        if not imgs: return torch.zeros(3,224,224)
        try: return IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB"))
        except: return torch.zeros(3,224,224)

    def _find_text_path(self, d: str) -> Optional[Path]:
        # 같은 폴더/상위 폴더에 보고서 .txt가 있으면 사용. 기본적으로 없다고 가정.
        cand = list(Path(d).glob("*.txt"))
        if cand: return cand[0]
        return None

    def _load_text(self, p: Optional[Path]):
        if p is None:
            txt = ""
        else:
            try:
                with open(p, "r", encoding="utf-8") as f: txt = f.read()
            except:
                txt = ""
        enc = self.tok(txt, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

    def __getitem__(self, idx):
        it = self.items[idx]
        y = torch.tensor(np.array(it["y"], dtype=np.float32))
        img = self._load_image(it["dir"])
        txt_path = self._find_text_path(it["dir"])
        txt = self._load_text(txt_path)
        return {"labels": y, "image": img, "text": txt,
                "has_img": 1.0 if img.numel()>0 else 0.0,
                "has_txt": 0.0 if txt_path is None else 1.0}

class ImageHead(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_f, n_out)
    def forward(self, x): return self.head(self.backbone(x))

class TextHead(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, n_out)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:,0]
        return self.cls_head(pooled)

class MultiModalLateFusion(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = ImageHead(n_out)
        self.txt = TextHead(n_out)

def load_ckpt_as_model(ckpt_path: str, n_out: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = ckpt.get("mode", "multimodal")
    if mode == "multimodal":
        model = MultiModalLateFusion(n_out)
    elif mode == "image_only":
        model = ImageHead(n_out)
    elif mode == "text_only":
        model = TextHead(n_out)
    else:
        model = MultiModalLateFusion(n_out)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE), mode

def apply_updated_heads_if_any(model, client_dir: Path):
    npz = client_dir/"updated_heads.npz"
    if not npz.exists(): return False
    data = np.load(npz, allow_pickle=False)
    applied = False
    def try_copy(linear: nn.Linear, W_key, b_key):
        nonlocal applied
        if W_key in data and b_key in data:
            W = torch.from_numpy(data[W_key]).float()
            B = torch.from_numpy(data[b_key]).float()
            if list(W.shape)==list(linear.weight.shape) and list(B.shape)==list(linear.bias.shape):
                with torch.no_grad():
                    linear.weight.copy_(W.to(linear.weight.device))
                    linear.bias.copy_(B.to(linear.bias.device))
                applied = True
    if hasattr(model, "img") and hasattr(model.img, "head"):
        try_copy(model.img.head, "W_img", "b_img")
    if hasattr(model, "txt") and hasattr(model.txt, "cls_head"):
        try_copy(model.txt.cls_head, "W_txt", "b_txt")
    if hasattr(model, "head"):      # image-only
        try_copy(model.head, "W_img", "b_img")
    if hasattr(model, "cls_head"):  # text-only
        try_copy(model.cls_head, "W_txt", "b_txt")
    return applied

@torch.no_grad()
def evaluate_model_mmasked(model, loader):
    """
    멀티모달 모델도 per-sample 모달 마스크(has_img/has_txt) 기반 late fusion:
      logits = (li*mi + lt*mt) / clamp(mi+mt, 1)
    텍스트가 없으면 이미지만 쓰고, 이미지가 없으면 텍스트만 씀.
    """
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot = 0.0; y_true=[]; y_prob=[]

    for batch in loader:
        y = batch["labels"].to(DEVICE)

        if isinstance(model, MultiModalLateFusion):
            mi = batch["has_img"].to(DEVICE).unsqueeze(1)  # (B,1)
            mt = batch["has_txt"].to(DEVICE).unsqueeze(1)

            li = model.img(batch["image"].to(DEVICE))
            lt = model.txt(**{k:v.to(DEVICE) for k,v in batch["text"].items()})
            logits = (li*mi + lt*mt) / (mi+mt).clamp(min=1.0)

        elif isinstance(model, ImageHead):
            logits = model(batch["image"].to(DEVICE))

        else:  # TextHead
            logits = model(**{k:v.to(DEVICE) for k,v in batch["text"].items()})

        loss = crit(logits, y)
        tot += loss.item()*y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)
    f1_micro = f1_score(y_true, (y_prob>=0.5).astype(int), average="micro", zero_division=0)
    f1_macro = f1_score(y_true, (y_prob>=0.5).astype(int), average="macro", zero_division=0)
    aucs=[]
    for j in range(y_true.shape[1]):
        col = y_true[:,j]
        if len(np.unique(col))<2: aucs.append(float("nan"))
        else:
            try: aucs.append(roc_auc_score(col, y_prob[:,j]))
            except: aucs.append(float("nan"))
    macro_auc = float(np.nanmean(aucs))
    return tot/len(loader.dataset), f1_micro, f1_macro, macro_auc, aucs

# --------------------------
# 메인
# --------------------------
def parse_clients(s: str):
    s=s.strip().lower()
    if s in ("all",""): return list(range(1,21))
    out=[]
    for part in s.split(","):
        part=part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            out.extend(list(range(int(a), int(b)+1)))
        else:
            out.append(int(part))
    return sorted(list(dict.fromkeys(out)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20")
    ap.add_argument("--label_csv", type=str, required=True)
    # 테스트 세트 지정(세 가지 방식 중 하나 이상)
    ap.add_argument("--test_dirs", type=str, default="")
    ap.add_argument("--test_list", type=str, default="")
    ap.add_argument("--test_root", type=str, default="")
    ap.add_argument("--batch", type=int, default=32)
    # 업데이트 하이퍼파라미터
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--z_weight", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    # (A) 업데이트(임베딩 + Z)
    clients = parse_clients(args.clients)
    for cid in clients:
        try:
            print(f"\n==== [A] Update heads from embeddings with Z | client {cid:02d} ====")
            update_heads_with_Z(cid, args.label_csv, args.epochs, args.lr, args.z_weight, args.tau)
        except Exception as e:
            print(f"[WARN] client {cid:02d} 업데이트 건너뜀: {e}")

    # (B) 테스트셋(원본 파일) 로딩
    study_dirs = parse_test_dirs(args)
    label_table = load_label_table(args.label_csv)
    ds = TestDirsDataset(study_dirs, label_table)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    n_out = len(LABEL_COLUMNS)

    rows=[]
    for cid in clients:
        ckpt_path = cdir(cid)/"best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (no best.pt)")
            continue

        # baseline
        base_model, mode = load_ckpt_as_model(str(ckpt_path), n_out)
        base_loss, base_f1_micro, base_f1_macro, base_macro_auc, base_aucs = evaluate_model_mmasked(base_model, dl)
        print(f"[client {cid:02d} | {mode:11s}] BASE  loss={base_loss:.4f} f1_micro={base_f1_micro:.4f} f1_macro={base_f1_macro:.4f} macro_auc={base_macro_auc:.4f}")

        # updated
        upd_model, _ = load_ckpt_as_model(str(ckpt_path), n_out)
        applied = apply_updated_heads_if_any(upd_model, cdir(cid))
        if applied:
            upd_loss, upd_f1_micro, upd_f1_macro, upd_macro_auc, upd_aucs = evaluate_model_mmasked(upd_model, dl)
            print(f"[client {cid:02d}]      UPDT loss={upd_loss:.4f} f1_micro={upd_f1_micro:.4f} f1_macro={upd_f1_macro:.4f} macro_auc={upd_macro_auc:.4f}")
        else:
            upd_loss=upd_f1_micro=upd_f1_macro=upd_macro_auc=float("nan"); upd_aucs=[float("nan")]*len(LABEL_COLUMNS)
            print(f"[client {cid:02d}]      UPDT (updated_heads.npz 없음/shape 불일치)")

        # 저장
        outj = {
            "client_id": cid, "mode": mode,
            "baseline": {
                "loss": base_loss, "f1_micro": base_f1_micro, "f1_macro": base_f1_macro,
                "macro_auroc": base_macro_auc,
                "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, base_aucs)}
            },
            "updated": {
                "loss": (None if (upd_loss!=upd_loss) else float(upd_loss)),
                "f1_micro": (None if (upd_f1_micro!=upd_f1_micro) else float(upd_f1_micro)),
                "f1_macro": (None if (upd_f1_macro!=upd_f1_macro) else float(upd_f1_macro)),
                "macro_auroc": (None if (upd_macro_auc!=upd_macro_auc) else float(upd_macro_auc)),
                "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, upd_aucs)}
            }
        }
        with open(cdir(cid)/"compare_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(outj, f, indent=2, ensure_ascii=False)

        rows.append({
            "client_id": cid, "mode": mode,
            "baseline_loss": base_loss, "baseline_f1_micro": base_f1_micro, "baseline_f1_macro": base_f1_macro, "baseline_macro_auroc": base_macro_auc,
            "updated_loss": upd_loss,   "updated_f1_micro": (None if (upd_f1_micro!=upd_f1_micro) else upd_f1_micro),
            "updated_f1_macro": (None if (upd_f1_macro!=upd_f1_macro) else upd_f1_macro),
            "updated_macro_auroc": (None if (upd_macro_auc!=upd_macro_auc) else upd_macro_auc),
        })

    if rows:
        headers = ["client_id","mode",
                   "baseline_loss","baseline_f1_micro","baseline_f1_macro","baseline_macro_auroc",
                   "updated_loss","updated_f1_micro","updated_f1_macro","updated_macro_auroc"]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
    else:
        print("[WARN] no evaluated clients; check checkpoints/paths.")

if __name__ == "__main__":
    main()