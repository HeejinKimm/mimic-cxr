# evaluate_on_test.py
import os, torch, numpy as np, pandas as pd
from pathlib import Path
from local_train.config import Cfg
from local_train.local_update import (
    load_label_table, idstr_to_ints, _pad_to_dim, _fused_feature, evaluate_heads, LinearHead
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLUMNS = Cfg.LABEL_COLUMNS

def load_updated_heads(client_dir: Path):
    p = client_dir / "updated_heads.npz"
    if not p.exists():
        raise FileNotFoundError(f"updated_heads.npz 없음: {p}")
    data = np.load(p)
    head_img = LinearHead(data["W_img"].shape[1], len(LABEL_COLUMNS)).to(DEVICE)
    head_txt = LinearHead(data["W_txt"].shape[1], len(LABEL_COLUMNS)).to(DEVICE)
    head_img.fc.weight.data = torch.tensor(data["W_img"]).to(DEVICE)
    head_img.fc.bias.data = torch.tensor(data["b_img"]).to(DEVICE)
    head_txt.fc.weight.data = torch.tensor(data["W_txt"]).to(DEVICE)
    head_txt.fc.bias.data = torch.tensor(data["b_txt"]).to(DEVICE)
    return head_img, head_txt

def build_test_dataset(test_csv: str, label_csv: str):
    df = pd.read_csv(test_csv)
    label_table = load_label_table(label_csv)
    Xi, Xt, Y, m_img, m_txt = [], [], [], [], []

    for _, r in df.iterrows():
        sid, stid = idstr_to_ints(r["subject_id"], r["study_id"])
        y = label_table.get((sid, stid), [0]*len(LABEL_COLUMNS))
        Xi.append(np.load(r["image_dir"] + "_embed.npy"))   # 💡 임베딩 전처리 필요
        Xt.append(np.load(r["text_path"] + "_embed.npy"))
        m_img.append(1.0)
        m_txt.append(1.0)
        Y.append(y)

    return (torch.tensor(np.array(Xi), device=DEVICE, dtype=torch.float32),
            torch.tensor(np.array(Xt), device=DEVICE, dtype=torch.float32),
            torch.tensor(np.array(Y), device=DEVICE, dtype=torch.float32),
            torch.tensor(np.array(m_img), device=DEVICE),
            torch.tensor(np.array(m_txt), device=DEVICE))

def main():
    cid = 1  # 예시: client_01 head로 평가
    client_dir = Path(Cfg.BASE) / "local_train_outputs" / f"client_{cid:02d}"
    head_img, head_txt = load_updated_heads(client_dir)

    test_csv = Cfg.CLIENT_CSV_DIR / "test.csv"
    label_csv = str(Cfg.LABEL_CSV_NEGBIO)
    Ximg, Xtxt, Y, m_img, m_txt = build_test_dataset(test_csv, label_csv)

    metrics = evaluate_heads(Ximg, Xtxt, Y, m_img, m_txt, torch.arange(len(Y)), head_img, head_txt)
    print(f"[Client {cid:02d} on TEST]", metrics)

if __name__ == "__main__":
    main()
