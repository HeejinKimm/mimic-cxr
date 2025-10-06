# convert_pt_to_safetensors.py
import argparse, sys
from pathlib import Path
from typing import Optional
from collections import OrderedDict

def extract_state_dict(obj, prefer_key: Optional[str] = None):
    """
    다양한 .pt 포맷에서 state_dict를 뽑아낸다.
    우선순위:
      1) prefer_key가 dict에 있고 dict/OrderedDict라면
      2) 'model' 키
      3) 'state_dict' 키
      4) dict 자체가 텐서 dict
      5) 객체가 state_dict()를 제공
    실패 시 None
    """
    try:
        import torch
    except Exception:
        print("[error] torch 가 필요합니다.")
        return None

    sd = None
    if isinstance(obj, (dict, OrderedDict)):
        d = obj
        if prefer_key and prefer_key in d and isinstance(d[prefer_key], (dict, OrderedDict)):
            sd = d[prefer_key]
        elif "model" in d and isinstance(d["model"], (dict, OrderedDict)):
            sd = d["model"]
        elif "state_dict" in d and isinstance(d["state_dict"], (dict, OrderedDict)):
            sd = d["state_dict"]
        elif all(hasattr(v, "dtype") for v in d.values() if v is not None):
            sd = d
    elif hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            sd = obj.state_dict()
        except Exception:
            sd = None

    if sd is None:
        return None

    # DataParallel 등 'module.' 프리픽스 제거
    cleaned = OrderedDict()
    for k, v in sd.items():
        if hasattr(k, "startswith") and k.startswith("module."):
            nk = k[len("module."):]
        else:
            nk = k
        cleaned[nk] = v
    return cleaned

def main():
    ap = argparse.ArgumentParser(description="Convert PyTorch .pt checkpoints to .safetensors")
    ap.add_argument("--root", type=str, default=".", help=r"검색 루트 (기본: 현재 디렉토리)")
    ap.add_argument("--pattern", type=str, default="outputs/client_*/*.pt",
                    help=r"체크포인트 검색 패턴 (기본: outputs/client_*/*.pt)")
    ap.add_argument("--key", type=str, default=None,
                    help="체크포인트 dict 안에서 우선적으로 찾을 키 이름 (예: model)")
    ap.add_argument("--dry", action="store_true", help="변환하지 않고 목록만 출력")
    ap.add_argument("--force", action="store_true", help="이미 존재하는 .safetensors도 덮어쓰기")
    ap.add_argument("--max-preview", type=int, default=10, help="미리보기로 출력할 파일 개수 (기본 10)")
    args = ap.parse_args()

    root = Path(args.root)
    files = list(root.rglob(args.pattern))

    print(f"[info] root={root.resolve()}")
    print(f"[info] pattern='{args.pattern}' → found {len(files)} .pt file(s)")
    for i, p in enumerate(files[:max(0, args.max_preview)], 1):
        print(f"  {i:02d}) {p}")

    if args.dry or not files:
        return

    # libs
    try:
        import torch
    except Exception as e:
        print("[error] torch import 실패:", e); sys.exit(1)
    try:
        from safetensors.torch import save_file as safe_save
    except Exception as e:
        print("[error] safetensors 미설치: pip install safetensors"); sys.exit(1)

    ok, skip, fail = 0, 0, 0
    for pt in files:
        out = pt.with_suffix(".safetensors")
        if out.exists() and not args.force:
            print(f"[skip] exists → {out}")
            skip += 1
            continue

        try:
            # 안전을 위해 CPU로 로드 (pickle 주의: 신뢰된 파일만 로드하세요)
            ckpt = torch.load(str(pt), map_location="cpu")
        except TypeError:
            # 일부 환경에서 weights_only가 필요한 경우도 있으나, 광범위 호환을 위해 일반 load 우선
            try:
                ckpt = torch.load(str(pt), map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"[fail] load: {pt} → {e}")
                fail += 1
                continue
        except Exception as e:
            print(f"[fail] load: {pt} → {e}")
            fail += 1
            continue

        sd = extract_state_dict(ckpt, prefer_key=args.key)
        if sd is None:
            print(f"[fail] state_dict 추출 불가: {pt}  (키가 특이하면 --key 사용)")
            fail += 1
            continue

        try:
            # 텐서를 CPU로 강제 이동 (safetensors는 device-agnostic 저장)
            for k, v in list(sd.items()):
                try:
                    sd[k] = v.detach().cpu()
                except Exception:
                    pass

            safe_save(sd, str(out))
            print(f"[ok] {pt.name}  →  {out.name}")
            ok += 1
        except Exception as e:
            print(f"[fail] save: {pt} → {e}")
            fail += 1

    print(f"\n[summary] ok={ok}, skip={skip}, fail={fail}")

if __name__ == "__main__":
    main()
