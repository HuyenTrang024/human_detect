"""
evaluate_ddnet.py - Đánh giá một checkpoint DD-Net theo từng VIDEO.

Tách riêng khỏi train_ddnet.py có chủ ý: trong lúc chỉnh tham số chỉ nhìn kết quả VAL
(train_ddnet.py tự in). Chỉ chạy file này trên TEST khi đã chốt cấu hình, một lần, để con số
test còn trung thực và so được với bài báo RoCoG-v2.

Cách tính nhãn cho một video: cắt nhiều cửa sổ T frame (trượt đều), cho model dự đoán từng
cửa sổ, lấy TRUNG BÌNH xác suất -> lớp có xác suất cao nhất.

In ra:
  - độ chính xác theo video và theo cửa sổ, macro-F1
  - ma trận nhầm lẫn 7x7 và độ chính xác từng lớp
  - độ chính xác theo từng người (S00, S04, S04b, S06, S09)
  - các cặp lớp hay nhầm nhất
Ghi ra cạnh checkpoint: results_<split>.json, predictions_<split>.csv, confusion_<split>.png

Ví dụ:
    python evaluate_ddnet.py --ckpt checkpoints/scratch_f16/best.pt --split test
    python evaluate_ddnet.py --ckpt checkpoints/scratch_f16/best.pt --split val
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from ddnet_model import DDNet, load_checkpoint
from gesture_dataset import Clip, EvalWindowDataset, load_split_clips


# ---------------------------------------------------------------------------
# Dự đoán
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_clips(model: DDNet, clips: List[Clip], T: int, device: torch.device,
                  stride: int = config.EVAL_STRIDE, trim_sec: float = config.TRIM_SEC,
                  window_mode: str = "sliding", batch_size: int = 256,
                  class_weight: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
    """
    Trả dict:
      clip_probs  (n_clip, C)  xác suất trung bình các cửa sổ của mỗi clip
      clip_labels (n_clip,)
      win_correct (n_win,)     cửa sổ đoán đúng hay không
      loss        float        cross-entropy trung bình trên các cửa sổ
    """
    ds = EvalWindowDataset(clips, T, stride, trim_sec, window_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    C = model.hparams["num_classes"]
    sums = np.zeros((len(clips), C), np.float64)
    counts = np.zeros(len(clips), np.int64)
    win_correct, losses, n = [], 0.0, 0
    was_training = model.training
    model.eval()
    w = class_weight.to(device) if class_weight is not None else None
    for x, y, ci in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses += F.cross_entropy(logits, y, weight=w, reduction="sum").item()
        n += len(y)
        prob = logits.softmax(-1).cpu().numpy()
        win_correct.append((prob.argmax(1) == y.cpu().numpy()))
        ci = ci.numpy()
        np.add.at(sums, ci, prob)
        np.add.at(counts, ci, 1)
    model.train(was_training)
    return dict(
        clip_probs=sums / np.maximum(counts, 1)[:, None],
        clip_labels=np.array([c.label for c in clips]),
        win_correct=np.concatenate(win_correct) if win_correct else np.zeros(0, bool),
        loss=losses / max(n, 1),
    )


# ---------------------------------------------------------------------------
# Chỉ số
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> Dict:
    cm = confusion_matrix(y_true, y_pred, C)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(C), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(C), where=(precision + recall) > 0)
    present = support > 0
    return dict(
        accuracy=float(tp.sum() / max(cm.sum(), 1)),
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        recall=recall.tolist(), precision=precision.tolist(), f1=f1.tolist(),
        support=support.astype(int).tolist(), confusion=cm.tolist(),
    )


def format_confusion(cm: np.ndarray, names: Sequence[str]) -> str:
    short = [n[:8] for n in names]
    width = max(9, max(len(s) for s in short) + 1)
    lines = [" " * 15 + "".join(f"{s:>{width}}" for s in short) + "   <- dự đoán"]
    for i, n in enumerate(names):
        row = "".join(f"{v:>{width}}" for v in cm[i])
        lines.append(f"{n[:14]:<15}{row}")
    lines.append("(hàng = nhãn thật)")
    return "\n".join(lines)


def top_confusions(cm: np.ndarray, names: Sequence[str], k: int = 5) -> List[str]:
    pairs = [(cm[i, j], i, j) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    return [f"{names[i]} -> {names[j]}: {v} video" for v, i, j in pairs[:k]]


def per_subject_accuracy(clips: List[Clip], y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    # S04 và S04b là cùng một người (hai lượt quay) -> gộp như rocog_labels.subject_group
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for c, p in zip(clips, y_pred):
        stats[c.subject_group][0] += int(p == c.label)
        stats[c.subject_group][1] += 1
    return {s: {"accuracy": v[0] / v[1], "n": v[1]} for s, v in sorted(stats.items())}


def evaluate(model: DDNet, clips: List[Clip], class_names: Sequence[str], T: int,
             device: torch.device, **kw) -> Dict:
    pred = predict_clips(model, clips, T, device, **kw)
    y_true = pred["clip_labels"]
    y_pred = pred["clip_probs"].argmax(1)
    m = classification_metrics(y_true, y_pred, len(class_names))
    m.update(
        window_accuracy=float(pred["win_correct"].mean()) if len(pred["win_correct"]) else 0.0,
        loss=pred["loss"], n_clips=len(clips), n_windows=int(len(pred["win_correct"])),
        per_subject=per_subject_accuracy(clips, y_pred),
    )
    m["_y_pred"] = y_pred
    m["_probs"] = pred["clip_probs"]
    return m


def print_report(m: Dict, class_names: Sequence[str], title: str = "") -> None:
    if title:
        print(f"\n========== {title} ==========")
    print(f"Độ chính xác theo video : {m['accuracy'] * 100:.1f}%  ({m['n_clips']} video)")
    print(f"Độ chính xác theo cửa sổ: {m['window_accuracy'] * 100:.1f}%  ({m['n_windows']} cửa sổ)")
    print(f"Macro-F1                : {m['macro_f1'] * 100:.1f}%")
    print(f"Loss                    : {m['loss']:.4f}\n")
    print(format_confusion(np.array(m["confusion"]), class_names))
    print(f"\n{'Lớp':<15}{'recall':>8}{'precision':>11}{'F1':>7}{'số video':>10}")
    for i, n in enumerate(class_names):
        print(f"{n:<15}{m['recall'][i] * 100:>7.1f}%{m['precision'][i] * 100:>10.1f}%"
              f"{m['f1'][i] * 100:>6.1f}%{m['support'][i]:>10}")
    print("\nTheo người:")
    for s, v in m["per_subject"].items():
        print(f"  {s:<8}{v['accuracy'] * 100:>6.1f}%  ({v['n']} video)")
    tc = top_confusions(np.array(m["confusion"]), class_names)
    if tc:
        print("\nHay nhầm nhất:")
        for line in tc:
            print("  " + line)


def save_confusion_png(cm: np.ndarray, names: Sequence[str], path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    row = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Đánh giá DD-Net theo video")
    ap.add_argument("--ckpt", required=True, help="file .pt do train_ddnet.py lưu")
    ap.add_argument("--split", default="test", help="test / val / train hoặc tên file chia")
    ap.add_argument("--ann-dir", default=str(config.ANNOTATION_DIR))
    ap.add_argument("--kp-dir", default=str(config.KEYPOINT_DIR))
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, meta = load_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()
    class_names = meta.get("class_names")
    T = model.hparams["T"]
    fps = meta.get("fps", config.FPS_TARGET)
    window_mode = meta.get("window_mode", "sliding")
    stride = args.stride or meta.get("eval_stride", config.EVAL_STRIDE)
    trim = meta.get("trim_sec", config.TRIM_SEC)
    min_ok = meta.get("min_ok_ratio", config.MIN_OK_RATIO)
    print(f"Checkpoint: {args.ckpt}")
    print(f"  T={T}, fps={fps}, cửa sổ={window_mode}, lớp={class_names}")
    print(f"  pose={meta.get('pose_model')}, train ở epoch {meta.get('epoch')}, "
          f"val acc lúc lưu={meta.get('val_accuracy')}")

    clips = load_split_clips([args.split], args.ann_dir, args.kp_dir, fps, min_ok)
    if not clips:
        raise SystemExit("Không có clip nào để đánh giá.")
    m = evaluate(model, clips, class_names, T, device, stride=stride, trim_sec=trim,
                 window_mode=window_mode)
    print_report(m, class_names, title=f"{args.split.upper()} - {Path(args.ckpt).parent.name}")

    out_dir = Path(args.ckpt).parent
    y_pred, probs = m.pop("_y_pred"), m.pop("_probs")
    with open(out_dir / f"results_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"predictions_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "subject", "label", "pred", "correct"] + [f"p_{n}" for n in class_names])
        for c, p, pr in zip(clips, y_pred, probs):
            w.writerow([c.rel_path, c.subject, class_names[c.label], class_names[p], int(p == c.label)]
                       + [f"{v:.4f}" for v in pr])
    png = out_dir / f"confusion_{args.split}.png"
    saved = save_confusion_png(np.array(m["confusion"]), class_names, png,
                               f"{args.split} - acc {m['accuracy'] * 100:.1f}%")
    print(f"\nĐã lưu kết quả vào {out_dir}" + (f" (kèm {png.name})" if saved else ""))


if __name__ == "__main__":
    main()