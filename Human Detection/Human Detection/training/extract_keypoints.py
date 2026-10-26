
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from gesture.skeleton_adapter import SkeletonAdapter

import numpy as np

from gesture import config
from .rocog_labels import (VideoEntry, read_splits, resolve_video_path,
                          sample_balanced)

NUM_MP_POINTS = 33


# ---------------------------------------------------------------------------
# Làm việc với các module sẵn có của bạn (pose_estimate.py, skeleton_adapter.py)
# ---------------------------------------------------------------------------
def adapter_version() -> str:
    """Mã băm nội dung skeleton_adapter.py -> phát hiện dữ liệu trích bằng adapter cũ."""
    from gesture import skeleton_adapter
    src = Path(skeleton_adapter.__file__).read_bytes()
    return hashlib.md5(src).hexdigest()[:10]


def make_pose_estimator():
    """
    Tạo PoseEstimator MỚI (gọi lại cho mỗi video). MediaPipe ở chế độ VIDEO nhớ trạng
    thái bám người và đòi timestamp tăng dần, nên dùng lại đối tượng giữa hai video sẽ
    lỗi hoặc làm khung xương đầu video mới bị kéo theo video trước.
    """
    from gesture.pose_estimate import PoseEstimator
    return PoseEstimator()


def close_pose_estimator(est) -> None:
    for name in ("close", "release"):
        fn = getattr(est, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


_POSE_ACCEPTS_TS: Optional[str] = None


def call_pose(est, image_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
    """
    Gọi est.get_keypoints(...) và chuẩn hoá đầu ra về mảng (33, 3) [x, y, visibility]
    toạ độ PIXEL trong ảnh truyền vào, hoặc None.

    Chấp nhận các kiểu get_keypoints() hay gặp:
      - get_keypoints(image) hoặc get_keypoints(image, timestamp_ms)
      - trả list 33 tuple (x, y, vis) / (x, y, z, vis), hoặc mảng numpy tương đương
      - toạ độ pixel, hoặc toạ độ chuẩn hoá 0..1 (tự nhân với kích thước ảnh)
    """
    global _POSE_ACCEPTS_TS
    if _POSE_ACCEPTS_TS is None:
        try:
            params = inspect.signature(est.get_keypoints).parameters
        except (TypeError, ValueError):
            params = {}
        if "timestamp_ms" in params:
            _POSE_ACCEPTS_TS = "kw"
        elif len(params) >= 2:
            _POSE_ACCEPTS_TS = "pos"
        else:
            _POSE_ACCEPTS_TS = "no"

    if _POSE_ACCEPTS_TS == "kw":
        out = est.get_keypoints(image_bgr, timestamp_ms=timestamp_ms)
    elif _POSE_ACCEPTS_TS == "pos":
        out = est.get_keypoints(image_bgr, timestamp_ms)
    else:
        out = est.get_keypoints(image_bgr)

    if out is None:
        return None
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != NUM_MP_POINTS:
        return None
    if arr.shape[1] >= 4:            # (x, y, z, vis) -> (x, y, vis)
        arr = arr[:, [0, 1, -1]]
    elif arr.shape[1] == 2:          # không có visibility -> coi như thấy hết
        arr = np.concatenate([arr, np.ones((NUM_MP_POINTS, 1), np.float32)], axis=1)
    arr = arr[:, :3].copy()

    h, w = image_bgr.shape[:2]
    if np.nanmax(np.abs(arr[:, :2])) <= 1.5 and max(h, w) > 4:   # toạ độ chuẩn hoá 0..1
        arr[:, 0] *= w
        arr[:, 1] *= h
    return arr


def run_adapter(adapter, kps_full: Optional[np.ndarray]):
    """Gọi SkeletonAdapter.convert() với list 33 tuple (x, y, vis) toạ độ ảnh gốc."""
    if kps_full is None:
        return None
    return adapter.convert([tuple(map(float, p)) for p in kps_full])


# ---------------------------------------------------------------------------
# Xử lý một video
# ---------------------------------------------------------------------------
_WORKER: Dict[str, object] = {}


def _init_worker(yolo_weights: Optional[str], device: Optional[str]) -> None:
    """Mỗi tiến trình tạo YOLO một lần (model không chuyển qua lại giữa tiến trình được)."""
    from gesture.person_crop import FullFrameDetector, PersonCropper, YoloPersonDetector
    detector = FullFrameDetector() if yolo_weights is None else YoloPersonDetector(yolo_weights, device=device)
    _WORKER["cropper"] = PersonCropper(detector)
    _WORKER["adapter_version"] = adapter_version()


def extract_video(video_path: Path) -> Dict[str, np.ndarray]:
    """Chạy toàn bộ pipeline trên một video, trả các mảng theo từng frame."""

    cropper = _WORKER["cropper"]
    cropper.reset()
    adapter = SkeletonAdapter()      # mới cho mỗi video (tương đương adapter.reset())
    pose = make_pose_estimator()     # mới cho mỗi video

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Không mở được video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps < 1 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_list, joints_list, vis_list, ok_list = [], [], [], []
    raw_list, bbox_list, scale_list = [], [], []
    n_person = n_pose = 0
    J = None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1
            t_list.append(t)

            crop = cropper.process(frame)
            kps_full = None
            if crop is not None and crop.crop.size > 0:
                n_person += 1
                bbox_list.append(crop.box.astype(np.float32))
                kps = call_pose(pose, crop.crop, int(round(t * 1000)))
                if kps is not None:
                    n_pose += 1
                    kps_full = kps.copy()
                    kps_full[:, 0] += crop.offset[0]
                    kps_full[:, 1] += crop.offset[1]
            else:
                bbox_list.append(np.full(4, -1, np.float32))
                # Mất người -> quên vị trí và thước đo cũ, giống lúc chạy thật
                adapter.reset()

            raw_list.append(kps_full if kps_full is not None
                            else np.full((NUM_MP_POINTS, 3), np.nan, np.float32))

            res = run_adapter(adapter, kps_full)
            if res is not None:
                if J is None:
                    J = res.joints.shape[0]
                joints_list.append(np.asarray(res.joints, np.float32))
                vis_list.append(np.asarray(res.visible, bool))
                scale_list.append(float(getattr(res, "scale", np.nan)))
                ok_list.append(True)
            else:
                joints_list.append(None)
                vis_list.append(None)
                scale_list.append(np.nan)
                ok_list.append(False)
    finally:
        cap.release()
        close_pose_estimator(pose)

    if J is None:
        J = _num_joints_fallback()
    joints = np.stack([j if j is not None else np.full((J, 2), np.nan, np.float32) for j in joints_list]) \
        if joints_list else np.zeros((0, J, 2), np.float32)
    visible = np.stack([v if v is not None else np.zeros(J, bool) for v in vis_list]) \
        if vis_list else np.zeros((0, J), bool)

    return dict(
        t=np.asarray(t_list, np.float64),
        joints=joints.astype(np.float32),
        visible=visible,
        ok=np.asarray(ok_list, bool),
        kps_raw=np.stack(raw_list).astype(np.float32) if raw_list else np.zeros((0, 33, 3), np.float32),
        bbox=np.stack(bbox_list).astype(np.float32) if bbox_list else np.zeros((0, 4), np.float32),
        scale=np.asarray(scale_list, np.float32),
        fps=np.float32(fps), width=np.int32(width), height=np.int32(height),
        n_person=np.int32(n_person), n_pose=np.int32(n_pose),
    )


def readapt(npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Tính lại joints/visible/ok/scale từ kps_raw đã lưu (sau khi sửa skeleton_adapter.py)."""
    from gesture.skeleton_adapter import SkeletonAdapter
    adapter = SkeletonAdapter()
    J = None
    joints_list, vis_list, ok_list, scale_list = [], [], [], []
    for kps, box in zip(npz["kps_raw"], npz["bbox"]):
        if box[0] < 0:
            adapter.reset()
        kps_full = None if np.isnan(kps).all() else kps
        res = run_adapter(adapter, kps_full)
        if res is not None:
            J = J or res.joints.shape[0]
            joints_list.append(np.asarray(res.joints, np.float32))
            vis_list.append(np.asarray(res.visible, bool))
            scale_list.append(float(getattr(res, "scale", np.nan)))
            ok_list.append(True)
        else:
            joints_list.append(None)
            vis_list.append(None)
            scale_list.append(np.nan)
            ok_list.append(False)
    J = J or _num_joints_fallback()
    out = dict(npz)
    out["joints"] = np.stack([j if j is not None else np.full((J, 2), np.nan, np.float32)
                              for j in joints_list]).astype(np.float32)
    out["visible"] = np.stack([v if v is not None else np.zeros(J, bool) for v in vis_list])
    out["ok"] = np.asarray(ok_list, bool)
    out["scale"] = np.asarray(scale_list, np.float32)
    out["adapter_version"] = np.str_(adapter_version())
    return out


def _num_joints_fallback() -> int:
    try:
        from gesture import skeleton_adapter
        return int(getattr(skeleton_adapter, "NUM_JOINTS"))
    except Exception:
        return 15


def save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    """Ghi file tạm rồi đổi tên, để file .npz không bao giờ bị ghi dở khi bị ngắt giữa chừng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **data)
    os.replace(tmp, path)


def process_entry(job) -> Dict[str, object]:
    """Hàm chạy trong tiến trình con cho một video. Trả về một dòng báo cáo."""
    entry, video_path, out_path, opts = job
    row: Dict[str, object] = dict(rel_path=entry.rel_path, split=entry.split,
                                  class_name=entry.class_name, subject=entry.subject)
    t0 = time.time()
    try:
        if opts["from_raw"]:
            if not out_path.exists():
                row["status"] = "no_npz"
                return row
            with np.load(out_path, allow_pickle=False) as z:
                data = readapt({k: z[k] for k in z.files})
        else:
            if video_path is None:
                row["status"] = "missing_video"
                return row
            data = extract_video(video_path)
            data.update(
                label=np.int32(entry.label),
                class_name=np.str_(entry.class_name),
                subject=np.str_(entry.subject),
                rel_path=np.str_(entry.rel_path),
                pose_model=np.str_(opts["pose_model"]),
                yolo_weights=np.str_(opts["yolo_weights"] or "none"),
                adapter_version=np.str_(_WORKER.get("adapter_version", adapter_version())),
            )
        save_npz(out_path, data)

        n = len(data["t"])
        row.update(
            status="ok",
            n_frames=n,
            fps=round(float(data["fps"]), 2),
            duration_s=round(n / float(data["fps"]), 2) if n else 0,
            person_ratio=round(int(data.get("n_person", 0)) / n, 3) if n else 0,
            pose_ratio=round(int(data.get("n_pose", 0)) / n, 3) if n else 0,
            ok_ratio=round(float(np.mean(data["ok"])), 3) if n else 0,
            seconds=round(time.time() - t0, 1),
        )
    except Exception as e:  # một video lỗi không được làm dừng cả lượt chạy
        row["status"] = f"error: {type(e).__name__}: {e}"
        row["trace"] = traceback.format_exc(limit=3)
    return row


# ---------------------------------------------------------------------------
# Video xem thử
# ---------------------------------------------------------------------------
# Các cặp điểm MediaPipe để vẽ khung xương
_MP_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
             (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


def write_preview(video_path: Path, npz_path: Path, out_path: Path) -> None:
    """Vẽ bbox + khung xương đã lưu lên video gốc để kiểm tra bằng mắt."""
    import cv2
    with np.load(npz_path, allow_pickle=False) as z:
        kps, boxes, ok = z["kps_raw"], z["bbox"], z["ok"]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        r, frame = cap.read()
        if not r or i >= len(kps):
            break
        if boxes[i][0] >= 0:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if ok[i] else (0, 0, 255), 2)
        p = kps[i]
        if not np.isnan(p).all():
            for a, b in _MP_EDGES:
                cv2.line(frame, (int(p[a, 0]), int(p[a, 1])), (int(p[b, 0]), int(p[b, 1])), (255, 200, 0), 2)
            # tay trái của người = xanh lá, tay phải = đỏ -> kiểm tra trái/phải
            cv2.circle(frame, (int(p[15, 0]), int(p[15, 1])), 6, (0, 255, 0), -1)
            cv2.circle(frame, (int(p[16, 0]), int(p[16, 1])), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"frame {i} ok={bool(ok[i])}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trích keypoint từ video RoCoG")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="train / val / test / train_full / syn_train / syn_val, "
                         "hoặc tên file annotation bất kỳ (không cần .txt)")
    ap.add_argument("--ann-dir", default=str(config.ANNOTATION_DIR))
    ap.add_argument("--dataset-root", default=str(config.DATASET_ROOT))
    ap.add_argument("--out-dir", default=str(config.KEYPOINT_DIR))
    ap.add_argument("--yolo", default=config.YOLO_WEIGHTS, help="trọng số YOLO")
    ap.add_argument("--no-yolo", action="store_true",
                    help="bỏ YOLO, chạy pose trên cả khung hình (nhanh hơn, nhưng khác lúc chạy thật)")
    ap.add_argument("--device", default=None, help="thiết bị cho YOLO, ví dụ 0 hoặc cpu")
    ap.add_argument("--pose-model", default=config.POSE_MODEL_NAME,
                    help="chỉ để ghi lại vào .npz; model thật do pose_estimate.py quyết định")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="lấy tối đa bấy nhiêu video mỗi lớp (cho dữ liệu tổng hợp)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N video đầu (để thử)")
    ap.add_argument("--overwrite", action="store_true", help="trích lại cả video đã có .npz")
    ap.add_argument("--from-raw", action="store_true",
                    help="không chạy YOLO/pose, chỉ tính lại joints từ kps_raw trong .npz đã có")
    ap.add_argument("--preview-count", type=int, default=0,
                    help="ghi video xem thử (khung xương vẽ lên video) cho N video đầu")
    ap.add_argument("--min-ok-ratio", type=float, default=config.MIN_OK_RATIO)
    args = ap.parse_args()

    # lấy mẫu cân bằng RIÊNG từng tập (mỗi tập tối đa max-per-class video mỗi lớp)
    entries: List[VideoEntry] = []
    for s in args.splits:
        entries.extend(sample_balanced(read_splits([s], args.ann_dir), args.max_per_class))
    # một video có thể nằm trong nhiều file chia (train_full) -> bỏ trùng
    seen, uniq = set(), []
    for e in entries:
        if e.rel_path not in seen:
            seen.add(e.rel_path)
            uniq.append(e)
    entries = uniq
    if args.limit:
        entries = entries[: args.limit]

    out_dir = Path(args.out_dir)
    yolo_weights = None if args.no_yolo else args.yolo
    opts = dict(from_raw=args.from_raw, pose_model=args.pose_model, yolo_weights=yolo_weights)

    jobs, rows = [], []
    for e in entries:
        out_path = out_dir / e.npz_relpath()
        if out_path.exists() and not (args.overwrite or args.from_raw):
            rows.append(dict(rel_path=e.rel_path, split=e.split, class_name=e.class_name,
                             subject=e.subject, status="exists"))
            continue
        video_path = None if args.from_raw else resolve_video_path(e, args.dataset_root)
        jobs.append((e, video_path, out_path, opts))

    print(f"{len(entries)} video, {len(jobs)} cần xử lý, {len(entries) - len(jobs)} đã có .npz")
    if jobs and not args.from_raw:
        n_missing = sum(1 for j in jobs if j[1] is None)
        if n_missing:
            print(f"[CẢNH BÁO] không tìm thấy {n_missing} video trong {args.dataset_root}")

    t_start = time.time()
    if jobs:
        if args.from_raw or args.workers <= 1:
            if not args.from_raw:
                _init_worker(yolo_weights, args.device)
            for k, job in enumerate(jobs, 1):
                row = process_entry(job)
                rows.append(row)
                _print_progress(k, len(jobs), row, t_start)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=_init_worker,
                          initargs=(yolo_weights, args.device)) as pool:
                for k, row in enumerate(pool.imap_unordered(process_entry, jobs), 1):
                    rows.append(row)
                    _print_progress(k, len(jobs), row, t_start)

    # ---- báo cáo ----
    report = out_dir / ("readapt_report.csv" if args.from_raw else "extract_report.csv")
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rel_path", "split", "class_name", "subject", "status", "n_frames", "fps",
              "duration_s", "person_ratio", "pose_ratio", "ok_ratio", "seconds"]
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print_summary(rows, args.min_ok_ratio)
    print(f"\nBáo cáo chi tiết: {report}")

    errors = [r for r in rows if str(r.get("status", "")).startswith("error")]
    if errors:
        print(f"\n{len(errors)} video lỗi, ví dụ:\n{errors[0].get('trace', errors[0]['status'])}")

    if args.preview_count and not args.from_raw:
        prev_dir = out_dir / "_preview"
        done = 0
        for e in entries:
            vp = resolve_video_path(e, args.dataset_root)
            npz = out_dir / e.npz_relpath()
            if vp is None or not npz.exists():
                continue
            write_preview(vp, npz, prev_dir / (e.stem + ".mp4"))
            done += 1
            if done >= args.preview_count:
                break
        print(f"Đã ghi {done} video xem thử vào {prev_dir} "
              f"(chấm xanh lá = cổ tay TRÁI của người, chấm đỏ = cổ tay PHẢI)")


def _print_progress(k: int, n: int, row: Dict[str, object], t_start: float) -> None:
    elapsed = time.time() - t_start
    eta = elapsed / k * (n - k)
    status = row.get("status")
    extra = f"ok={row.get('ok_ratio')}" if status == "ok" else status
    print(f"[{k}/{n}] {Path(str(row['rel_path'])).name}  {extra}  "
          f"(đã chạy {elapsed / 60:.1f} phút, còn ~{eta / 60:.1f} phút)", flush=True)


def print_summary(rows: List[Dict[str, object]], min_ok: float) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    status_count = Counter(str(r.get("status")).split(":")[0] for r in rows)
    print("\n===== TÓM TẮT =====")
    for s, c in status_count.items():
        print(f"  {s}: {c}")
    if not ok_rows:
        return
    by_class: Dict[str, List[float]] = defaultdict(list)
    for r in ok_rows:
        by_class[str(r["class_name"])].append(float(r["ok_ratio"]))
    print(f"\n{'Lớp':<15}{'số video':>10}{'ok trung bình':>16}{'video kém':>12}")
    for c in sorted(by_class):
        v = by_class[c]
        bad = sum(1 for x in v if x < min_ok)
        print(f"{c:<15}{len(v):>10}{np.mean(v):>16.3f}{bad:>12}")
    bad_rows = sorted((r for r in ok_rows if float(r["ok_ratio"]) < min_ok),
                      key=lambda r: float(r["ok_ratio"]))
    if bad_rows:
        print(f"\nVideo có tỉ lệ frame dùng được < {min_ok} (sẽ bị loại khi train, nên xem lại):")
        for r in bad_rows[:15]:
            print(f"  ok={r['ok_ratio']:<6} person={r['person_ratio']:<6} pose={r['pose_ratio']:<6} {r['rel_path']}")


if __name__ == "__main__":
    sys.exit(main())