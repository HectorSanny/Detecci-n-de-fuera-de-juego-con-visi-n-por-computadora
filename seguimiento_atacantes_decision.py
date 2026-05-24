import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "yolov8x.pt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

INPUT_JSON_DIR = Path(__file__).resolve().parent.parent / "outputs_offside_results"
VIDEO_DIR = Path(__file__).resolve().parent.parent / "videos"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_tracking_ball_touch_eval"

# Detection thresholds
CONF_PERSON = 0.25
CONF_BALL = 0.12

BALL_CLASS_ID = 32  # COCO sports ball
PERSON_CLASS_ID = 0

# FPS reduction
TARGET_FPS = 15

# Ignore first frames (touch previous action)
IGNORE_FIRST_PROCESSED_FRAMES = 3

# Tracking config (IoU matching)
TRACK_IOU_THRES = 0.25
TRACK_MAX_MISSING = 8  # frames allowed without match

# Offside label assignment to track
LABEL_IOU_THRES = 0.20

# Touch detection
TOUCH_DIST_PX = 25
TOUCH_CONSEC_FRAMES = 2

# fallback if video ends during touch
FALLBACK_DIST_FACTOR = 1.15


# ============================================================
# METRICS
# ============================================================
def compute_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1


# ============================================================
# HELPERS
# ============================================================
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def bbox_to_bbox_distance(b1, b2):
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2

    dx = max(x3 - x2, x1 - x4, 0)
    dy = max(y3 - y2, y1 - y4, 0)

    return float(np.sqrt(dx * dx + dy * dy))


def draw_bbox(frame, bbox, color, thickness=3):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, x, y, color, scale=0.8, thickness=2):
    cv2.putText(frame, text, (x, max(25, y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def status_color(status):
    if status == "offside":
        return (0, 0, 255)
    if status == "onside":
        return (0, 255, 0)
    return (255, 255, 255)


def find_video(base):
    mp4 = VIDEO_DIR / f"{base}.mp4"
    if mp4.exists():
        return mp4

    for ext in VIDEO_EXTS:
        p = VIDEO_DIR / f"{base}{ext}"
        if p.exists():
            return p

    return None


def get_ground_truth_from_name(video_name):
    name = video_name.lower()
    if "offside" in name:
        return "OFFSIDE"
    if "onside" in name:
        return "ONSIDE"
    return None


def match_status_to_bbox(bbox, ref_attackers):
    best_iou_val = 0
    best_status_val = None

    for ra in ref_attackers:
        iou_val = bbox_iou(bbox, tuple(ra["bbox"]))
        if iou_val > best_iou_val:
            best_iou_val = iou_val
            best_status_val = ra["status"]

    if best_iou_val >= LABEL_IOU_THRES:
        return best_status_val, best_iou_val

    return None, best_iou_val


# ============================================================
# SIMPLE TRACKER (IoU-based)
# ============================================================
class SimpleTracker:
    def __init__(self, iou_thr=0.25, max_missing=8):
        self.iou_thr = iou_thr
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}  # id -> {"bbox":..., "missing":...}

    def update(self, detections):
        """
        detections: list of bbox tuples
        returns: list of (track_id, bbox)
        """

        if len(self.tracks) == 0:
            out = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det, "missing": 0}
                out.append((tid, det))
            return out

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        used_tracks = set()
        used_dets = set()
        matches = []

        # greedy IoU matching
        for det_idx, det_bbox in enumerate(detections):
            best_iou_val = 0
            best_track_idx = None

            for t_idx, tb in enumerate(track_bboxes):
                tid = track_ids[t_idx]
                if tid in used_tracks:
                    continue

                iou_val = bbox_iou(det_bbox, tb)
                if iou_val > best_iou_val:
                    best_iou_val = iou_val
                    best_track_idx = t_idx

            if best_track_idx is not None and best_iou_val >= self.iou_thr:
                tid = track_ids[best_track_idx]
                matches.append((tid, det_bbox))
                used_tracks.add(tid)
                used_dets.add(det_idx)

        # update matched tracks
        for tid, bb in matches:
            self.tracks[tid]["bbox"] = bb
            self.tracks[tid]["missing"] = 0

        # create new tracks
        for det_idx, det_bbox in enumerate(detections):
            if det_idx in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": det_bbox, "missing": 0}
            matches.append((tid, det_bbox))

        # update missing
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

        return matches


# ============================================================
# MAIN PROCESS
# ============================================================
def process_play(json_path, model):
    base = json_path.stem.replace("_offside_labeled", "")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    attacking_team = data.get("attacking_team", None)
    if attacking_team is None:
        print(f"[WARN] No attacking_team in {json_path.name}")
        return None

    players = data["players"]
    attackers = [p for p in players if p.get("team") == attacking_team and "status" in p]

    if len(attackers) == 0:
        print(f"[WARN] No attackers with status in {base}")
        return None

    ref_attackers = [{"bbox": p["bbox"], "status": p["status"]} for p in attackers]

    video_path = find_video(base)
    if video_path is None:
        print(f"[WARN] Video not found for {base}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video {video_path.name}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30

    frame_skip = max(1, int(round(orig_fps / TARGET_FPS)))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / f"{base}_tracked_touch_{TARGET_FPS}fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (w, h))

    tracker = SimpleTracker(iou_thr=TRACK_IOU_THRES, max_missing=TRACK_MAX_MISSING)

    track_label = {}  # tid -> "offside"/"onside"

    touch_status = None
    touch_frame = None
    fallback_used = False

    touch_counter = {}  # tid -> count

    best_candidate_status_val = None
    best_candidate_frame = None
    best_candidate_dist = 1e9

    frame_id = 0
    processed_id = 0

    print(f"\n[INFO] Processing {video_path.name} | frames={total_frames} | fps={orig_fps:.2f} | skip={frame_skip} -> {TARGET_FPS}fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        touch_found = False

        results = model.predict(
            source=frame,
            conf=min(CONF_PERSON, CONF_BALL),
            imgsz=1280,
            verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            writer.write(frame)
            processed_id += 1
            frame_id += 1
            continue

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        person_dets = []
        ball_bbox = None
        best_ball_conf = 0

        for bb, cid, sc in zip(boxes, clss, confs):
            bb_t = tuple(bb.tolist())

            if cid == PERSON_CLASS_ID and sc >= CONF_PERSON:
                person_dets.append(bb_t)

            if cid == BALL_CLASS_ID and sc >= CONF_BALL:
                if sc > best_ball_conf:
                    best_ball_conf = sc
                    ball_bbox = bb_t

        tracked = tracker.update(person_dets)

        if ball_bbox is not None:
            draw_bbox(frame, ball_bbox, (255, 255, 255), 2)
            draw_label(frame, "BALL", ball_bbox[0], ball_bbox[1] - 10, (255, 255, 255), 0.8, 2)

        for tid, bb in tracked:
            if tid not in track_label:
                st, iou_val = match_status_to_bbox(bb, ref_attackers)
                if st is not None:
                    track_label[tid] = st

            if tid not in track_label:
                continue

            st = track_label[tid]
            color = status_color(st)

            draw_bbox(frame, bb, color, 3)
            draw_label(frame, f"ID:{tid} {st.upper()}", bb[0], bb[1] - 10, color, 0.75, 2)

            # IMPORTANT FIX: ignore first processed frames
            if processed_id >= IGNORE_FIRST_PROCESSED_FRAMES and ball_bbox is not None and touch_status is None:
                d = bbox_to_bbox_distance(bb, ball_bbox)

                if d < best_candidate_dist:
                    best_candidate_dist = d
                    best_candidate_status_val = st
                    best_candidate_frame = processed_id

                if d < TOUCH_DIST_PX:
                    touch_counter[tid] = touch_counter.get(tid, 0) + 1
                else:
                    touch_counter[tid] = 0

                if touch_counter[tid] >= TOUCH_CONSEC_FRAMES:
                    touch_status = st
                    touch_frame = processed_id
                    touch_found = True
                    break

        cv2.putText(frame, f"{base} | Frame {processed_id} ({TARGET_FPS}fps)", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        if touch_status is not None:
            cv2.putText(frame, f"TOUCH CONFIRMED: {touch_status.upper()}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        elif best_candidate_status_val is not None:
            cv2.putText(frame, f"BEST: {best_candidate_status_val.upper()} dist={best_candidate_dist:.1f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        writer.write(frame)

        processed_id += 1
        frame_id += 1

        if touch_found:
            print(f"[INFO] Touch confirmed early at processed frame {touch_frame}. STOP.")
            break

    cap.release()
    writer.release()

    if touch_status is None:
        if best_candidate_status_val is not None and best_candidate_dist < TOUCH_DIST_PX * FALLBACK_DIST_FACTOR:
            touch_status = best_candidate_status_val
            touch_frame = best_candidate_frame
            fallback_used = True

    if touch_status is None:
        decision = "UNKNOWN"
    else:
        decision = "OFFSIDE" if touch_status == "offside" else "ONSIDE"

    gt = get_ground_truth_from_name(video_path.name)

    summary = {
        "play": base,
        "video": video_path.name,
        "attacking_team": attacking_team,
        "touch_frame": touch_frame,
        "touch_status": touch_status,
        "decision": decision,
        "fallback_used": fallback_used,
        "best_candidate_dist": float(best_candidate_dist),
        "ground_truth": gt,
        "original_fps": float(orig_fps),
        "target_fps": TARGET_FPS,
        "frame_skip": int(frame_skip),
        "touch_dist_px": TOUCH_DIST_PX,
        "touch_consec_frames": TOUCH_CONSEC_FRAMES,
        "ignore_first_processed_frames": IGNORE_FIRST_PROCESSED_FRAMES
    }

    summary_path = OUTPUT_DIR / f"{base}_touch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"[OK] Saved -> {out_path.name}")
    if fallback_used:
        print(f"[FALLBACK USED] best_dist={best_candidate_dist:.2f}")
    print(f"[DECISION] VIDEO={video_path.name} | GT={gt} | FINAL={decision}")

    return summary


# ============================================================
# RUN ALL + EVALUATION
# ============================================================
def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    json_files = sorted(INPUT_JSON_DIR.glob("*_offside_labeled.json"))
    if len(json_files) == 0:
        print("[WARN] No *_offside_labeled.json found in outputs_offside_results")
        return

    all_summaries = []

    tp = tn = fp = fn = 0
    ignored = 0

    for jf in json_files:
        s = process_play(jf, model)
        if s is None:
            continue

        all_summaries.append(s)

        gt = s["ground_truth"]
        pred = s["decision"]

        if gt is None or pred == "UNKNOWN":
            ignored += 1
            continue

        # Positive = OFFSIDE
        if gt == "OFFSIDE" and pred == "OFFSIDE":
            tp += 1
        elif gt == "ONSIDE" and pred == "ONSIDE":
            tn += 1
        elif gt == "ONSIDE" and pred == "OFFSIDE":
            fp += 1
        elif gt == "OFFSIDE" and pred == "ONSIDE":
            fn += 1

    global_summary_path = OUTPUT_DIR / "ALL_TOUCH_SUMMARY.json"
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4)

    acc, prec, rec, f1 = compute_metrics(tp, tn, fp, fn)

    print("\n======================")
    print("[FINAL EVALUATION]")
    print("======================")
    print(f"TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    print(f"Ignored (UNKNOWN or missing GT): {ignored}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("======================")
    print(f"[FINISHED] Saved global summary -> {global_summary_path.name}")


if __name__ == "__main__":
    run_all()