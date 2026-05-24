import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolov8x.pt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
OUTPUT_DIR_NAME = ("outputs_finales"
                   "")
FRAME_OFFSETS = (-2, -1, 0, 1, 2)
MIN_DET_CONF = 0.22


def get_attack_direction_from_filename(filename: str):
    n = filename.lower()
    if n.endswith("izq") or "_izq" in n or "-izq" in n or "izq" in n:
        return "left"
    if n.endswith("der") or "_der" in n or "-der" in n or "der" in n:
        return "right"
    return "right"


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 80:
        scale = 1.5
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return frame, scale
    return frame, 1.0


def is_valid_bbox(bbox, frame_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if w * h < 120:
        return False
    if h < 18 or w < 8:
        return False
    if y2 < frame_h * 0.18:
        return False
    if (w / max(h, 1)) > 1.25:
        return False
    return True


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


def suppress_near_duplicates(boxes, scores, iou_thr=0.82):
    if len(boxes) <= 1:
        return boxes, scores
    idxs = np.argsort(-np.array(scores))
    keep = []
    for i in idxs:
        bi = boxes[i]
        dup = False
        for j in keep:
            if bbox_iou(bi, boxes[j]) >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append(i)
    keep = sorted(keep)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def detect_people_in_frame(frame, model):
    frame_proc, scale = preprocess_frame(frame)
    h_proc = frame_proc.shape[0]
    roi_y = int(h_proc * 0.18)
    roi = frame_proc[roi_y:h_proc, :]

    results = model.predict(
        source=roi,
        conf=MIN_DET_CONF,
        imgsz=1280,
        iou=0.55,
        max_det=250,
        verbose=False
    )

    dets = []
    if len(results) == 0 or results[0].boxes is None:
        return dets

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    scores = results[0].boxes.conf.cpu().numpy()

    for bbox, cls_id, score in zip(boxes, classes, scores):
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = bbox.astype(int)
        y1 += roi_y
        y2 += roi_y

        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        bb = (x1, y1, x2, y2)
        if not is_valid_bbox(bb, frame.shape[0]):
            continue

        dets.append({"bbox": bb, "score": float(score)})

    return dets


def detect_multi_frame(video_path, model):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, None, None

    all_dets = []
    ref_frame = None

    for off in FRAME_OFFSETS:
        fid = max(0, min(total - 1, off))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        if ref_frame is None or off == 0:
            ref_frame = frame.copy()

        for d in detect_people_in_frame(frame, model):
            d["frame_id"] = fid
            all_dets.append(d)

    cap.release()

    if ref_frame is None or len(all_dets) == 0:
        return None, None, None

    clusters = []
    for det in all_dets:
        assigned = False
        for cl in clusters:
            if bbox_iou(det["bbox"], cl["best_bbox"]) >= 0.35:
                cl["items"].append(det)
                cl["boxes"].append(det["bbox"])
                cl["scores"].append(det["score"])
                if det["score"] > cl["best_score"]:
                    cl["best_bbox"] = det["bbox"]
                    cl["best_score"] = det["score"]
                assigned = True
                break
        if not assigned:
            clusters.append({
                "items": [det],
                "boxes": [det["bbox"]],
                "scores": [det["score"]],
                "best_bbox": det["bbox"],
                "best_score": det["score"]
            })

    merged = []
    for cl in clusters:
        if len(cl["items"]) < 2:
            continue
        weights = np.array(cl["scores"], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-6)
        boxes = np.array(cl["boxes"], dtype=np.float32)
        merged_box = np.sum(boxes * weights[:, None], axis=0).astype(int)
        merged.append({"bbox": tuple(merged_box), "score": float(np.max(cl["scores"]))})

    final_boxes = [m["bbox"] for m in merged]
    final_scores = [m["score"] for m in merged]
    final_boxes, final_scores = suppress_near_duplicates(final_boxes, final_scores)

    return ref_frame, final_boxes, all_dets


def save_vis(path, img):
    cv2.imwrite(str(path), img)


def green_field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 35, 35])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_field_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def estimate_field_region(frame):
    mask = green_field_mask(frame)
    field = largest_field_component(mask)
    return field


def point_in_polygon(pt, poly):
    x, y = pt
    return cv2.pointPolygonTest(poly.astype(np.int32), (float(x), float(y)), False) >= 0


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_advanced_point(bbox, direction="right"):
    x1, y1, x2, y2 = bbox
    foot_y = int(y1 + 0.80 * (y2 - y1))
    return (x2, foot_y) if direction == "right" else (x1, foot_y)


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, bbox, text, color=(255, 255, 255)):
    x1, y1, _, _ = bbox
    cv2.putText(frame, text, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_point(frame, pt, color=(0, 0, 255), radius=6):
    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)


def point_in_bbox(pt, bbox):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def manual_filter_bboxes(frame, bboxes):
    removed_idx = set()
    window_name = "Filtrado manual (click elimina/restaura) - ENTER confirmar - ESC cancelar"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, b in enumerate(bboxes):
                if point_in_bbox((x, y), b):
                    if i in removed_idx:
                        removed_idx.remove(i)
                    else:
                        removed_idx.add(i)
                    break

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        vis = frame.copy()

        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = b

            if i in removed_idx:
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 3

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(vis, f"{i}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(vis, f"Eliminados: {len(removed_idx)}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.putText(vis, "Click en arbitros/arqueros para eliminarlos",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(vis, "ENTER = confirmar | ESC = cancelar",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:
            break
        if key == 27:
            cv2.destroyAllWindows()
            return bboxes

    cv2.destroyAllWindows()
    filtered = [b for i, b in enumerate(bboxes) if i not in removed_idx]
    return filtered


def process_video(video_path, output_folder, model):
    output_folder.mkdir(parents=True, exist_ok=True)
    direction = get_attack_direction_from_filename(video_path.stem)

    frame, valid_bboxes, raw_dets = detect_multi_frame(video_path, model)
    if frame is None or not valid_bboxes:
        print(f"[WARN] Sin detecciones útiles: {video_path.name}")
        return

    base = video_path.stem
    save_vis(output_folder / f"{base}_01_original.jpg", frame)

    field_mask = estimate_field_region(frame)

    h, w = frame.shape[:2]
    field_poly = np.array([
        [0, int(0.12 * h)],
        [w - 1, int(0.10 * h)],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.int32)

    kept = []
    removed = []
    for d in valid_bboxes:
        cx, cy = bbox_center(d)
        inside = point_in_polygon((cx, cy), field_poly)
        if inside:
            kept.append(d)
        else:
            removed.append(d)

    # filtrado manual
    if len(kept) > 0:
        print(f"[INFO] {video_path.name} -> filtrado manual: click en arbitros/arqueros y ENTER")
        kept = manual_filter_bboxes(frame, kept)

    # guardar JSON con bbox finales
    json_data = {
        "video": video_path.name,
        "direction": direction,
        "players": []
    }

    for i, bbox in enumerate(kept):
        adv = get_advanced_point(bbox, direction)
        json_data["players"].append({
            "id": i,
            "bbox": list(map(int, bbox)),
            "advanced_point": [int(adv[0]), int(adv[1])]
        })

    json_path = output_folder / f"{base}_players.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # imagen final
    final_vis = frame.copy()
    for i, bbox in enumerate(kept):
        draw_bbox(final_vis, bbox, (0, 255, 0), 3)
        adv = get_advanced_point(bbox, direction)
        draw_point(final_vis, adv, (255, 255, 0), 6)
        draw_label(final_vis, bbox, f"p{i}")

    cv2.putText(final_vis, f"Direction: {direction}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.putText(final_vis, f"Players: {len(kept)}", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    save_vis(output_folder / f"{base}_FINAL.jpg", final_vis)

    print(f"[OK] {video_path.name} | campo={len(valid_bboxes)} | manual={len(kept)} | json={json_path.name}")


def run_all():
    BASE_DIR = Path(__file__).resolve().parent.parent
    videos_folder = BASE_DIR / "videos"
    output_folder = BASE_DIR / OUTPUT_DIR_NAME

    model = YOLO(MODEL_PATH)

    videos = [f for f in videos_folder.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    print(f"[INFO] Videos encontrados: {len(videos)}")

    for vid in videos:
        process_video(vid, output_folder, model)

    print("[FIN]")


if __name__ == "__main__":
    run_all()