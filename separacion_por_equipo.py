import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans

INPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_finales"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_team_finales"

SEG_MODEL_PATH = "yolov8x-seg.pt"


def extract_feature_from_bbox(frame, bbox):
    """
    Fallback cuando no hay mascara usable.
    Se usa región torso para reducir césped.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    y_top = int(h * 0.15)
    y_bot = int(h * 0.70)
    x_left = int(w * 0.25)
    x_right = int(w * 0.75)

    crop_torso = crop[y_top:y_bot, x_left:x_right]
    if crop_torso.size == 0:
        return None

    lab = cv2.cvtColor(crop_torso, cv2.COLOR_BGR2LAB)
    mean = lab.reshape(-1, 3).mean(axis=0)
    return mean.astype(np.float32)


def mask_color_feature(frame, mask, bbox):
    """
    Extrae feature LAB promedio SOLO usando mascara.
    Si la mascara es mala -> fallback a bbox.
    """
    x1, y1, x2, y2 = bbox

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    # torso
    y_top = int(h * 0.15)
    y_bot = int(h * 0.70)
    x_left = int(w * 0.25)
    x_right = int(w * 0.75)

    crop_torso = crop[y_top:y_bot, x_left:x_right]
    if crop_torso.size == 0:
        return None

    # si no hay mascara -> fallback
    if mask is None:
        return extract_feature_from_bbox(frame, bbox)

    mask_crop = mask[y1:y2, x1:x2]
    if mask_crop.size == 0:
        return extract_feature_from_bbox(frame, bbox)

    mask_crop = mask_crop[y_top:y_bot, x_left:x_right]
    mask_bin = (mask_crop > 0)

    # mascara demasiado pequeña -> fallback
    if np.count_nonzero(mask_bin) < 50:
        return extract_feature_from_bbox(frame, bbox)

    lab = cv2.cvtColor(crop_torso, cv2.COLOR_BGR2LAB)
    pixels = lab[mask_bin]

    if len(pixels) < 20:
        return extract_feature_from_bbox(frame, bbox)

    return pixels.mean(axis=0).astype(np.float32)


def process_json(players_json_path, seg_model):
    base = players_json_path.stem.replace("_players", "")

    with open(players_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = INPUT_DIR / f"{base}_01_original.jpg"
    frame = cv2.imread(str(img_path))

    if frame is None:
        print(f"[WARN] No existe imagen original para {base}")
        return

    players = data["players"]
    direction = data.get("direction", "right")

    if len(players) < 2:
        print(f"[WARN] Muy pocos jugadores en {base}")
        return

    # correr YOLO-seg
    results = seg_model.predict(source=frame, conf=0.20, imgsz=1280, verbose=False)

    seg_boxes = None
    seg_classes = None
    seg_masks = None

    if len(results) > 0 and results[0].masks is not None:
        r = results[0]
        seg_boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        seg_classes = r.boxes.cls.cpu().numpy().astype(int)
        seg_masks = r.masks.data.cpu().numpy()  # [N,H,W]

    # indices de personas segmentadas
    person_idxs = []
    if seg_classes is not None:
        person_idxs = [i for i, c in enumerate(seg_classes) if c == 0]

    features = []
    valid_players = []

    for p in players:
        bbox = p["bbox"]
        x1, y1, x2, y2 = bbox

        best_mask = None
        best_iou = 0

        if seg_boxes is not None and len(person_idxs) > 0:
            for i in person_idxs:
                bx1, by1, bx2, by2 = seg_boxes[i]

                ix1 = max(x1, bx1)
                iy1 = max(y1, by1)
                ix2 = min(x2, bx2)
                iy2 = min(y2, by2)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih

                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter

                if union <= 0:
                    continue

                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_mask = seg_masks[i]

        # si IoU es muy bajo, ignoramos la máscara y usamos fallback bbox
        if best_iou < 0.10:
            best_mask = None

        feat = mask_color_feature(frame, best_mask, bbox)
        if feat is None:
            continue

        features.append(feat)
        valid_players.append(p)

    if len(features) < 2:
        print(f"[WARN] No hay suficientes jugadores para clustering en {base}")
        return

    X = np.array(features, dtype=np.float32)

    # clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)

    team_json = {
        "video": data.get("video", base),
        "direction": direction,
        "players": []
    }

    out = frame.copy()

    for p, lab in zip(valid_players, labels):
        bbox = p["bbox"]
        adv = p["advanced_point"]

        team_name = "TeamA" if lab == 0 else "TeamB"
        color = (255, 0, 0) if lab == 0 else (0, 0, 255)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, team_name, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.circle(out, (adv[0], adv[1]), 7, (255, 255, 0), -1)

        team_json["players"].append({
            "id": p["id"],
            "bbox": bbox,
            "advanced_point": adv,
            "team": team_name
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_out_path = OUTPUT_DIR / f"{base}_teams_seg.jpg"
    cv2.imwrite(str(img_out_path), out)

    json_out_path = OUTPUT_DIR / f"{base}_teamdata.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(team_json, f, indent=4)

    print(f"[OK] {base} -> {img_out_path.name} + {json_out_path.name}")


def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seg_model = YOLO(SEG_MODEL_PATH)

    json_files = sorted(INPUT_DIR.glob("*_players.json"))
    if len(json_files) == 0:
        print("[WARN] No hay JSON *_players.json en outputs_field_filter_adv")
        return

    for jf in json_files:
        process_json(jf, seg_model)

    print("[FIN]")


if __name__ == "__main__":
    run_all()