import cv2
import numpy as np
import json
from pathlib import Path

# ---------------- PATHS ----------------
INPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_team_finales"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_offside_results"

# ---------------- COLORS (BGR) ----------------
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


# ============================================================
#                    GEOMETRY / MATH
# ============================================================

def line_intersection(l1, l2):
    """
    Intersection of infinite lines defined by segments l1 and l2.
    Each line is (x1,y1,x2,y2).
    Returns (px,py) float or None if parallel.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    return (px, py)


def x_at_y(p1, p2, y_target):
    """
    Returns x coordinate where line passing through p1->p2 intersects horizontal line y=y_target.
    p1,p2 are (x,y). Returns float x or None if horizontal.
    """
    x1, y1 = p1
    x2, y2 = p2

    if abs(y2 - y1) < 1e-6:
        return None

    t = (y_target - y1) / (y2 - y1)
    x = x1 + t * (x2 - x1)
    return float(x)


def compute_offside_by_baseline(attackers, defenders, vp, direction, frame_h):
    """
    Correct offside decision in perspective:
    Compare each player's "x position" at a fixed baseline y = y_base.
    That baseline acts like a comparable line in the image.

    We compute x_intersect of line (VP -> advanced_point) with y=y_base.

    - direction == "right": larger x_intersect => more advanced
    - direction == "left": smaller x_intersect => more advanced

    Returns:
      best_def_player, best_def_x, statuses_list, y_base
    where statuses_list = [(player_dict, "offside"/"onside", x_intersect), ...]
    """
    y_base = int(frame_h * 0.88)

    attacker_vals = []
    defender_vals = []

    for p in attackers:
        adv = p["advanced_point"]
        xv = x_at_y(vp, adv, y_base)
        if xv is not None and np.isfinite(xv):
            attacker_vals.append((p, xv))

    for p in defenders:
        adv = p["advanced_point"]
        xv = x_at_y(vp, adv, y_base)
        if xv is not None and np.isfinite(xv):
            defender_vals.append((p, xv))

    if len(attacker_vals) == 0 or len(defender_vals) == 0:
        return None, None, None, y_base

    if direction == "right":
        best_def_player, best_def_x = max(defender_vals, key=lambda x: x[1])
    else:
        best_def_player, best_def_x = min(defender_vals, key=lambda x: x[1])

    statuses = []
    for pl, xv in attacker_vals:
        if direction == "right":
            st = "offside" if xv > best_def_x else "onside"
        else:
            st = "offside" if xv < best_def_x else "onside"
        statuses.append((pl, st, xv))

    return best_def_player, best_def_x, statuses, y_base


# ============================================================
#                    MANUAL UI HELPERS
# ============================================================

def get_attack_team_by_click(image, players):
    """
    Click on a player bbox to select attacking team.
    """
    selected_team = {"team": None}
    window = "CLICK attacker team player (ESC cancel)"

    def point_in_bbox(x, y, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for p in players:
                if point_in_bbox(x, y, p["bbox"]):
                    selected_team["team"] = p["team"]
                    break

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = image.copy()

        cv2.putText(vis, "CLICK on a player from the ATTACKING team", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.putText(vis, "ESC to cancel", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF

        if selected_team["team"] is not None:
            break

        if key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return selected_team["team"]


def get_two_lines_click(image):
    """
    User clicks 4 points:
      - first 2 points => line1
      - next 2 points  => line2
    ENTER confirms when 4 points are placed.
    ESC cancels.
    """
    points = []
    window = "CLICK 4 points (2 per depth line). ENTER confirm. ESC cancel."

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = image.copy()

        for p in points:
            cv2.circle(vis, p, 6, (0, 255, 255), -1)

        if len(points) >= 2:
            cv2.line(vis, points[0], points[1], (0, 255, 0), 3)

        if len(points) >= 4:
            cv2.line(vis, points[2], points[3], (0, 255, 0), 3)

        cv2.putText(vis, f"Points: {len(points)}/4", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.putText(vis, "ENTER confirm | ESC cancel", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            return None, None

        if key == 13 and len(points) == 4:
            break

    cv2.destroyAllWindows()

    line1 = (points[0][0], points[0][1], points[1][0], points[1][1])
    line2 = (points[2][0], points[2][1], points[3][0], points[3][1])

    return line1, line2


# ============================================================
#                    DRAW HELPERS
# ============================================================

def draw_bbox(img, bbox, color, thickness=3):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_text(img, text, pos, color, scale=0.9, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_point(img, pt, color, r=6):
    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1)


# ============================================================
#                    MAIN PROCESS
# ============================================================

def process_play(json_path):
    base = json_path.stem.replace("_teamdata", "")

    img_path = INPUT_DIR / f"{base}_teams_seg.jpg"
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"[WARN] Missing image: {img_path.name}")
        return

    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    direction = data.get("direction", "right")
    players = data["players"]

    # ---------------- SELECT ATTACKING TEAM ----------------
    attacking_team = get_attack_team_by_click(img, players)
    if attacking_team is None:
        print(f"[SKIP] {base}")
        return

    defending_team = "TeamB" if attacking_team == "TeamA" else "TeamA"

    # ---------------- SELECT DEPTH LINES FOR VP ----------------
    line1, line2 = get_two_lines_click(img)
    if line1 is None:
        print(f"[SKIP] No depth lines selected: {base}")
        return

    vp = line_intersection(line1, line2)
    if vp is None:
        print(f"[WARN] Could not compute VP: {base}")
        return

    vp = (int(vp[0]), int(vp[1]))

    # ---------------- SPLIT TEAMS ----------------
    attackers = [p for p in players if p["team"] == attacking_team]
    defenders = [p for p in players if p["team"] == defending_team]

    # ---------------- OFFSIDE DECISION ----------------
    best_def, best_def_x, statuses, y_base = compute_offside_by_baseline(
        attackers, defenders, vp, direction, h
    )

    if best_def is None:
        print(f"[WARN] Offside computation failed: {base}")
        return

    # Assign status + baseline_x to attackers
    for pl, st, xv in statuses:
        pl["status"] = st
        pl["x_baseline"] = xv

    data["attacking_team"] = attacking_team
    data["defending_team"] = defending_team
    data["vanishing_point"] = [vp[0], vp[1]]
    data["baseline_y"] = y_base
    data["last_defender_baseline_x"] = best_def_x

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # IMAGE 01: BEFORE (all rays)
    # ============================================================
    before = img.copy()

    # draw depth guide lines
    cv2.line(before, (line1[0], line1[1]), (line1[2], line1[3]), GREEN, 3)
    cv2.line(before, (line2[0], line2[1]), (line2[2], line2[3]), GREEN, 3)

    # draw VP
    draw_point(before, vp, RED, 10)
    draw_text(before, "VP", (vp[0] + 10, vp[1] + 10), RED, 0.9, 2)

    # draw baseline
    cv2.line(before, (0, y_base), (w - 1, y_base), WHITE, 2)
    draw_text(before, f"baseline y={y_base}", (30, y_base - 10), WHITE, 0.8, 2)

    # draw all rays
    for p in players:
        adv = p["advanced_point"]
        cv2.line(before, (adv[0], adv[1]), vp, YELLOW, 2)
        draw_point(before, adv, CYAN, 6)

    cv2.imwrite(str(OUTPUT_DIR / f"{base}_01_before_filter.jpg"), before)

    # ============================================================
    # IMAGE 02: AFTER (attackers + last defender only)
    # ============================================================
    after = img.copy()

    draw_point(after, vp, RED, 10)
    cv2.line(after, (0, y_base), (w - 1, y_base), WHITE, 2)

    # draw last defender ray
    def_adv = best_def["advanced_point"]
    cv2.line(after, (def_adv[0], def_adv[1]), vp, GREEN, 5)
    draw_bbox(after, best_def["bbox"], WHITE, 4)
    draw_text(after, "LAST DEF", (best_def["bbox"][0], max(25, best_def["bbox"][1] - 10)), WHITE, 0.9, 2)

    # show last defender baseline x point
    def_x_baseline = x_at_y(vp, def_adv, y_base)
    if def_x_baseline is not None:
        cv2.circle(after, (int(def_x_baseline), y_base), 8, WHITE, -1)

    # draw attackers
    for pl, st, xv in statuses:
        adv = pl["advanced_point"]
        color = RED if st == "offside" else GREEN

        cv2.line(after, (adv[0], adv[1]), vp, color, 3)
        draw_bbox(after, pl["bbox"], color, 3)
        draw_point(after, adv, CYAN, 6)

        # intersection point on baseline
        cv2.circle(after, (int(xv), y_base), 7, color, -1)

        draw_text(after, st.upper(), (pl["bbox"][0], max(25, pl["bbox"][1] - 10)), color, 0.9, 2)

    draw_text(after, f"Attack: {attacking_team}", (30, 45), YELLOW, 1.0, 3)
    draw_text(after, f"Direction: {direction}", (30, 85), YELLOW, 1.0, 3)

    cv2.imwrite(str(OUTPUT_DIR / f"{base}_02_after_filter.jpg"), after)

    # ============================================================
    # IMAGE 03: FINAL LABELED (clean final output)
    # ============================================================
    final = img.copy()

    # baseline + VP
    draw_point(final, vp, RED, 10)
    cv2.line(final, (0, y_base), (w - 1, y_base), WHITE, 2)

    # last defender
    draw_bbox(final, best_def["bbox"], WHITE, 4)
    draw_text(final, "LAST DEFENDER", (best_def["bbox"][0], max(25, best_def["bbox"][1] - 10)), WHITE, 0.9, 2)
    draw_point(final, best_def["advanced_point"], CYAN, 6)

    # attackers with labels
    for pl, st, xv in statuses:
        bbox = pl["bbox"]
        adv = pl["advanced_point"]

        color = RED if st == "offside" else GREEN

        draw_bbox(final, bbox, color, 3)
        draw_point(final, adv, CYAN, 6)

        # label on top
        draw_text(final, st.upper(), (bbox[0], max(25, bbox[1] - 10)), color, 0.95, 2)

        # show baseline intersection
        cv2.circle(final, (int(xv), y_base), 7, color, -1)

    draw_text(final, f"ATTACKING TEAM: {attacking_team}", (30, 45), YELLOW, 1.0, 3)
    draw_text(final, f"DEFENDING TEAM: {defending_team}", (30, 85), YELLOW, 1.0, 3)

    cv2.imwrite(str(OUTPUT_DIR / f"{base}_03_final_labeled.jpg"), final)

    # ============================================================
    # JSON OUTPUT
    # ============================================================
    out_json = OUTPUT_DIR / f"{base}_offside_labeled.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] {base} -> saved before/after/final + json")


# ============================================================
#                        RUN ALL
# ============================================================

def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(INPUT_DIR.glob("*_teamdata.json"))
    if len(json_files) == 0:
        print("[WARN] No *_teamdata.json found in outputs_team_separation_seg")
        return

    for jf in json_files:
        process_play(jf)

    print("[FINISHED]")


if __name__ == "__main__":
    run_all()