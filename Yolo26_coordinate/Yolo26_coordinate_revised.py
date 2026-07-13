#!/usr/bin/env python3
r"""
Yolo26_coordinate_revised.py

Revised YOLO detection script with:
  1) recursive input scan protection, so generated *_view_*.jpg files are not reprocessed
  2) class-specific confidence thresholds, e.g. --class_conf D00=0.15,D10=0.25,D20=0.25
  3) cleaner professional output images with smaller labels/captions
  4) optional TrueType font support, e.g. Times New Roman on Windows
  5) GeoJSON + per-class CSV outputs retained from the original script

Example:
python Yolo26_coordinate_revised.py ^
  --images "E:\NZ_March26\20260318_2107\cam2" ^
  --out "E:\NZ_March26\20260318_2107\detection_outputs\cam2_styled" ^
  --csv "E:\NZ_March26\20260318_2107\precise_cam2_coords.csv" ^
  --model "E:\NZ_March26\20260318_2107\yolo26-fined-tuned-main\yolo26-fined-tuned-main\weights\YOLO26s_RDD_FRDC_Distilled_v2.pt" ^
  --conf 0.10 ^
  --class_conf D00=0.15,D10=0.25,D20=0.25 ^
  --imgsz 1280 ^
  --workers 1 ^
  --device 0 ^
  --caption_font_size 18 ^
  --label_font_size 16 ^
  --box_thickness 2 ^
  --font_path "C:\Windows\Fonts\times.ttf"
"""

import os
import json
import argparse
import multiprocessing as mp
import csv
import re
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageFont = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --------- Globals per worker (set in init_worker) ----------
G_MODEL = None
G_MODEL_NAMES = None
G_GEO_LOOKUP = None
G_ARGS = None
G_IMAGES_ROOT = None


# ----------------------------
# Helpers: files/folders
# ----------------------------
def iter_images_recursive(root_folder: str):
    """
    Recursively scan the image folder, but skip common output folders and generated *_view_*.jpg files.
    This prevents generated composite images from being processed again.
    """
    skip_dirs = {
        "out", "out1", "out2", "out3",
        "output", "outputs",
        "detection_output", "detection_outputs",
        "results", "runs",
        "yolo_out", "yolo_outputs",
    }

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Prevent recursive scan into output folders.
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]

        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            stem = os.path.splitext(fn)[0]

            if ext not in IMG_EXTS:
                continue

            # Skip generated composite/zoom output files.
            if "_view_" in stem.lower():
                continue

            yield os.path.join(dirpath, fn)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_resize_to_width(img: np.ndarray, max_width: int):
    if img is None:
        return None
    h, w = img.shape[:2]
    if max_width is None or max_width <= 0 or w <= max_width:
        return img
    scale = max_width / float(w)
    new_w = max_width
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_same_height(left: np.ndarray, right: np.ndarray):
    hl = left.shape[0]
    hr = right.shape[0]
    if hl == hr:
        return left, right
    if hl > hr:
        diff = hl - hr
        right = cv2.copyMakeBorder(right, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        diff = hr - hl
        left = cv2.copyMakeBorder(left, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return left, right


def slugify(name: str) -> str:
    """Make safe filenames from class names."""
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    if not s:
        s = "unknown"
    return s


# ----------------------------
# Helpers: professional drawing
# ----------------------------
def load_ttf_font(size: int, font_path: str = None):
    """
    Loads a professional TrueType font if Pillow is available.
    On Windows, Times New Roman is usually: C:\\Windows\\Fonts\\times.ttf
    """
    if not PIL_AVAILABLE:
        return None

    candidates = []
    if font_path:
        candidates.append(font_path)

    candidates.extend([
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ])

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_text_cv2_fallback(img_bgr, text, xy, font_size=20, color=(255, 255, 255), bg_color=None, padding=6):
    """Fallback if Pillow is not installed. Uses OpenCV Hershey font."""
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, font_size / 32.0)
    thickness = max(1, int(round(font_size / 18.0)))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if bg_color is not None:
        # bg_color is RGB-style; convert to BGR for cv2.
        bg_bgr = (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))
        cv2.rectangle(
            img_bgr,
            (x - padding, y - padding),
            (x + tw + padding, y + th + baseline + padding),
            bg_bgr,
            -1,
        )

    # color is RGB-style; convert to BGR for cv2.
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    cv2.putText(img_bgr, text, (x, y + th), font, scale, bgr, thickness, cv2.LINE_AA)
    return img_bgr


def draw_text_pil(
    img_bgr,
    text,
    xy,
    font_size=20,
    color=(255, 255, 255),
    bg_color=None,
    font_path=None,
    padding=6,
):
    """
    Draw clean TrueType text on an OpenCV BGR image.
    color/bg_color are RGB-style tuples.
    """
    if not PIL_AVAILABLE:
        return draw_text_cv2_fallback(img_bgr, text, xy, font_size, color, bg_color, padding)

    font = load_ttf_font(font_size, font_path)
    if font is None:
        return draw_text_cv2_fallback(img_bgr, text, xy, font_size, color, bg_color, padding)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)

    if bg_color is not None:
        draw.rectangle(
            [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ],
            fill=bg_color,
        )

    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_caption_clean(width, lines, font_size=18, line_height=26, font_path=None, align="left"):
    """Small professional black caption panel."""
    lines = [str(x) for x in lines if x is not None and str(x).strip() != ""]
    if not lines:
        lines = [""]

    caption_height = max(1, line_height * len(lines) + 14)
    cap = np.zeros((caption_height, width, 3), dtype=np.uint8)

    y = 8
    for line in lines:
        x = 14
        if align == "center":
            # Estimate center using PIL if available, otherwise keep left.
            if PIL_AVAILABLE:
                try:
                    font = load_ttf_font(font_size, font_path)
                    tmp = Image.new("RGB", (width, caption_height))
                    d = ImageDraw.Draw(tmp)
                    bbox = d.textbbox((0, 0), line, font=font)
                    tw = bbox[2] - bbox[0]
                    x = max(0, (width - tw) // 2)
                except Exception:
                    x = 14

        cap = draw_text_pil(
            cap,
            line,
            (x, y),
            font_size=font_size,
            color=(255, 255, 255),
            bg_color=None,
            font_path=font_path,
            padding=4,
        )
        y += line_height

    return cap


def draw_professional_box(img_bgr, x1, y1, x2, y2, label, font_size=16, font_path=None, thickness=2):
    """Draw a compact bounding box and compact label."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    # BGR color for the box. Chosen to be visible on road surfaces.
    box_color = (255, 180, 0)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, thickness)

    label_y = max(6, y1 - font_size - 10)
    img_bgr = draw_text_pil(
        img_bgr,
        label,
        (x1, label_y),
        font_size=font_size,
        color=(255, 255, 255),
        bg_color=(0, 0, 0),
        font_path=font_path,
        padding=5,
    )
    return img_bgr


# ----------------------------
# Helpers: CSV/Geo values
# ----------------------------
def _none_if_nan(v):
    """Convert pandas/numpy NaN values to None so CSV/JSON output is clean."""
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _as_float(v):
    """Best-effort numeric conversion. Returns None for missing/blank/NaN values."""
    v = _none_if_nan(v)
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt_float(v, decimals=3):
    v = _as_float(v)
    if v is None:
        return "NA"
    return f"{v:.{decimals}f}"


def _lookup_keys_for_name(name: str):
    """
    Build tolerant lookup keys:
      - exact name
      - basename only
      - lower-case variants
      - stem variants without extension
    """
    if not name:
        return []
    name = str(name).strip()
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    keys = [name, base, stem, name.lower(), base.lower(), stem.lower()]
    out = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def load_geo_csv(csv_path: str):
    """
    Loads the camera coordinate CSV.

    New expected columns:
      camera_name, lat, lon, ht, yaw, pitch, roll,
      std_east, std_north, std_ht, std_roll, std_pitch, std_yaw

    Backward compatibility:
      image, easting, northing, height/ht
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    col_by_lower = {c.lower(): c for c in df.columns}

    # New CSV uses camera_name. Old CSV used image.
    if "camera_name" in col_by_lower:
        image_col = col_by_lower["camera_name"]
    elif "image" in col_by_lower:
        image_col = col_by_lower["image"]
    else:
        raise ValueError(
            "CSV must contain a 'camera_name' column (new format) or an 'image' column (old format). "
            f"Found: {list(df.columns)}"
        )

    aliases = {
        "camera_name": ["camera_name", "image"],
        "lat": ["lat", "latitude"],
        "lon": ["lon", "longitude", "lng"],
        "ht": ["ht", "height", "alt", "altitude"],
        "yaw": ["yaw"],
        "pitch": ["pitch"],
        "roll": ["roll"],
        "std_east": ["std_east", "std_easting"],
        "std_north": ["std_north", "std_northing"],
        "std_ht": ["std_ht", "std_height"],
        "std_roll": ["std_roll"],
        "std_pitch": ["std_pitch"],
        "std_yaw": ["std_yaw"],
        "easting": ["easting", "east", "x"],
        "northing": ["northing", "north", "y"],
    }

    def get_raw(row, canonical_name):
        for alias in aliases[canonical_name]:
            c = col_by_lower.get(alias.lower())
            if c is not None:
                return row.get(c)
        return None

    lookup = {}
    for _, r in df.iterrows():
        camera_name = str(r.get(image_col, "")).strip()
        if not camera_name:
            continue

        rec = {"camera_name": camera_name}

        for f in [
            "lat", "lon", "ht", "yaw", "pitch", "roll",
            "std_east", "std_north", "std_ht", "std_roll", "std_pitch", "std_yaw",
            "easting", "northing",
        ]:
            rec[f] = _as_float(get_raw(r, f))

        # Preserve any other CSV fields too, but convert NaN to None.
        for c in df.columns:
            if c not in rec:
                rec[c] = _none_if_nan(r.get(c))

        for key in _lookup_keys_for_name(camera_name):
            if key not in lookup:
                lookup[key] = rec

    return lookup


def lookup_geo_row(lookup: dict, image_path_or_name: str):
    """Find a coordinate row for an image using exact/basename/stem/lowercase keys."""
    for key in _lookup_keys_for_name(image_path_or_name):
        row = lookup.get(key)
        if row is not None:
            return row
    return None


def build_geojson_geometry(row, x1, y1, x2, y2):
    """
    Prefer real camera location if lat/lon are available.
    If only old easting/northing exist, use them as XY point coordinates.
    Otherwise fall back to the original pixel polygon.
    """
    if row is not None:
        lat = _as_float(row.get("lat"))
        lon = _as_float(row.get("lon"))
        ht = _as_float(row.get("ht"))
        if lat is not None and lon is not None:
            coords = [lon, lat] if ht is None else [lon, lat, ht]
            return {"type": "Point", "coordinates": coords}, "camera_lat_lon"

        easting = _as_float(row.get("easting"))
        northing = _as_float(row.get("northing"))
        if easting is not None and northing is not None:
            coords = [easting, northing] if ht is None else [easting, northing, ht]
            return {"type": "Point", "coordinates": coords}, "camera_easting_northing"

    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }, "pixel_bbox_polygon"


def geo_output_fields():
    return [
        "lat", "lon", "ht", "yaw", "pitch", "roll",
        "std_east", "std_north", "std_ht", "std_roll", "std_pitch", "std_yaw",
        "easting", "northing",
    ]


def geo_props_from_row(row):
    props = {}
    for f in geo_output_fields():
        props[f] = None if row is None else _as_float(row.get(f))
    return props


def compact_geo_lines(filename, row, geo_vals):
    """Create compact metadata lines for the output image."""
    if row is not None and geo_vals.get("lat") is not None and geo_vals.get("lon") is not None:
        return [
            filename,
            f"lat {_fmt_float(geo_vals.get('lat'), 7)}   lon {_fmt_float(geo_vals.get('lon'), 7)}   ht {_fmt_float(geo_vals.get('ht'), 2)} m",
            f"yaw {_fmt_float(geo_vals.get('yaw'), 1)}   pitch {_fmt_float(geo_vals.get('pitch'), 1)}   roll {_fmt_float(geo_vals.get('roll'), 1)}",
        ]

    if row is not None and geo_vals.get("easting") is not None and geo_vals.get("northing") is not None:
        return [
            filename,
            f"E {_fmt_float(geo_vals.get('easting'), 2)}   N {_fmt_float(geo_vals.get('northing'), 2)}   ht {_fmt_float(geo_vals.get('ht'), 2)} m",
        ]

    return [filename, "camera coordinate row not found"]


# ----------------------------
# Helpers: class confidence
# ----------------------------
def parse_class_conf(text: str):
    """
    Parses class-specific confidence thresholds.

    Example:
      --class_conf D00=0.15,D10=0.25,D20=0.25

    Matching works by:
      - exact class name, e.g. "Transverse Crack (D10)"
      - defect code inside name, e.g. "D10"
      - class id as string, e.g. "0"
    """
    out = {}

    if not text:
        return out

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            raise ValueError(
                f"Invalid --class_conf item '{part}'. Use format like D10=0.25,D00=0.18"
            )

        key, val = part.split("=", 1)
        key = key.strip()
        val = float(val.strip())

        if key:
            out[key.lower()] = val

    return out


def model_class_name(names, class_id):
    """Robustly read class name from Ultralytics names dict/list."""
    try:
        if isinstance(names, dict):
            return names.get(class_id, str(class_id))
        if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
            return names[int(class_id)]
    except Exception:
        pass
    return str(class_id)


def get_defect_code(class_name):
    """Extract D-code such as D00, D10, D20 from a class name."""
    m = re.search(r"\((D\d+)\)", str(class_name), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(D\d+)\b", str(class_name), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def get_class_min_conf(class_id, class_name, default_conf, class_conf_map):
    """Returns the minimum confidence for this class."""
    cname = str(class_name).lower()
    cid = str(class_id).lower()

    # Match class id.
    if cid in class_conf_map:
        return class_conf_map[cid]

    # Match exact/lower class name.
    if cname in class_conf_map:
        return class_conf_map[cname]

    # Match D-code inside class name.
    code = get_defect_code(class_name)
    if code and code.lower() in class_conf_map:
        return class_conf_map[code.lower()]

    return default_conf


# ----------------------------
# Worker init / worker function
# ----------------------------
def init_worker(model_path: str, csv_path: str, args_dict: dict, images_root: str):
    global G_MODEL, G_MODEL_NAMES, G_GEO_LOOKUP, G_ARGS, G_IMAGES_ROOT
    G_ARGS = args_dict
    G_IMAGES_ROOT = images_root

    # Avoid OpenCV oversubscription per process.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    # Load geo lookup once per process.
    G_GEO_LOOKUP = load_geo_csv(csv_path)

    # Load YOLO once per process.
    G_MODEL = YOLO(model_path)
    G_MODEL_NAMES = G_MODEL.names


def process_one_image(img_path: str):
    """
    Runs inference + saves view images for each detection.
    Returns:
      - status + counts
      - geojson features (list)
      - class rows (list of dicts): per detection row with image/camera pose/class_name/conf
    """
    global G_MODEL, G_MODEL_NAMES, G_GEO_LOOKUP, G_ARGS, G_IMAGES_ROOT

    out_dir = G_ARGS["out"]
    conf_th = G_ARGS["conf"]
    class_conf_map = G_ARGS.get("class_conf_map", {})
    max_width = G_ARGS["max_width"]
    crop_size = G_ARGS["crop_size"]
    device = G_ARGS["device"]
    imgsz = G_ARGS["imgsz"]
    verbose = G_ARGS["verbose"]

    font_path = G_ARGS.get("font_path")
    caption_font_size = G_ARGS.get("caption_font_size", 18)
    label_font_size = G_ARGS.get("label_font_size", 16)
    box_thickness = G_ARGS.get("box_thickness", 2)
    hide_overview_labels = G_ARGS.get("hide_overview_labels", False)

    rel = os.path.relpath(img_path, G_IMAGES_ROOT)

    img = cv2.imread(img_path)
    if img is None:
        return {"status": "READ_FAIL", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    # Inference.
    try:
        # Use the lowest threshold for inference, then apply class-specific filtering after prediction.
        all_thresholds = [conf_th] + list(class_conf_map.values())
        predict_conf = min(all_thresholds) if all_thresholds else conf_th

        kwargs = dict(source=img, conf=predict_conf, device=device, verbose=False)
        if imgsz and imgsz > 0:
            kwargs["imgsz"] = imgsz
        results = G_MODEL.predict(**kwargs)
    except Exception as e:
        return {"status": f"PREDICT_ERR: {e}", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {"status": "NO_DET", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    r0 = results[0]
    boxes = r0.boxes

    # Apply class-specific filtering.
    kept = []
    for box in boxes:
        cls = int(box.cls[0])
        class_name = model_class_name(G_MODEL_NAMES, cls)
        conf = float(box.conf[0])
        min_conf = get_class_min_conf(cls, class_name, conf_th, class_conf_map)

        if conf < min_conf:
            if verbose:
                print(f"[FILTER] {os.path.basename(img_path)} | {class_name} | {conf:.3f} < {min_conf:.3f}")
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        kept.append({
            "box": box,
            "cls": cls,
            "class_name": class_name,
            "conf": conf,
            "min_conf": min_conf,
            "xyxy": (x1, y1, x2, y2),
        })

    if len(kept) == 0:
        return {"status": "NO_DET", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    det_count = len(kept)

    # Draw our own cleaner boxes instead of using r0.plot().
    img_with_boxes = img.copy()

    for k, det in enumerate(kept):
        x1, y1, x2, y2 = det["xyxy"]
        class_name = det["class_name"]
        conf = det["conf"]
        code = get_defect_code(class_name) or class_name

        short_label = f"Box {k + 1} | {code} | {conf:.2f}"
        label_to_draw = "" if hide_overview_labels else short_label

        # Draw box always. Draw label only when enabled.
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 180, 0), box_thickness)
        if label_to_draw:
            img_with_boxes = draw_text_pil(
                img_with_boxes,
                label_to_draw,
                (x1, max(6, y1 - label_font_size - 10)),
                font_size=label_font_size,
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_path=font_path,
                padding=5,
            )

    img_resized = safe_resize_to_width(img_with_boxes, max_width)
    if img_resized is None:
        return {"status": "PLOT_FAIL", "rel": rel, "det": det_count, "written": 0, "features": [], "rows": []}

    filename = os.path.basename(img_path)

    # Camera pose/position from CSV, matched by camera_name / basename.
    row = lookup_geo_row(G_GEO_LOOKUP, filename)
    geo_vals = geo_props_from_row(row)

    lines = compact_geo_lines(filename, row, geo_vals)

    caption = draw_caption_clean(
        img_resized.shape[1],
        lines,
        font_size=caption_font_size,
        line_height=max(24, caption_font_size + 8),
        font_path=font_path,
        align="left",
    )
    img_labeled = np.vstack((img_resized, caption))

    h0, w0 = img.shape[:2]

    features = []
    rows = []
    written = 0

    # If scanning subfolders, basenames can collide. Keep outputs unique using rel path.
    rel_no_ext = os.path.splitext(rel)[0]
    safe_rel = rel_no_ext.replace("\\", "__").replace("/", "__").replace(":", "")

    for i, det in enumerate(kept):
        box = det["box"]
        x1, y1, x2, y2 = det["xyxy"]
        cls = det["cls"]
        class_name = det["class_name"]
        conf = det["conf"]
        min_conf = det["min_conf"]

        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        geometry, geometry_source = build_geojson_geometry(row, x1, y1, x2, y2)

        props = {
            "image_name": filename,
            "camera_name": None if row is None else row.get("camera_name"),
            "image_path": os.path.abspath(img_path),
            "box_id": i + 1,
            "class_id": cls,
            "class_name": class_name,
            "confidence": conf,
            "min_conf_used": min_conf,
            "geometry_source": geometry_source,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_cx": bbox_cx,
            "bbox_cy": bbox_cy,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }
        props.update(geo_vals)

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        })

        out_row = {
            "image": filename,
            "camera_name": None if row is None else row.get("camera_name"),
            "image_path": os.path.abspath(img_path),
            "class_id": cls,
            "class_name": class_name,
            "confidence": conf,
            "min_conf_used": min_conf,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_cx": bbox_cx,
            "bbox_cy": bbox_cy,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }
        out_row.update(geo_vals)
        rows.append(out_row)

        # Clamp for safe crop.
        x1c = max(0, min(w0 - 1, x1))
        x2c = max(0, min(w0, x2))
        y1c = max(0, min(h0 - 1, y1))
        y2c = max(0, min(h0, y2))
        if x2c <= x1c or y2c <= y1c:
            continue

        cropped = img[y1c:y2c, x1c:x2c].copy()
        if cropped.size == 0:
            continue

        try:
            crop_resized = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        except Exception:
            continue

        # Add a thin border around the crop.
        cv2.rectangle(crop_resized, (0, 0), (crop_size - 1, crop_size - 1), (255, 180, 0), max(1, box_thickness))

        # Smaller, cleaner one-line crop caption. Avoid duplicate coordinate text.
        crop_caption_lines = [f"Box {i + 1} | {class_name} | conf {conf:.2f}"]
        id_caption = draw_caption_clean(
            crop_resized.shape[1],
            crop_caption_lines,
            font_size=caption_font_size,
            line_height=max(24, caption_font_size + 8),
            font_path=font_path,
            align="center",
        )

        crop_labeled = np.vstack((crop_resized, id_caption))

        left, right = pad_to_same_height(img_labeled, crop_labeled)
        try:
            combined = np.hstack((left, right))
        except Exception:
            continue

        out_path = os.path.join(out_dir, f"{safe_rel}_view_{i + 1}.jpg")
        if cv2.imwrite(out_path, combined):
            written += 1

    return {"status": "OK", "rel": rel, "det": det_count, "written": written, "features": features, "rows": rows}


# ----------------------------
# Streaming writers: parent only
# ----------------------------
def write_feature_jsonl(jsonl_fp, feature):
    jsonl_fp.write(json.dumps(feature, ensure_ascii=False) + "\n")


def jsonl_to_featurecollection(jsonl_path: str, out_geojson_path: str):
    with open(out_geojson_path, "w", encoding="utf-8") as out_f:
        out_f.write('{"type":"FeatureCollection","features":[\n')
        first = True
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        out_f.write(",\n")
                    out_f.write(line)
                    first = False
        out_f.write("\n]}\n")


def get_csv_writer_for_class(open_files, writers, out_dir: str, class_name: str, fieldnames):
    """
    Lazily open and return a csv.DictWriter for the given class.
    Keeps file handles open for the duration of the run, parent process only.
    """
    key = class_name
    if key in writers:
        return writers[key]

    safe = slugify(class_name)
    csv_path = os.path.join(out_dir, f"{safe}.csv")
    f = open(csv_path, "a", newline="", encoding="utf-8")
    open_files[key] = f
    w = csv.DictWriter(f, fieldnames=fieldnames)

    # If file is empty, write header.
    if f.tell() == 0:
        w.writeheader()

    writers[key] = w
    return w


# ----------------------------
# Main
# ----------------------------
def main():
    mp.freeze_support()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(
        description="Fast parallel YOLO over recursive folders, outputs styled views + GeoJSON + per-class CSVs."
    )
    ap.add_argument("--images", required=True, help="Root folder containing images. Subfolders are scanned.")
    ap.add_argument("--model", required=True, help="Path to YOLO .pt model.")
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV path with camera_name,lat,lon,ht,yaw,pitch,roll,std_* fields. Legacy image/easting/northing/height also supported.",
    )
    ap.add_argument("--out", required=True, help="Output folder for *_view_#.jpg, GeoJSON, and per-class CSVs.")
    ap.add_argument("--conf", type=float, default=0.5, help="Default confidence threshold. Default: 0.5")
    ap.add_argument(
        "--class_conf",
        default="",
        help=(
            "Optional per-class confidence thresholds. "
            "Example: D00=0.15,D10=0.25,D20=0.25 or \"Transverse Crack (D10)=0.25\""
        ),
    )
    ap.add_argument("--max_width", type=int, default=800, help="Max width for left output preview. Default: 800")
    ap.add_argument("--crop_size", type=int, default=600, help="Crop resize square size. Default: 600")
    ap.add_argument("--geojson_name", default="detections.geojson", help="Final GeoJSON filename inside --out.")
    ap.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0. Default: cpu")
    ap.add_argument("--imgsz", type=int, default=0, help="Force inference image size, e.g. 1280. 0 = default.")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes. Use 1 on GPU unless you know multi-process GPU is stable.",
    )
    ap.add_argument("--chunksize", type=int, default=8, help="Multiprocessing chunksize. Default: 8")
    ap.add_argument("--verbose", action="store_true", help="Extra debug output.")

    # Styling options.
    ap.add_argument(
        "--font_path",
        default=r"C:\Windows\Fonts\times.ttf",
        help="Path to TrueType font for output labels/captions. Default: Windows Times New Roman path.",
    )
    ap.add_argument(
        "--caption_font_size",
        type=int,
        default=18,
        help="Font size for bottom metadata and crop captions. Default: 18",
    )
    ap.add_argument(
        "--label_font_size",
        type=int,
        default=16,
        help="Font size for compact box labels on overview image. Default: 16",
    )
    ap.add_argument(
        "--box_thickness",
        type=int,
        default=2,
        help="Bounding box line thickness. Default: 2",
    )
    ap.add_argument(
        "--hide_overview_labels",
        action="store_true",
        help="Draw boxes on the overview image but hide the small box labels. Useful for very clean figures.",
    )

    args = ap.parse_args()

    ensure_dir(args.out)

    class_conf_map = parse_class_conf(args.class_conf)

    img_list = list(iter_images_recursive(args.images))
    total = len(img_list)

    print(f"[INIT] images root : {args.images}")
    print(f"[INIT] found       : {total} images")
    print(f"[INIT] model       : {args.model}")
    print(f"[INIT] csv         : {args.csv}")
    print(f"[INIT] out         : {args.out}")
    print(f"[INIT] device      : {args.device}")
    print(f"[INIT] workers     : {args.workers}")
    print(f"[INIT] conf/imgsz  : {args.conf} / {args.imgsz if args.imgsz > 0 else 'default'}")
    print(f"[INIT] class_conf  : {class_conf_map if class_conf_map else 'none'}")
    print(f"[INIT] PIL font    : {'available' if PIL_AVAILABLE else 'not available; using cv2 fallback'}")
    print(f"[INIT] font_path   : {args.font_path}")

    if total == 0:
        print("[DONE] No images found.")
        return

    args_dict = {
        "out": args.out,
        "conf": args.conf,
        "class_conf_map": class_conf_map,
        "max_width": args.max_width,
        "crop_size": args.crop_size,
        "device": args.device,
        "imgsz": args.imgsz,
        "verbose": args.verbose,
        "font_path": args.font_path,
        "caption_font_size": args.caption_font_size,
        "label_font_size": args.label_font_size,
        "box_thickness": args.box_thickness,
        "hide_overview_labels": args.hide_overview_labels,
    }

    # Temp JSONL for features.
    jsonl_path = os.path.join(args.out, "_features.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    processed = 0
    ok_images = 0
    no_det = 0
    read_fail = 0
    predict_err = 0
    total_dets = 0
    total_views = 0

    # Per-class CSV writers, parent only.
    fieldnames = [
        "image", "camera_name", "image_path",
        "class_id", "class_name", "confidence", "min_conf_used",
        "lat", "lon", "ht", "yaw", "pitch", "roll",
        "std_east", "std_north", "std_ht", "std_roll", "std_pitch", "std_yaw",
        "easting", "northing",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    ]
    open_files = {}
    writers = {}

    ctx = mp.get_context("spawn")

    try:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jsonl_fp:
            with ctx.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(args.model, args.csv, args_dict, args.images),
            ) as pool:
                for res in pool.imap_unordered(process_one_image, img_list, chunksize=args.chunksize):
                    processed += 1
                    status = res["status"]
                    rel = res["rel"]
                    det = int(res.get("det", 0))
                    written = int(res.get("written", 0))

                    if status == "OK":
                        ok_images += 1
                        total_dets += det
                        total_views += written
                        print(f"[OK]   ({processed}/{total}) DET={det:3d} VIEWS={written:2d}  {rel}")
                    elif status == "NO_DET":
                        no_det += 1
                        print(f"[SKIP] ({processed}/{total}) NO_DET/FILTERED  {rel}")
                    elif status == "READ_FAIL":
                        read_fail += 1
                        print(f"[WARN] ({processed}/{total}) READ_FAIL       {rel}")
                    elif status.startswith("PREDICT_ERR"):
                        predict_err += 1
                        print(f"[ERR]  ({processed}/{total}) {status}  {rel}")
                    else:
                        print(f"[WARN] ({processed}/{total}) {status}  {rel}")

                    # Write GeoJSON features.
                    for f in res.get("features", []):
                        write_feature_jsonl(jsonl_fp, f)

                    # Write per-class CSV rows, one row per detection.
                    for row in res.get("rows", []):
                        cname = row.get("class_name", "unknown")
                        w = get_csv_writer_for_class(open_files, writers, args.out, cname, fieldnames)
                        w.writerow(row)

    finally:
        for f in open_files.values():
            try:
                f.close()
            except Exception:
                pass

    # Build final GeoJSON from JSONL.
    out_geojson_path = os.path.join(args.out, args.geojson_name)
    jsonl_to_featurecollection(jsonl_path, out_geojson_path)

    print("\n[SUMMARY]")
    print(f"  processed images     : {processed}")
    print(f"  ok images            : {ok_images}")
    print(f"  skipped/no det       : {no_det}")
    print(f"  read fail            : {read_fail}")
    print(f"  predict errors       : {predict_err}")
    print(f"  total detections     : {total_dets}")
    print(f"  view images written  : {total_views}")
    print(f"  geojson              : {out_geojson_path}")
    print(f"  per-class CSVs       : {args.out}/*.csv")
    print(f"  out folder           : {args.out}")


if __name__ == "__main__":
    main()