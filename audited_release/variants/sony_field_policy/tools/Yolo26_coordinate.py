#!/usr/bin/env python3
# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --------- Globals per worker (set in init_worker) ----------
G_MODEL = None
G_MODEL_NAMES = None
G_GEO_LOOKUP = None
G_ARGS = None
G_IMAGES_ROOT = None

# ----------------------------
# Helpers
# ----------------------------
def iter_images_recursive(root_folder: str):
    for dirpath, _, filenames in os.walk(root_folder):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                yield os.path.join(dirpath, fn)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_resize_to_width(img: np.ndarray, max_width: int):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    new_w = max_width
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_caption(width: int, lines, font, font_scale, thickness, color, line_height=40):
    caption_height = line_height * len(lines)
    cap = np.zeros((caption_height, width, 3), dtype=np.uint8)
    for j, line in enumerate(lines):
        ts = cv2.getTextSize(line, font, font_scale, thickness)[0]
        tx = max(0, (width - ts[0]) // 2)
        ty = line_height * (j + 1) - 10
        cv2.putText(cap, line, (tx, ty), font, font_scale, color, thickness)
    return cap

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
      camera_name,lat,lon,ht,yaw,pitch,roll,
      std_east,std_north,std_ht,std_roll,std_pitch,std_yaw

    Backward compatibility:
      image,easting,northing,height/ht

    Matching is done against the image basename, so a CSV camera_name like:
      Cam2_2026-03-19_08-30-53_capt0000_1.jpg
    will match an image path ending in that filename.
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

    # Canonical fields we want to carry into output. Values missing in old CSV become None.
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
        # Old projected/local coordinate support.
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

        rec = {}
        rec["camera_name"] = camera_name

        # Standard numeric fields.
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
            return {
                "type": "Point",
                "coordinates": coords,
            }, "camera_lat_lon"

        easting = _as_float(row.get("easting"))
        northing = _as_float(row.get("northing"))
        if easting is not None and northing is not None:
            coords = [easting, northing] if ht is None else [easting, northing, ht]
            return {
                "type": "Point",
                "coordinates": coords,
            }, "camera_easting_northing"

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

def slugify(name: str) -> str:
    """
    Make safe filenames like 'Long Crack' -> 'longCrack' or 'long_crack' (we'll use snake-ish).
    """
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    if not s:
        s = "unknown"
    return s

# ----------------------------
# Worker init / worker function
# ----------------------------
def init_worker(model_path: str, csv_path: str, args_dict: dict, images_root: str):
    global G_MODEL, G_MODEL_NAMES, G_GEO_LOOKUP, G_ARGS, G_IMAGES_ROOT
    G_ARGS = args_dict
    G_IMAGES_ROOT = images_root

    # avoid OpenCV oversubscription per process
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    # load geo lookup once per process
    G_GEO_LOOKUP = load_geo_csv(csv_path)

    # load YOLO once per process
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
    max_width = G_ARGS["max_width"]
    crop_size = G_ARGS["crop_size"]
    device = G_ARGS["device"]
    imgsz = G_ARGS["imgsz"]
    verbose = G_ARGS["verbose"]

    rel = os.path.relpath(img_path, G_IMAGES_ROOT)

    img = cv2.imread(img_path)
    if img is None:
        return {"status": "READ_FAIL", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    # Inference
    try:
        kwargs = dict(source=img, conf=conf_th, device=device, verbose=False)
        if imgsz and imgsz > 0:
            kwargs["imgsz"] = imgsz
        results = G_MODEL.predict(**kwargs)
    except Exception as e:
        return {"status": f"PREDICT_ERR: {e}", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {"status": "NO_DET", "rel": rel, "det": 0, "written": 0, "features": [], "rows": []}

    r0 = results[0]
    boxes = r0.boxes
    det_count = len(boxes)

    img_with_boxes = r0.plot()
    img_resized = safe_resize_to_width(img_with_boxes, max_width)
    if img_resized is None:
        return {"status": "PLOT_FAIL", "rel": rel, "det": det_count, "written": 0, "features": [], "rows": []}

    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    # Camera pose/position from CSV (matched by camera_name / basename)
    row = lookup_geo_row(G_GEO_LOOKUP, filename)
    geo_vals = geo_props_from_row(row)

    if row is not None and geo_vals.get("lat") is not None and geo_vals.get("lon") is not None:
        lines = [
            filename,
            f"lat: {_fmt_float(geo_vals.get('lat'), 10)}  lon: {_fmt_float(geo_vals.get('lon'), 10)}",
            f"ht: {_fmt_float(geo_vals.get('ht'), 3)}  yaw: {_fmt_float(geo_vals.get('yaw'), 3)}",
            f"pitch: {_fmt_float(geo_vals.get('pitch'), 3)}  roll: {_fmt_float(geo_vals.get('roll'), 3)}",
        ]
    elif row is not None and geo_vals.get("easting") is not None and geo_vals.get("northing") is not None:
        lines = [
            filename,
            f"easting: {_fmt_float(geo_vals.get('easting'), 2)}",
            f"northing: {_fmt_float(geo_vals.get('northing'), 2)}",
            f"ht: {_fmt_float(geo_vals.get('ht'), 2)}",
        ]
    else:
        lines = [filename, "camera coordinate row not found"]

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_THICKNESS = 2
    FONT_COLOR = (255, 255, 255)

    caption = draw_caption(img_resized.shape[1], lines, FONT, FONT_SCALE, FONT_THICKNESS, FONT_COLOR)
    img_labeled = np.vstack((img_resized, caption))

    h0, w0 = img.shape[:2]

    features = []
    rows = []
    written = 0

    # IMPORTANT: If scanning subfolders, basenames can collide. Keep outputs unique using rel path.
    rel_no_ext = os.path.splitext(rel)[0]
    safe_rel = rel_no_ext.replace("\\", "__").replace("/", "__").replace(":", "")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        class_name = G_MODEL_NAMES.get(cls, str(cls))
        conf = float(box.conf[0])

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
            "class_name": class_name,
            "confidence": conf,
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

        # GeoJSON feature:
        # - Point at camera lon/lat/ht when new CSV coordinates exist
        # - Point at old easting/northing/ht for legacy CSVs
        # - Pixel bbox polygon only when no coordinates are available
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        })

        # Per-class CSV row (one row per detection)
        out_row = {
            "image": filename,
            "camera_name": None if row is None else row.get("camera_name"),
            "image_path": os.path.abspath(img_path),
            "class_name": class_name,
            "confidence": conf,
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

        # Clamp for safe crop
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

        # Small caption under crop
        id_caption_height = 90
        id_caption = np.zeros((id_caption_height, crop_resized.shape[1], 3), dtype=np.uint8)
        labels = [f"Box {i+1}", f"Class: {class_name}"]
        y_offset = 30
        for label in labels:
            ts = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            tx = max(0, (id_caption.shape[1] - ts[0]) // 2)
            cv2.putText(id_caption, label, (tx, y_offset), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y_offset += 30

        crop_labeled = np.vstack((crop_resized, id_caption))

        left, right = pad_to_same_height(img_labeled, crop_labeled)
        try:
            combined = np.hstack((left, right))
        except Exception:
            continue

        out_path = os.path.join(out_dir, f"{safe_rel}_view_{i+1}.jpg")
        if cv2.imwrite(out_path, combined):
            written += 1

    return {"status": "OK", "rel": rel, "det": det_count, "written": written, "features": features, "rows": rows}

# ----------------------------
# Streaming writers (parent only)
# ----------------------------
def write_feature_jsonl(jsonl_fp, feature):
    jsonl_fp.write(json.dumps(feature, ensure_ascii=False) + "\n")

def jsonl_to_featurecollection(jsonl_path: str, out_geojson_path: str):
    with open(out_geojson_path, "w", encoding="utf-8") as out_f:
        out_f.write('{"type":"FeatureCollection","features":[\n')
        first = True
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
    Keeps file handles open for the duration of the run (parent process only).
    """
    key = class_name
    if key in writers:
        return writers[key]

    safe = slugify(class_name)
    csv_path = os.path.join(out_dir, f"{safe}.csv")
    f = open(csv_path, "a", newline="", encoding="utf-8")
    open_files[key] = f
    w = csv.DictWriter(f, fieldnames=fieldnames)

    # If file is empty, write header
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

    ap = argparse.ArgumentParser(description="Fast parallel YOLO over recursive folders, outputs views + GeoJSON + per-class CSVs.")
    ap.add_argument("--images", required=True, help="Root folder containing images (scan subfolders).")
    ap.add_argument("--model", required=True, help="Path to YOLO .pt model.")
    ap.add_argument("--csv", required=True, help="CSV path with camera_name,lat,lon,ht,yaw,pitch,roll,std_* fields. Legacy image/easting/northing/height also supported.")
    ap.add_argument("--out", required=True, help="Output folder for *_view_#.jpg, GeoJSON, and per-class CSVs.")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold. Default: 0.5")
    ap.add_argument("--max_width", type=int, default=800, help="Max width for output preview. Default: 800")
    ap.add_argument("--crop_size", type=int, default=600, help="Crop resize square. Default: 600")
    ap.add_argument("--geojson_name", default="detections.geojson", help="Final GeoJSON filename inside --out.")
    ap.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0. Default: cpu")
    ap.add_argument("--imgsz", type=int, default=0, help="Force inference image size (e.g. 1280). 0 = default.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="Parallel worker processes.")
    ap.add_argument("--chunksize", type=int, default=8, help="Multiprocessing chunksize. Tune for speed. Default 8")
    ap.add_argument("--verbose", action="store_true", help="Extra debug output.")
    args = ap.parse_args()

    ensure_dir(args.out)

    img_list = list(iter_images_recursive(args.images))
    total = len(img_list)
    print(f"[INIT] images root : {args.images}")
    print(f"[INIT] found       : {total} images")
    print(f"[INIT] model       : {args.model}")
    print(f"[INIT] csv         : {args.csv}")
    print(f"[INIT] out         : {args.out}")
    print(f"[INIT] device      : {args.device}")
    print(f"[INIT] workers     : {args.workers}")
    print(f"[INIT] conf/imgsz  : {args.conf} / {args.imgsz if args.imgsz>0 else 'default'}")

    if total == 0:
        print("[DONE] No images found.")
        return

    args_dict = {
        "out": args.out,
        "conf": args.conf,
        "max_width": args.max_width,
        "crop_size": args.crop_size,
        "device": args.device,
        "imgsz": args.imgsz,
        "verbose": args.verbose
    }

    # Temp JSONL for features
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

    # Per-class CSV writers (parent-only)
    fieldnames = [
        "image", "camera_name", "image_path", "class_name", "confidence",
        "lat", "lon", "ht", "yaw", "pitch", "roll",
        "std_east", "std_north", "std_ht", "std_roll", "std_pitch", "std_yaw",
        "easting", "northing",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    ]
    open_files = {}   # class_name -> file handle
    writers = {}      # class_name -> csv.DictWriter

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
                        print(f"[SKIP] ({processed}/{total}) NO_DET            {rel}")
                    elif status == "READ_FAIL":
                        read_fail += 1
                        print(f"[WARN] ({processed}/{total}) READ_FAIL          {rel}")
                    elif status.startswith("PREDICT_ERR"):
                        predict_err += 1
                        print(f"[ERR]  ({processed}/{total}) {status}  {rel}")
                    else:
                        print(f"[WARN] ({processed}/{total}) {status}  {rel}")

                    # Write GeoJSON features
                    for f in res.get("features", []):
                        write_feature_jsonl(jsonl_fp, f)

                    # Write per-class CSV rows (one row per detection)
                    for row in res.get("rows", []):
                        cname = row.get("class_name", "unknown")
                        w = get_csv_writer_for_class(open_files, writers, args.out, cname, fieldnames)
                        w.writerow(row)

    finally:
        # Close all csv files
        for f in open_files.values():
            try:
                f.close()
            except Exception:
                pass

    # Build final GeoJSON from JSONL
    out_geojson_path = os.path.join(args.out, args.geojson_name)
    jsonl_to_featurecollection(jsonl_path, out_geojson_path)

    print("\n[SUMMARY]")
    print(f"  processed images     : {processed}")
    print(f"  ok images            : {ok_images}")
    print(f"  skipped (no det)     : {no_det}")
    print(f"  read fail            : {read_fail}")
    print(f"  predict errors       : {predict_err}")
    print(f"  total detections     : {total_dets}")
    print(f"  view images written  : {total_views}")
    print(f"  geojson              : {out_geojson_path}")
    print(f"  per-class CSVs       : {args.out}/*.csv")
    print(f"  out folder           : {args.out}")

if __name__ == "__main__":
    main()
