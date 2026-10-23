import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def order_corners(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def detect_document(image, min_area_ratio=0.1, max_side=1200):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR color image.")
    if not 0 < min_area_ratio < 1:
        raise ValueError("Minimum area ratio must be between 0 and 1.")
    if max_side < 32:
        raise ValueError("Maximum processing side must be at least 32 pixels.")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    work = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    work_height, work_width = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
    edges = cv2.Canny(enhanced, 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    image_area = work_height * work_width
    for source in (closed, threshold, cv2.bitwise_not(threshold)):
        contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area_ratio * image_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.015, 0.025, 0.035):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                corners = order_corners(polygon)
                if (np.any(corners <= 5) or np.any(corners[:, 0] >= work_width - 6)
                    or np.any(corners[:, 1] >= work_height - 6)):
                    continue
                area = cv2.contourArea(corners)
                if area < min_area_ratio * image_area:
                    continue
                if np.min(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)) < 12:
                    continue
                fit = min(area, contour_area) / max(area, contour_area)
                if fit < 0.85:
                    continue
                score = area / image_area * fit
                candidates.append((score, corners, area / image_area, fit))
    if not candidates:
        return None, edges
    score, corners, area_ratio, fit = max(candidates, key=lambda candidate: candidate[0])
    corners = corners * np.array([width / work_width, height / work_height], dtype=np.float32)
    return {
        "corners": corners.tolist(),
        "area_ratio": float(area_ratio),
        "contour_fit": float(fit),
        "ranking_score": float(score),
    }, edges


def rectify_document(image, corners):
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    width = max(2, round(max(np.linalg.norm(top_right - top_left),
                             np.linalg.norm(bottom_right - bottom_left))))
    height = max(2, round(max(np.linalg.norm(bottom_left - top_left),
                              np.linalg.norm(bottom_right - top_right))))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]), destination)
    return cv2.warpPerspective(image, transform, (width, height))


def synthetic_cases():
    random = np.random.default_rng(42)
    height, width = 600, 800
    plain = np.full((height, width, 3), 55, dtype=np.uint8)
    textured = np.clip(plain.astype(np.int16) + random.normal(0, 14, plain.shape), 0, 255).astype(np.uint8)
    perspective = np.array([[220, 70], [620, 130], [570, 530], [140, 470]], dtype=np.float32)
    rotated = cv2.boxPoints(((400, 300), (330, 430), 28))
    diamond = cv2.boxPoints(((400, 300), (310, 310), 45))
    for name, background, corners, brightness in (
        ("perspective", plain, perspective, 230),
        ("rotation", plain, rotated, 230),
        ("diamond", plain, diamond, 230),
        ("texture", textured, perspective, 230),
        ("dim", plain // 3, perspective, 80),
        ("uneven_light", textured, perspective, 230),
        ("low_contrast", plain, perspective, 85),
    ):
        image = background.copy()
        cv2.fillConvexPoly(image, corners.astype(np.int32), (brightness,) * 3)
        if name == "uneven_light":
            lighting = np.linspace(0.3, 1.0, width)[None, :, None]
            image = (image * lighting).astype(np.uint8)
        yield name, image, order_corners(corners)
    yield "empty", plain.copy(), None
    yield "noise", textured.copy(), None
    circle = plain.copy()
    cv2.circle(circle, (400, 300), 180, (230, 230, 230), -1)
    yield "circle", circle, None


def write_image(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def process_image(image, output, min_area_ratio):
    output.mkdir(parents=True, exist_ok=True)
    candidate, edges = detect_document(image, min_area_ratio=min_area_ratio)
    overlay = image.copy()
    crop_path = output / "crop.png"
    if crop_path.exists():
        crop_path.unlink()
    if candidate is not None:
        corners = np.array(candidate["corners"], dtype=np.float32)
        cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True, (0, 200, 0), 3)
        for index, point in enumerate(corners):
            position = tuple(np.rint(point).astype(int))
            cv2.circle(overlay, position, 6, (0, 0, 255), -1)
            cv2.putText(overlay, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        write_image(crop_path, rectify_document(image, corners))
    result = {
        "document_candidate_found": candidate is not None,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "candidate": candidate,
    }
    write_image(output / "overlay.png", overlay)
    write_image(output / "edges.png", edges)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Classical OpenCV document-boundary experiment.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Image file or directory of images.")
    source.add_argument("--demo", action="store_true", help="Generate and process synthetic examples.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--min-area", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.min_area < 1:
        parser.error("--min-area must be between 0 and 1")
    if args.demo:
        for name, image, corners in synthetic_cases():
            output = args.output / name
            result = process_image(image, output, args.min_area)
            write_image(output / "input.png", image)
            (output / "ground_truth.json").write_text(
                json.dumps({"corners": None if corners is None else corners.tolist()}, indent=2),
                encoding="utf-8")
            print(f"{name}: candidate_found={result['document_candidate_found']}")
        return
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("No supported images found in the directory")
    failures = 0
    for path in paths:
        try:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image")
            result = process_image(image, args.output / path.name, args.min_area)
            print(f"{path.name}: candidate_found={result['document_candidate_found']}")
        except (OSError, ValueError, cv2.error) as error:
            failures += 1
            print(f"{path.name}: ERROR: {error}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()