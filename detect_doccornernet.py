import argparse
import hashlib
import importlib.util
import json
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detect_document import rectify_document, write_image


UPSTREAM_REVISION = "d877dc3"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "doccornernet"


def prepare_input(image, size=224):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(((values - mean) / std)[None])


def decode_prediction(output, image_shape, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size != 9 or not np.all(np.isfinite(values)):
        raise ValueError("Expected nine finite values: eight coordinates and a probability.")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Exported coordinates and probability must be in [0, 1].")
    height, width = image_shape[:2]
    corners = values[:8].reshape(4, 2) * np.array([width, height], dtype=np.float32)
    corners = np.minimum(corners, [width - 1, height - 1]).astype(np.float32)
    geometry_valid = bool(cv2.isContourConvex(corners) and cv2.contourArea(corners) >= 4)
    score = float(values[8])
    return {
        "document_candidate_found": score >= threshold and geometry_valid,
        "presence_score": score,
        "threshold": threshold,
        "geometry_valid": geometry_valid,
        "corners": corners.tolist(),
        "normalized_corners": values[:8].reshape(4, 2).tolist(),
        "image_size": {"width": width, "height": height},
    }


def prepare_model():
    import tensorflow as tf

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.github.com/repos/mapo80/DocCornerNet-CoordClass/commits/{UPSTREAM_REVISION}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "document-detection-poc"})
    with urllib.request.urlopen(request, timeout=60) as response:
        revision = json.load(response)["sha"]
    base = f"https://raw.githubusercontent.com/mapo80/DocCornerNet-CoordClass/{revision}"
    artifacts = {
        "model.py": "model.py",
        "LICENSE": "LICENSE",
        "config.json": "checkpoints/mobilenetv2_224_best/config.json",
        "best_model.weights.h5": "checkpoints/mobilenetv2_224_best/best_model.weights.h5",
    }
    hashes = {}
    for filename, remote_path in artifacts.items():
        url = f"{base}/{remote_path}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        (MODEL_DIR / filename).write_bytes(content)
        hashes[filename] = hashlib.sha256(content).hexdigest()
    spec = importlib.util.spec_from_file_location("doccornernet_upstream", MODEL_DIR / "model.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    if config["input_norm"] != "imagenet":
        raise ValueError("This runner requires the ImageNet-normalized checkpoint.")
    keys = ("backbone", "alpha", "fpn_ch", "simcc_ch", "img_size", "num_bins", "tau",
            "backbone_minimalistic", "backbone_include_preprocessing")
    training_model = upstream.create_model(
        **{key: config[key] for key in keys}, backbone_weights=None)
    training_model.load_weights(str(MODEL_DIR / "best_model.weights.h5"))
    probability = tf.keras.layers.Activation("sigmoid")(training_model.output["score_logit"])
    combined = tf.keras.layers.Concatenate()([training_model.output["coords"], probability])
    inference_model = tf.keras.Model(training_model.input, combined)
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    content = converter.convert()
    model_path = MODEL_DIR / "model.tflite"
    model_path.write_bytes(content)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    probe = prepare_input(np.full((300, 400, 3), 128, dtype=np.uint8), config["img_size"])
    interpreter.set_tensor(input_info["index"], probe)
    interpreter.invoke()
    actual = interpreter.get_tensor(output_info["index"])
    expected = inference_model(probe, training=False).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    metadata = {
        "repository": "https://github.com/mapo80/DocCornerNet-CoordClass",
        "revision": revision,
        "checkpoint": "mobilenetv2_224_best",
        "artifact_sha256": hashes,
        "model_sha256": hashlib.sha256(content).hexdigest(),
        "tensorflow_version": tf.__version__,
        "input": {"shape": input_info["shape"].tolist(), "normalization": "imagenet", "color": "RGB"},
        "output": {"shape": output_info["shape"].tolist(), "format": "TL,TR,BR,BL normalized xy then sigmoid probability"},
        "keras_tflite_max_absolute_error": float(np.max(np.abs(actual - expected))),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {model_path}; Keras/TFLite parity check passed.")


def run_images(input_path, output_path, threshold):
    import tensorflow as tf

    model_path = MODEL_DIR / "model.tflite"
    metadata_path = MODEL_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ValueError("Run --prepare first to download weights and export the model.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model hash differs from metadata; run --prepare again.")
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if (len(inputs) != 1 or len(outputs) != 1
            or inputs[0]["shape"].tolist() != [1, 224, 224, 3]
            or outputs[0]["shape"].tolist() != [1, 9]
            or inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32):
        raise ValueError("Unexpected model signature; use this runner's --prepare export.")
    interpreter.set_tensor(inputs[0]["index"], np.zeros((1, 224, 224, 3), np.float32))
    interpreter.invoke()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(path for path in input_path.iterdir()
                   if path.is_file() and path.suffix.lower() in extensions) if input_path.is_dir() else [input_path]
    if not paths:
        raise ValueError("No supported images found.")
    output_path.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        try:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image.")
            started = time.perf_counter()
            tensor = prepare_input(image)
            interpreter.set_tensor(inputs[0]["index"], tensor)
            inference_started = time.perf_counter()
            interpreter.invoke()
            inference_ms = (time.perf_counter() - inference_started) * 1000
            prediction = interpreter.get_tensor(outputs[0]["index"])
            result = decode_prediction(prediction, image.shape, threshold)
            result.update({"input": path.name, "inference_ms": inference_ms,
                           "preprocess_and_predict_ms": (time.perf_counter() - started) * 1000,
                           "model_sha256": metadata["model_sha256"]})
            destination = output_path / path.name
            destination.mkdir(parents=True, exist_ok=True)
            crop_path = destination / "crop.png"
            if crop_path.exists():
                crop_path.unlink()
            overlay = image.copy()
            if result["document_candidate_found"]:
                corners = np.array(result["corners"], np.float32)
                cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True, (0, 200, 0), 3)
                for index, point in enumerate(corners):
                    position = tuple(np.rint(point).astype(int))
                    cv2.circle(overlay, position, 6, (0, 0, 255), -1)
                    cv2.putText(overlay, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                write_image(crop_path, rectify_document(image, corners))
            write_image(destination / "overlay.png", overlay)
            (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            records.append(result)
            print(f"{path.name}: found={result['document_candidate_found']}, score={result['presence_score']:.4f}")
        except (OSError, ValueError, cv2.error) as error:
            records.append({"input": path.name, "error": str(error)})
            print(f"{path.name}: ERROR: {error}")
    valid = [record for record in records if "error" not in record]
    latencies = [record["inference_ms"] for record in valid]
    summary = {
        "total": len(records), "processed": len(valid), "errors": len(records) - len(valid),
        "candidates_found": sum(record["document_candidate_found"] for record in valid),
        "threshold": threshold, "model": metadata,
        "mean_inference_ms": float(np.mean(latencies)) if latencies else None,
        "p95_inference_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "note": "Unlabeled predictions, not accuracy. Single document per image. Desktop CPU latency, not mobile latency.",
        "results": records,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Processed {len(valid)}/{len(records)}; candidates={summary['candidates_found']}")
    if summary["errors"]:
        raise SystemExit(1)


def benchmark_images(input_path, output_path, runs=20, warmup=20, threads=1):
    import tensorflow as tf

    model_path = MODEL_DIR / "model.tflite"
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model hash differs from metadata; run --prepare again.")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(path for path in input_path.iterdir()
                   if path.is_file() and path.suffix.lower() in extensions) if input_path.is_dir() else [input_path]
    if not paths:
        raise ValueError("No supported images found.")
    started = time.perf_counter()
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    setup_ms = (time.perf_counter() - started) * 1000
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if (input_info["shape"].tolist() != [1, 224, 224, 3]
            or output_info["shape"].tolist() != [1, 9]):
        raise ValueError("Unexpected model signature; run --prepare again.")
    inference_times = []
    frame_times = []
    records = []
    first_invoke_ms = None

    def summarize(values):
        samples = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values), "mean_ms": float(samples.mean()),
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "min_ms": float(samples.min()), "max_ms": float(samples.max()),
            "serial_throughput_fps": float(1000 / samples.mean()),
        }

    for path in paths:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        if first_invoke_ms is None:
            interpreter.set_tensor(input_info["index"], prepare_input(image))
            started = time.perf_counter()
            interpreter.invoke()
            first_invoke_ms = (time.perf_counter() - started) * 1000
            for _ in range(warmup):
                interpreter.invoke()
        image_inference = []
        image_frames = []
        for _ in range(runs):
            started = time.perf_counter()
            interpreter.set_tensor(input_info["index"], prepare_input(image))
            invoke_started = time.perf_counter()
            interpreter.invoke()
            inference_ms = (time.perf_counter() - invoke_started) * 1000
            prediction = interpreter.get_tensor(output_info["index"])
            decode_prediction(prediction, image.shape)
            frame_ms = (time.perf_counter() - started) * 1000
            image_inference.append(inference_ms)
            image_frames.append(frame_ms)
        inference_times.extend(image_inference)
        frame_times.extend(image_frames)
        records.append({"input": path.name, "image_size": list(image.shape[:2]),
                        "inference": summarize(image_inference),
                        "preprocess_inference_postprocess": summarize(image_frames)})
    report = {
        "model_sha256": metadata["model_sha256"],
        "checkpoint": metadata["checkpoint"],
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": platform.python_version(), "tensorflow": tf.__version__,
                        "tflite_threads": threads, "opencv_threads": cv2.getNumThreads()},
        "images": len(paths), "runs_per_image": runs, "warmup_invocations": warmup,
        "interpreter_setup_ms": setup_ms, "first_invoke_ms": first_invoke_ms,
        "inference": summarize(inference_times),
        "preprocess_inference_postprocess": summarize(frame_times),
        "scope": "Desktop CPU, batch size 1. Inference is invoke() only. Frame timing includes RGB conversion, Pillow resize, normalization, tensor copies, inference and corner decoding. Excludes TensorFlow import, file reads, image decode, overlays, perspective crop, file writes, camera capture and display. Sequential repeated static images; not mobile or live-camera FPS.",
        "results": records,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "latency.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="DocCornerNet 224px TFLite experiment.")
    parser.add_argument("--prepare", action="store_true", help="Download upstream source/weights and export TFLite.")
    parser.add_argument("--input", type=Path, help="Image or directory (nonrecursive).")
    parser.add_argument("--output", type=Path, default=Path("outputs/doccornernet"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark", action="store_true", help="Measure repeated warm inference without writing images.")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per image for --benchmark.")
    parser.add_argument("--warmup", type=int, default=20, help="Extra warmup invocations before benchmark samples.")
    parser.add_argument("--threads", type=int, default=1, help="TFLite CPU threads for --benchmark.")
    args = parser.parse_args()
    if not args.prepare and args.input is None:
        parser.error("Provide --prepare or --input.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("--runs and --threads must be positive; --warmup must be nonnegative.")
    if args.benchmark and args.input is None:
        parser.error("--benchmark requires --input.")
    if args.input is not None and not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.prepare:
        prepare_model()
    if args.input is not None:
        if args.benchmark:
            benchmark_images(args.input, args.output, args.runs, args.warmup, args.threads)
        else:
            run_images(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()