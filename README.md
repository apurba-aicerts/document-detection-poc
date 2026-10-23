# Document Detection POC

Document-boundary experiments: an OpenCV baseline and a separate DocCornerNet TFLite model.
Both predict at most one document per image. No OCR or identity verification is performed.

For objectives, methodology, measured results, limitations, and next-stage acceptance
gates, see the [POC report](POC_REPORT.md).

## Run

Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe detect_document.py --demo
.\.venv\Scripts\python.exe detect_document.py --input "C:\photos\page.jpg"
.\.venv\Scripts\python.exe detect_document.py --input "C:\photos" --output outputs\photos
.\.venv\Scripts\python.exe -m unittest test_detect_document -v
```

Directory mode processes supported images directly inside the directory, not subdirectories.
Use `--min-area 0.05` to allow candidates covering as little as 5% of the image.
The default is 10%. No candidate is a normal result; unreadable inputs cause a nonzero exit.

## Outputs

Each image gets an output directory containing:

- `overlay.png`: detected quadrilateral and numbered corners on the original image.
- `edges.png`: Canny edge map at processing resolution (maximum side 1200 pixels).
- `crop.png`: perspective-corrected crop, only when a candidate exists.
- `result.json`: candidate presence, original image dimensions, corners, and geometry scores.

Coordinates use the decoded image's pixel space, with origin at the top left.
Corners follow clockwise image order starting at the smallest x+y corner, usually top left.
At a 45-degree rotation that starting corner can be ambiguous; all four remain distinct.
The crop is not guaranteed to have upright text.

## Pipeline

1. Resize, grayscale, denoise, and enhance local contrast for edge detection.
2. Combine candidates from closed Canny edges and both polarities of Otsu thresholding.
3. Keep large, convex quadrilaterals whose contour area closely matches their polygon area.
4. Reject candidates touching image boundaries and rank the remainder by area and contour fit.
5. Map corners back to original resolution and apply a perspective transform.

`ranking_score` is a geometry heuristic, not a calibrated probability of document presence.
A screen, tabletop, or another rectangle can be a false positive. The baseline cannot
establish semantic document presence, recover hidden boundaries, or reliably handle
curled pages, glare, severe shadows, low contrast, and clutter. It selects only one candidate.

## Experiments

`--demo` generates reproducible synthetic images and corner ground truth for perspective,
rotation, a 45-degree square, textured backgrounds, dim and uneven lighting, low contrast,
and negative examples (blank, texture, circle). These are controlled smoke tests, not
evidence of real-world accuracy. Tests require mask IoU above 0.95 and corner error below
8 pixels on the generated 800x600 positives, and reject the generated negatives.

For meaningful evaluation, collect real photos grouped by background, orientation,
lighting, document type, and capture session. Label presence and four visible corners.
Include hard negatives such as monitors and tables. Measure presence precision/recall,
polygon IoU, corner error divided by image diagonal, and latency per condition.
Tune on a development subset and report on separate capture sessions. The current
experiment does not yet automate that real-image benchmark.

## DocCornerNet + TFLite

Use a separate environment (tested on Windows, Python 3.13.3, TensorFlow 2.20.0):

```powershell
python -m venv .venv-doccornernet
.\.venv-doccornernet\Scripts\python.exe -m pip install -r requirements-doccornernet.txt
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --prepare
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --input "tests/aadhaar.jpeg" --output outputs/doccornernet-single
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --input tests --output outputs/doccornernet
.\.venv-doccornernet\Scripts\python.exe -m unittest -v
```

`--prepare` downloads and executes the model architecture from
[DocCornerNet-CoordClass](https://github.com/mapo80/DocCornerNet-CoordClass)
at revision `d877dc382c4ec56299d2b2662b979581bae9c8cd`, together with its MIT license,
224px MobileNetV2 checkpoint weights, and training configuration. It does not download
training images or train a model. Artifacts stay under ignored `models/doccornernet/`.
Only preparation needs network access; input images are processed locally, not uploaded.

The runner rebuilds the architecture with no ImageNet weight download, loads the
published weights, and exports a float32 TFLite model. Its input is `[1,224,224,3]`,
RGB, stretched to square using Pillow bilinear resizing, then ImageNet-normalized.
Unlike the upstream inference function (which returns coordinates and a score logit),
our export returns `[1,9]`: eight normalized coordinates followed by sigmoid probability.
Preparation checks numerical agreement with Keras on a gray probe; this verifies
conversion plumbing, not detection accuracy. Revision, artifact hashes, model hash,
TensorFlow version, tensor shapes, and parity error are recorded in `metadata.json`.

Inference preserves the model's TL, TR, BR, BL coordinate sequence and maps normalized
coordinates to decoded original-image width/height, clamping to valid pixel indices.
The default `--threshold 0.5` is an initial experiment setting, not a calibrated threshold.
A candidate also needs a convex, nondegenerate quadrilateral. JSON retains predictions
even when rejected. Crops use the baseline's geometric corner ordering and do not
guarantee upright text. No fallback to OpenCV detection is applied.

Each image gets `overlay.png`, `result.json`, and, when accepted, `crop.png`.
The output root also contains `summary.json` with per-image scores, geometry validity,
timings, and processing errors. Inference timing excludes decoding, preprocessing,
image writing, interpreter startup, and a warmup invocation; it is desktop CPU timing,
not a mobile performance measurement. The TensorFlow interpreter currently emits an
upstream deprecation warning; eventual LiteRT migration remains separate work.

Initial local batch: all 50 input images processed, 43 accepted candidates at threshold
0.5, versus 9 in the classical baseline. These are prediction counts, not accuracy.
Manual spot checks found both close outlines and incorrect interior boundaries despite
high presence scores. Do not equate presence confidence with corner precision. Multiple
cards, holder boundaries, partially visible documents, real negative images, and other
capture conditions still require labeled evaluation. Keep duplicate images and images
of the same physical document/capture session together when splitting evaluation data.

### Latency Benchmark

```powershell
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --benchmark --input tests --runs 20 --warmup 20 --threads 1 --output outputs/doccornernet-latency
```

This runs 20 measurements per image after an initial untimed invocation and 20 extra
warmups. Images are read and decoded once each, outside the measurements. The benchmark
does not generate overlays or crops. `latency.json` includes mean, median, P95, P99,
minimum, maximum, per-image summaries, model hash, and environment details.

Two scopes are recorded: TFLite `invoke()` alone, and RGB conversion + resize +
normalization + tensor copies + inference + corner decoding. Both exclude camera input,
disk reads, image decoding, rendering, perspective correction, and output writing.
Interpreter setup and its first invocation are reported separately; TensorFlow import
and process startup are not timed. Reciprocal mean latency is serial compute throughput,
not observed camera FPS. Results describe this desktop CPU, not an Android/iOS device.