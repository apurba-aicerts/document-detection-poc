# Document Detection Proof of Concept

| Field | Value |
| --- | --- |
| Evaluation date | 2026-10-23 |
| Status | Experimental feasibility demonstrated; not production approved |
| Audience | Backend developers, mobile developers, engineering leads, and QA |
| Implemented approaches | Classical OpenCV and pretrained DocCornerNet with TensorFlow Lite |
| Current interface | Local Python command-line tools |

## 1. Executive Summary

This POC investigates whether an image contains a document-like region and whether
its four corners can be located for boundary drawing, cropping, and perspective
correction. The intended eventual use includes mobile-camera images of identity
documents, but this implementation evaluates static images on a Windows desktop.

Two approaches were implemented and run on the same 50 local image files:

| Approach | Images processed | Accepted candidates | No accepted candidate |
| --- | ---: | ---: | ---: |
| OpenCV geometric baseline | 50 | 9 | 41 |
| DocCornerNet, MobileNetV2 224px | 50 | 43 | 7 |

These counts describe algorithm decisions, not correct detections. Real-image
ground-truth labels are not available, and manual spot checks found inaccurate
boundaries even when the neural model reported a high presence score.

The repeated DocCornerNet benchmark measured a mean of 3.02 ms for inference alone
and 12.18 ms for preprocessing, inference, and corner decoding on this desktop.
These are not mobile-camera, network API, or production-load measurements.

**Recommendation:** continue with DocCornerNet as a candidate for a labeled accuracy
evaluation and target-device trial. Do not approve automatic document acceptance or
downstream identity processing based only on presence scores or candidate counts.

## 2. Problem Statement

Document-processing systems need to distinguish the document region from its
surroundings before cropping or passing the image to downstream systems. Background
clutter, lighting variation, shadows, weak contrast, perspective, and small document
size make this difficult. Four corners must describe the intended outer boundary,
not an internal printed panel, card holder, screen, or other rectangular object.

This POC evaluates technical feasibility; it does not establish universally exact
boundaries. Hidden corners, partially visible documents, and curved pages require
additional assumptions or different methods.

## 3. Scope

### Implemented

- Image-file and nonrecursive image-directory processing.
- One candidate region per image.
- Four predicted corners in decoded original-image pixel coordinates.
- Boundary overlays and JSON prediction results.
- Perspective-corrected crops for accepted candidates.
- Synthetic smoke tests and focused automated tests.
- A repeatable desktop CPU latency benchmark for DocCornerNet.

### Not Implemented or Not Validated

- Live CameraX or AVFoundation integration and preview-coordinate mapping.
- Android or iOS device performance and sustained thermal testing.
- A production HTTP API, concurrent service, or production deployment.
- Reliable separation of multiple documents in one image.
- OCR, document-type classification, authenticity checks, or face matching.
- Dedicated blur, glare, readability, or capture-quality assessment.
- Full real-image accuracy evaluation, threshold calibration, or fine-tuning.
- Google ML Kit Document Scanner and Apple Vision evaluation.

Orientation analysis is not part of this report's implementation scope.

## 4. Technologies Evaluated

### 4.1 Classical OpenCV Baseline

The pipeline resizes images, converts them to grayscale, denoises them, enhances
local contrast, and generates contours from Canny edges and both threshold polarities.
Large, convex quadrilaterals are ranked using area and contour fit. Near-border
contours are rejected, and the highest-ranked candidate is selected.

This is a rule-based geometric detector, not a trained AI model. Its default minimum
candidate area is 10% of the image. It is useful as a simple baseline, but it cannot
establish that a rectangle is semantically a document.

### 4.2 DocCornerNet with TensorFlow Lite

The POC uses the pretrained `mobilenetv2_224_best` checkpoint from
[DocCornerNet-CoordClass](https://github.com/mapo80/DocCornerNet-CoordClass).
The model predicts document corners using marginal coordinate classification and
produces a separate document-presence logit. No model training was performed locally.

| Property | Implemented value |
| --- | --- |
| Upstream revision | `d877dc382c4ec56299d2b2662b979581bae9c8cd` |
| Input | Float32 `[1,224,224,3]`, RGB, ImageNet-normalized |
| Exported output | Float32 `[1,9]`, eight coordinates plus sigmoid presence score |
| Coordinate range | Normalized `[0,1]` |
| Intended corner sequence | Top-left, top-right, bottom-right, bottom-left |
| Default presence threshold | 0.5, not calibrated |
| Additional acceptance check | Convex quadrilateral with area at least four image pixels |
| Execution | Local CPU inference, one TFLite thread in the benchmark |

The combined output is our custom export contract, not a universal DocCornerNet
signature. Upstream inference code returns separate coordinates and a score logit.
Application integrations must inspect and validate the actual model artifact.

## 5. Architecture and Processing

```text
Encoded image
  -> Decode and establish image coordinate space
  -> BGR-to-RGB conversion
  -> Direct bilinear resize to 224 x 224
  -> ImageNet normalization
  -> TFLite inference
  -> Validate outputs and map corners to decoded image coordinates
  -> Presence threshold and geometry checks
  -> JSON result, optional overlay and perspective-corrected crop
```

For each RGB channel, preprocessing applies `(pixel / 255 - mean) / std` using
means `[0.485,0.456,0.406]` and standard deviations `[0.229,0.224,0.225]`.
Pillow bilinear resizing stretches the image directly to a square; no letterboxing
is used. Changing preprocessing changes the model's input distribution and requires
validation rather than being treated as an interchangeable implementation detail.

Normalized x and y values are multiplied by decoded image width and height,
respectively, then clamped to valid pixel indices. Integration must track any
additional resizing, mirroring, or preview transforms separately.

The JSON result retains scores, predicted coordinates, and geometry validity even
when a candidate is rejected. A candidate is a plausible region under the model and
rules; it is not a confirmed identity document or an authenticity judgment.

## 6. Environment and Reproducibility

| Component | Verified environment |
| --- | --- |
| Operating system | Windows 11, build 10.0.26300 |
| Python | 3.13.3 |
| TensorFlow | 2.20.0 |
| Processor identifier | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| TFLite threads | 1 |
| OpenCV reported threads | 24 |

The processor identifier is not a complete hardware specification. CPU power mode,
background load, and thermal conditions were not controlled as a formal benchmark.
Compatibility with other systems must be checked separately.

Dependencies are defined in [requirements.txt](requirements.txt) for the baseline
and [requirements-doccornernet.txt](requirements-doccornernet.txt) for DocCornerNet.
The latter constrains NumPy below 2.3 and adds TensorFlow and Pillow. Separate
environments avoid changing the original OpenCV setup. Version ranges support the
experiment; production builds should lock exact approved package versions.

Model preparation downloads revision-pinned source, checkpoint weights, configuration,
and the upstream MIT license. It rebuilds the model, loads published weights, exports
TFLite, and compares Keras and TFLite outputs on a gray probe. The recorded maximum
absolute difference was approximately `1.82e-6`. This verifies the conversion on that
probe, not model accuracy or complete cross-runtime equivalence.

Artifact hashes, tensor shapes, revision, and export runtime are recorded locally in
[models/doccornernet/metadata.json](models/doccornernet/metadata.json). Preparation
executes upstream architecture code; it belongs in a reviewed build step, not request
handling. Generated model artifacts and image outputs are ignored by Git and must
be distributed separately through an approved artifact mechanism when needed.

## 7. Evaluation Method

### Dataset

The local `tests/` directory contained 50 supported image files, including identity
document examples, transformed images, and duplicate/copy files. It is a convenience
test collection, not a balanced or statistically representative dataset. No complete
per-image presence labels or four-corner annotations were established.

Both implementations processed the same files using their initial settings. DocCornerNet
used a presence threshold of 0.5 with geometry validation; OpenCV used its geometric
ranking and default area rule. Their acceptance criteria differ, so comparing raw
candidate counts alone cannot establish which detector is more accurate.

### Automated Checks

The last recorded full suite passed 10 tests covering the classical detector and
DocCornerNet preprocessing/decoding. The synthetic positive cases exercise perspective,
rotation, texture, dim lighting, uneven lighting, and low contrast for the OpenCV
baseline. Its synthetic tests require polygon mask IoU above 0.95 and corner error
below eight pixels at the generated resolution.

DocCornerNet helper tests cover channel conversion and normalization, image-coordinate
mapping, score thresholding, degenerate geometry, boundary clamping, and malformed
outputs. A separate manual smoke run rejected all three generated negatives: blank,
noise, and circle. These checks do not establish real-image accuracy.

### Manual Inspection

Spot checks found a close card outline on an image missed by OpenCV, an incorrect
interior outline on another image despite high confidence, and a rotated-card outline
inside its holder. This was not a systematic annotation or review of all 50 results.

## 8. Functional Results

All 50 files completed processing without reported image-processing errors for each
approach. OpenCV accepted nine candidates, while DocCornerNet accepted 43.

The seven DocCornerNet nonacceptances may reflect low scores, invalid geometry, or
both. They must not be reported as seven confirmed document-free images. Likewise,
the 43 accepted candidates must not be reported as 43 correct localizations.

Presence precision, recall, false-positive rate, localization IoU, and real-image
corner error remain unmeasured because the required ground truth is missing.

Evidence is available locally in [outputs/tests](outputs/tests) and
[outputs/doccornernet/summary.json](outputs/doccornernet/summary.json). Avoid placing
identity-document examples in broadly shared reports without permission and redaction.

## 9. Latency Results

The dedicated benchmark used 50 images, 20 measured runs per image, batch size one,
one initial invocation followed by 20 additional warmups, and one TFLite CPU thread.
It produced 1,000 measured samples. Each image was read and decoded once outside the
measurement window; repeated static-image samples are not independent live captures.

| Timing scope | Mean | Median | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| TFLite `invoke()` only | 3.02 ms | 2.98 ms | 3.67 ms | 4.24 ms |
| Preprocessing + inference + corner decoding | 12.18 ms | 13.58 ms | 18.75 ms | 24.16 ms |

The broader measurement includes RGB conversion, Pillow resize, normalization,
tensor input/output copies, inference, and geometry decoding. Both measurements
exclude file reads, image decoding, perspective cropping, overlay generation,
output writing, camera capture, display, transport, and server queuing.

Interpreter setup was approximately 7.03 ms and its first invocation approximately
3.15 ms in this benchmark process. TensorFlow import and process startup were not
timed. The earlier single-pass batch report has different timings; this repeated
benchmark is the source for the latency table above.

These results demonstrate low warm compute latency on the tested desktop, not a
guaranteed mobile frame rate or production API response-time SLA. The report is
[outputs/doccornernet-latency/latency.json](outputs/doccornernet-latency/latency.json).

## 10. Issues Encountered and Responses

| Issue | Response | Remaining consideration |
| --- | --- | --- |
| Dependency installation/environment mismatch during initial setup | Used explicit interpreter paths and isolated environments | Verify imports and lock deployment dependencies |
| Image-frame outline accepted as a page by OpenCV | Expanded the near-border guard and reran the failing regression test | Documents close to the image boundary may also be rejected |
| No committed TFLite model found in the inspected upstream revision | Exported from published checkpoint weights | Package and version approved artifacts |
| README output description differed from upstream inference code | Added explicit sigmoid and a combined export with signature checks | Do not reuse assumptions for a different export |
| High presence confidence with inaccurate corners | Retained outputs for review and documented the failure | Labeled localization evaluation is required |
| TensorFlow TFLite interpreter deprecation warning | Verified the current runtime works and documented the warning | Test LiteRT migration before upgrading |
| Duplicate files and missing ground truth | Restricted conclusions to candidate counts and feasibility | Deduplicate/group data and annotate a proper evaluation set |

## 11. Backend and Deployment Considerations

- Initialize and warm one interpreter per worker or use a bounded interpreter pool.
  Do not invoke a shared interpreter concurrently without synchronization.
- Benchmark worker and thread counts together; avoid CPU oversubscription and use
  bounded queues, request timeouts, and backpressure.
- Validate decoded image content and impose upload-byte, pixel-count, memory, and
  crop-dimension limits. File extensions alone are not sufficient validation.
- Keep invalid input, no candidate, invalid geometry, and successful localization
  distinct. Include model version, image dimensions, coordinate-space definition,
  score, and configured threshold in any proposed API response.
- Preserve preprocessing and test reference predictions before changing resizing,
  decoder behavior, export precision, runtime, or delegate.
- Apply retention rules, access controls, encrypted transport, and restricted logs.
  Identity images and filenames may contain personal information. Do not save debug
  overlays and crops by default in production as this CLI currently does.
- Download and review source only during preparation; deploy immutable approved
  artifacts with trusted checksums and retain upstream license notices.
- Keep runtime inference separate from preparation/export dependencies where the
  target platform supports a smaller validated runtime.
- Mobile integration must additionally handle camera pixel formats, display mapping,
  frame backpressure, and sustained device performance. No mobile integration exists yet.

## 12. Proposed Acceptance Gates

These gates are recommendations, not completed approvals. Numeric accuracy targets
and latency budgets must be agreed with product owners before the next evaluation.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Functional feasibility | Images produce scores, corners, and optional corrected crops | Demonstrated locally |
| Artifact conversion | Verified signature and Keras/TFLite numerical agreement | Demonstrated on one probe |
| Real-image detection | Presence precision/recall on a labeled held-out set, including negatives | Not established |
| Boundary quality | Polygon IoU and normalized corner errors per condition | Not established |
| Target-device latency | Warm/cold P50/P95/P99 and sustained performance on target phones | Not tested |
| Backend readiness | Concurrent-load, resource-limit, malformed-input, and privacy testing | Not implemented |
| Operational readiness | Approved artifact, dependency lock, monitoring, and rollback process | Not established |

## 13. Recommended Next Steps

1. Define what boundary is required: physical card, printed card region, holder, or
   complete page. Define expected behavior for partial and multiple documents.
2. Label presence and four corners, add real negative examples, and group duplicate
   files and captures of the same physical document before splitting data.
3. Evaluate both methods with the same labeled metrics, broken down by document
   type, background, lighting, perspective, blur, and document size.
4. Calibrate thresholds on development data and evaluate once on a held-out set;
   consider fine-tuning only after failure analysis identifies the need.
5. Integrate the unchanged validated model on a target device and measure total
   camera-to-result latency, memory, stability, and thermal behavior.
6. Add a production backend only if server inference is required, then perform
   concurrency, security, privacy, and deployment-readiness checks.

## 14. Reproduction Commands

Run from the repository root in PowerShell. The DocCornerNet environment must be
created using a Python version compatible with the selected TensorFlow release.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe detect_document.py --demo
.\.venv\Scripts\python.exe detect_document.py --input tests --output outputs/tests
.\.venv\Scripts\python.exe -m unittest test_detect_document -v

python -m venv .venv-doccornernet
.\.venv-doccornernet\Scripts\python.exe -m pip install -r requirements-doccornernet.txt
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --prepare
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --input tests --output outputs/doccornernet
.\.venv-doccornernet\Scripts\python.exe -m unittest -v
.\.venv-doccornernet\Scripts\python.exe detect_doccornernet.py --benchmark --input tests --runs 20 --warmup 20 --threads 1 --output outputs/doccornernet-latency
```

## 15. Deliverables and Conclusion

| Deliverable | Location |
| --- | --- |
| OpenCV detector | [detect_document.py](detect_document.py) |
| DocCornerNet export, inference, and benchmark runner | [detect_doccornernet.py](detect_doccornernet.py) |
| Baseline regression tests | [test_detect_document.py](test_detect_document.py) |
| DocCornerNet helper tests | [test_doccornernet.py](test_doccornernet.py) |
| Setup and model contract | [README.md](README.md) |
| Real-image predictions | [outputs/doccornernet/summary.json](outputs/doccornernet/summary.json) |
| Repeated latency measurements | [outputs/doccornernet-latency/latency.json](outputs/doccornernet-latency/latency.json) |

**Conclusion:** local inference, model conversion, coordinate extraction, optional
cropping, and low desktop warm latency have been demonstrated. DocCornerNet produced
more accepted candidates than the classical baseline, but superior accuracy has not
been established. The next decision should be based on labeled boundary evaluation
and measurements on the intended deployment platform, not candidate counts alone.