# Trustworthy Analysis and Image Validation Design

## Objective

AgriSense must stop treating every decodable image as a leaf and stop forcing a
four-class stress answer when the evidence is unreliable. The next model may
replace the current production bundle only when it improves honestly measured
held-out performance and passes explicit leaf-validity, uncertainty, and API
edge-case gates.

The current measured baseline is 81.92% held-out accuracy and 79.02% macro-F1
over 177 test windows. The current upload endpoint checks file structure but has
no content gate. A reproduced dashboard screenshot was therefore accepted and
classified as Medium stress with 86.68% confidence. This is expected closed-set
softmax behavior, not evidence that the image contains a leaf.

## Considered Approaches

### 1. Confidence threshold only

Reject predictions whose maximum stress probability is low. This is simple but
unsafe: the reproduced non-leaf screenshot received high confidence, and several
wrong held-out predictions also exceed 75% confidence. This approach is rejected.

### 2. Strict hand-written color and shape rules

Require a green, centered, leaf-shaped foreground. This catches simple invalid
images but incorrectly rejects brown, yellow, heavily diseased, narrow, or
partially cropped leaves. These rules are useful for quality signals but are not
sufficient as the content decision. This approach is rejected as the primary
gate.

### 3. Calibrated multi-stage gate plus selective prediction

Use file and quality validation, a learned visual out-of-distribution score
calibrated from real training leaves, crop-consistency evidence, and an
uncertainty/abstention layer after stress inference. This approach is selected.
It separates “is this input suitable?” from “what stress evidence is present?”
and allows the service to return an honest inconclusive analysis.

## Input Contract and Validation States

The public input remains one current RGB leaf image plus seven sensor readings.
Image validation produces one of three states:

- `accepted`: a sufficiently clear, leaf-like image that may enter inference.
- `retry_required`: probably a leaf, but quality or out-of-distribution evidence
  is too weak for reliable analysis. Prediction is blocked and the response tells
  the user how to retake the image.
- `rejected`: corrupt, unsupported, blank, extremely small, or confidently
  non-leaf content. Prediction is blocked.

There is no UI override for blocked inputs. This prevents a warning from being
ignored and converted into a misleading persisted prediction.

The validation result contains a machine-readable reason code, a short user
message, quality measurements, leaf-likeness score, and threshold version. The
API uses HTTP 422 for decoded but unsuitable images and HTTP 400 for malformed or
unsupported uploads.

## Image Suitability Gate

Validation runs before an upload record is finalized:

1. **File validation:** require a file, allow JPEG/PNG/WebP by decoded format,
   enforce the configured byte limit, reject decompression bombs, animation, and
   invalid/truncated content, and normalize EXIF orientation.
2. **Geometry and quality:** enforce minimum dimensions; calculate brightness,
   contrast, sharpness, clipping, and near-uniform-image signals with Pillow and
   NumPy. Thresholds are reported separately so a quality failure is not
   mislabeled as “not a leaf.”
3. **Leaf-likeness:** run the same frozen MobileNetV2 encoder used by the stress
   model and compare its normalized embedding with reference centroids generated
   only from training-split leaf images. A validation-split percentile calibrates
   the acceptance threshold. The contract stores centroid and threshold versions,
   not an arbitrary UI constant.
4. **Negative calibration suite:** obvious non-leaf fixtures—documents, UI
   screenshots, people/objects, solid images, and noise—verify rejection behavior.
   These cases tune the rejection boundary but never enter the stress labels.

The predictor repeats the gate before inference as defense in depth, so a caller
cannot bypass validation by creating or reusing an upload ID outside the normal
frontend flow.

## Accuracy Improvement Strategy

The dataset builder will use substantially more globally unique PlantVillage
images while retaining plant-level splitting and zero source-image overlap.
Class-aware sampling will reduce the current Medium-class dominance without
fabricating labels. The stress mapping remains explicit and documented as a
research approximation.

Training becomes two-stage:

1. Train the image, crop, and sensor heads with MobileNetV2 frozen.
2. Unfreeze only the final MobileNetV2 block, use a much lower learning rate,
   retain augmentation and class weighting, and stop on validation macro-F1/loss.

The image branch also receives an auxiliary crop head derived from the existing
Plant Type labels. It supplies crop-consistency evidence and a useful visual
learning signal; it does not directly overwrite stress probabilities. Sensor
telemetry remains synthetic supporting context and cannot become the primary
decision shortcut.

Candidate promotion requires all of the following:

- held-out accuracy greater than 81.92%;
- held-out macro-F1 greater than 79.02%;
- no material class recall collapse, with every class represented in test data;
- image-only macro-F1 greater than sensor-only macro-F1;
- valid finite probabilities and improved or non-regressed Brier score;
- leaf validation false-rejection rate at or below 5% on held-out valid leaves;
- all committed obvious non-leaf edge fixtures rejected;
- model, calibration data, and preprocessing contract promoted atomically.

If no candidate passes, the current working model remains deployed and the report
records the failed candidate rather than publishing a misleading improvement.

## Selective and Explainable Analysis

Passing image validation does not guarantee that the stress class is reliable.
After prediction, a calibrated decision policy uses maximum probability, margin
between the top two classes, entropy, and validation-derived thresholds.

The analysis status is:

- `completed`: evidence meets the calibrated reliability gate;
- `inconclusive`: the image is a leaf, but the model cannot distinguish the
  stress classes reliably;
- `blocked`: the image suitability gate did not pass.

An inconclusive result does not present the top class as a diagnosis. The API may
retain candidate probabilities for audit, but the primary UI clearly says that a
reliable stress conclusion could not be made and asks for a better image or more
representative sensor readings.

A completed analysis returns:

- stress class probabilities and calibrated confidence;
- reliability level and the factors behind it;
- image validation and quality summary;
- selected versus image-inferred crop consistency;
- bounded seven-reading sensor trend and out-of-range indications;
- visual activation regions as supporting evidence, not proof;
- concise observations and safe next steps, including monitoring, retaking the
  image, verifying sensor readings, or consulting an agronomy professional.

Recommendations must not claim disease identification, prescribe chemicals, or
present synthetic telemetry as real agronomic evidence.

## API, Persistence, and Frontend Behavior

`POST /upload` validates the image before success. Accepted uploads return their
validation summary. Blocked uploads return a structured error with `status`,
`reason_code`, `message`, and actionable capture guidance.

`POST /predict` revalidates the upload and returns `analysis_status`, reliability,
crop consistency, observations, and recommendations in addition to the current
probability and explanation fields. A backward-compatible database migration adds
nullable JSON/status fields without deleting existing history.

The upload page shows validation progress before model analysis. Rejected inputs
stay on the page with a prominent alert, preview indication, reason, and retake
instructions. The prediction page has separate completed and inconclusive states;
it never styles an inconclusive top candidate as a normal stress diagnosis.

## Edge-Case Test Matrix

Automated tests cover:

- missing file, empty file, wrong extension, MIME spoofing, corrupt/truncated
  content, unsupported format, oversized payload, animation, and decompression
  bombs;
- tiny, blank, near-uniform, severely dark, severely bright, blurred, grayscale,
  transparent, rotated, and valid diseased-leaf images;
- obvious non-leaf fixtures and held-out valid leaves;
- missing/unknown upload IDs and attempts to bypass upload validation;
- empty, short, exact, and long sensor sequences; missing columns; strings;
  non-finite values; schema-boundary values; and out-of-range values;
- finite, normalized probabilities; low-margin and high-entropy abstention;
  completed analysis; inconclusive analysis; and crop mismatch;
- persistence and history compatibility for old and new records;
- frontend alerts, blocked navigation, accepted navigation, inconclusive display,
  API failures, and responsive rendering.

The end-to-end smoke suite includes one accepted leaf, one rejected non-leaf, and
one deliberately inconclusive prediction case.

## Documentation and Limitations

The README and model page will report the new measured metrics only after an
accepted retraining run. Documentation will distinguish stress grouping from
disease diagnosis, real images from synthetic telemetry, rejection from
uncertainty, and selective accuracy from full-coverage accuracy. No result will be
described as field-validated until an independent real-world dataset supports it.

