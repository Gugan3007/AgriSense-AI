# Trustworthy Analysis and Image Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject unsuitable/non-leaf uploads, abstain from unreliable stress conclusions, improve the honestly measured stress model, and return a persisted actionable analysis instead of a forced prediction.

**Architecture:** The image-first model gains an auxiliary crop head and two-stage MobileNetV2 fine-tuning. A schema-v2 contract carries calibrated image-quality, leaf-embedding, and selective-prediction thresholds. Flask validates before saving and before inference; React renders blocked, inconclusive, and completed states.

**Tech Stack:** Python 3.12, TensorFlow/Keras 2.20, Pillow, NumPy, pandas, scikit-learn, Flask/SQLAlchemy, React 19, Vite, Node test runner, Playwright.

## Global Constraints

- Preserve stress labels in exact order: `Healthy`, `Low`, `Medium`, `High`.
- Preserve one RGB image `128 × 128 × 3` and seven sensor rows `7 × 4` at the production boundary.
- Keep the schema-v1 production bundle usable until a schema-v2 candidate passes every gate.
- Never use stress confidence as the leaf/non-leaf detector.
- Block both `rejected` and `retry_required` inputs without an override.
- Use HTTP 400 for malformed uploads and HTTP 422 for decoded but unsuitable images.
- Keep probabilities finite in `[0,1]`, summing to one; keep sensor stress scores in `[0,100]`.
- Promotion requires accuracy `> 0.8192090395480226`, macro-F1 `> 0.7901860901925868`, Brier `<= 0.2740994393825531`, image-only F1 greater than sensor-only F1, valid-leaf false rejection `<= 0.05`, and all committed obvious non-leaf cases blocked.
- Do not diagnose disease, prescribe treatment, or describe synthetic telemetry as field evidence.

---

### Task 1: Expand leakage-safe data and carry crop targets

**Files:**
- Modify: `dataset/build_sequences.py`
- Regenerate: `dataset/sequential_data.csv`
- Regenerate: `dataset/README_dataset.md`
- Modify: `ml/preprocessing.py`
- Modify: `ml/tests/test_preprocessing.py`

**Interfaces:**
- Produces: `CROP_LABELS: list[str]` in stable order.
- Produces: `make_windows(...) -> tuple[paths, sensors, stress_indices, crop_indices]`.
- Produces crop one-hot targets named `crop_probabilities`.

- [ ] **Step 1: Write failing crop/data tests**

```python
def test_make_windows_returns_stress_and_crop_targets(self):
    paths, sensors, stress, crops = make_windows(
        self.frame, ["P1"], 7, self.normalization
    )
    self.assertEqual(paths.shape, (1,))
    self.assertEqual(sensors.shape, (1, 7, 4))
    self.assertEqual(stress.shape, (1,))
    self.assertEqual(crops.shape, (1,))

def test_checked_in_dataset_is_large_balanced_and_unique(self):
    frame = load_valid_rows()
    self.assertGreaterEqual(frame["Plant_ID"].nunique(), 192)
    self.assertEqual(frame["Resolved_Image_Path"].nunique(), len(frame))
    self.assertLess(float(frame["Stress_Level"].value_counts(normalize=True).max()), .45)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest ml.tests.test_preprocessing -v`

Expected: FAIL because three arrays are returned and the dataset contains 64 plants.

- [ ] **Step 3: Implement crop targets and expanded class-aware generation**

```python
CROP_LABELS = [
    "Apple", "Cherry_(including_sour)", "Corn_(maize)", "Grape", "Peach",
    "Pepper,_bell", "Potato", "Strawberry", "Tomato",
]

def make_windows(frame, plant_ids, sequence_length, normalization):
    paths, sensors, stress_labels, crop_labels = [], [], [], []
    stress_to_index = {name: i for i, name in enumerate(CLASS_LABELS)}
    crop_to_index = {name: i for i, name in enumerate(CROP_LABELS)}
    mean = np.asarray(normalization["mean"], dtype="float32")
    std = np.asarray(normalization["std"], dtype="float32")
    selected = frame[frame["Plant_ID"].isin(plant_ids)]
    for _, group in selected.groupby("Plant_ID", sort=True):
        group = group.sort_values("Timestamp")
        for start in range(len(group) - sequence_length + 1):
            window = group.iloc[start:start + sequence_length]
            current = window.iloc[-1]
            paths.append(str(current["Resolved_Image_Path"]))
            sensors.append((window[SENSOR_COLUMNS].to_numpy("float32") - mean) / std)
            stress_labels.append(stress_to_index[str(current["Stress_Level"])])
            crop_labels.append(crop_to_index[str(current["Plant_Type"])])
    return tuple(map(np.asarray, (paths, sensors, stress_labels, crop_labels)))
```

Update decode/dataset/auxiliary-target functions to carry crop one-hot labels. Set the generator default to 192 plants, choose only disease classes with remaining images, retain monotonic stress progression, cap the largest stress share below 45%, and preserve zero image reuse.

- [ ] **Step 4: Rebuild and verify GREEN**

Run: `.venv/bin/python dataset/build_sequences.py --plants 192`

Expected: at least 4,500 unique rows, 192 plants, all stress classes, zero reuse, maximum stress share below 45%.

Run: `.venv/bin/python -m unittest ml.tests.test_preprocessing -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset ml/preprocessing.py ml/tests/test_preprocessing.py
git commit -m "feat: expand balanced crop stress dataset"
```

---

### Task 2: Add crop evidence and two-stage image fine-tuning

**Files:**
- Modify: `ml/model.py`
- Modify: `ml/train.py`
- Modify: `ml/tune.py`
- Modify: `ml/tests/test_model.py`

**Interfaces:**
- Produces training outputs `stress_probabilities`, `image_probabilities`, `sensor_probabilities`, `crop_probabilities`.
- Produces inference outputs `stress_probabilities`, `crop_probabilities`.
- Produces: `configure_fine_tuning(model, unfreeze_last=20) -> list[str]`.

- [ ] **Step 1: Write failing architecture tests**

```python
def test_model_exposes_crop_evidence(self):
    training, inference = build_models(image_weights=None)
    self.assertIn("crop_probabilities", training.output_names)
    self.assertEqual(set(inference.output_names), {
        "stress_probabilities", "crop_probabilities"
    })

def test_fine_tuning_excludes_batch_normalization(self):
    training, _ = build_models(image_weights=None)
    opened = configure_fine_tuning(training, unfreeze_last=20)
    self.assertGreater(len(opened), 0)
    self.assertTrue(all("batch_normalization" not in name for name in opened))
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest ml.tests.test_model -v`

Expected: FAIL because the crop head and fine-tuning API are absent.

- [ ] **Step 3: Implement model outputs and tail fine-tuning**

```python
crop_probabilities = layers.Dense(
    len(CROP_LABELS), activation="softmax", name="crop_probabilities"
)(image_features)

training_model = tf.keras.Model(inputs, {
    "stress_probabilities": stress_probabilities,
    "image_probabilities": image_probabilities,
    "sensor_probabilities": sensor_probabilities,
    "crop_probabilities": crop_probabilities,
})
inference_model = tf.keras.Model(inputs, {
    "stress_probabilities": stress_probabilities,
    "crop_probabilities": crop_probabilities,
})

def configure_fine_tuning(model, unfreeze_last=20):
    encoder = model.get_layer("image_encoder")
    encoder.trainable = True
    opened = []
    cutoff = max(0, len(encoder.layers) - unfreeze_last)
    for index, layer in enumerate(encoder.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
        if layer.trainable:
            opened.append(layer.name)
    return opened
```

Add crop loss weight `0.2`. Train frozen heads first, restore best weights, open the final 20 non-BatchNorm layers, recompile at `1e-5`, and fine-tune with early stopping on validation stress macro-F1. Concatenate both histories before report output.

- [ ] **Step 4: Verify GREEN and QA training**

Run: `.venv/bin/python -m unittest ml.tests.test_model -v`

Run: `.venv/bin/python ml/train.py --epochs 1 --fine-tune-epochs 1 --max-steps 1 --qa-only`

Expected: tests pass; both training phases run; staged model reloads; production bundle is unchanged.

- [ ] **Step 5: Commit**

```bash
git add ml/model.py ml/train.py ml/tune.py ml/tests/test_model.py
git commit -m "feat: add crop evidence and image fine tuning"
```

---

### Task 3: Version leaf and reliability calibration

**Files:**
- Create: `ml/calibration.py`
- Modify: `ml/artifacts.py`
- Modify: `ml/utils.py`
- Create: `ml/tests/test_calibration.py`
- Modify: `ml/tests/test_artifacts.py`

**Interfaces:**
- Produces: `quality_metrics(image) -> dict[str, float]`.
- Produces: `fit_leaf_reference(...) -> dict` and `leaf_similarity(...) -> tuple[float, str]`.
- Produces: `fit_decision_policy(...) -> dict` and `classify_reliability(...) -> dict`.
- Produces schema-v2 contract fields `crop_labels`, `leaf_validation`, `decision_policy`; schema v1 remains readable.

- [ ] **Step 1: Write failing calibration tests**

```python
def test_leaf_reference_separates_leaf_and_negative(self):
    reference = fit_leaf_reference(
        np.array([[1., 0.], [.98, .02], [0., 1.], [.02, .98]]),
        np.array([0, 0, 1, 1]), self.valid_quality,
        np.array([[-1., 0.], [0., -1.]]), ["Apple", "Tomato"],
    )
    self.assertGreaterEqual(leaf_similarity(np.array([1., 0.]), reference)[0], reference["accept_threshold"])
    self.assertLess(leaf_similarity(np.array([-1., 0.]), reference)[0], reference["retry_threshold"])

def test_reliability_abstains_on_ambiguous_probabilities(self):
    result = classify_reliability(
        np.array([.27, .26, .25, .22]),
        {"min_confidence": .55, "min_margin": .12, "max_entropy": 1.20},
    )
    self.assertEqual(result["analysis_status"], "inconclusive")
```

Add artifact tests proving v1 loads and v2 rejects missing/invalid thresholds.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest ml.tests.test_calibration ml.tests.test_artifacts -v`

Expected: FAIL because calibration and schema-v2 validation are absent.

- [ ] **Step 3: Implement deterministic calibration**

```python
def quality_metrics(image):
    values = np.asarray(image, dtype="float32")
    if values.max() > 1:
        values /= 255.0
    gray = values.mean(axis=-1)
    return {
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "sharpness": float(np.square(np.diff(gray, axis=1)).mean() + np.square(np.diff(gray, axis=0)).mean()),
        "dark_clip": float((gray <= .02).mean()),
        "bright_clip": float((gray >= .98).mean()),
    }

def classify_reliability(probabilities, policy):
    p = np.asarray(probabilities, dtype="float64")
    ordered = np.sort(p)
    confidence, margin = float(ordered[-1]), float(ordered[-1] - ordered[-2])
    entropy = float(-(p * np.log(np.clip(p, 1e-8, 1))).sum())
    completed = confidence >= policy["min_confidence"] and margin >= policy["min_margin"] and entropy <= policy["max_entropy"]
    return {"analysis_status": "completed" if completed else "inconclusive", "confidence": confidence, "margin": margin, "entropy": entropy}
```

Fit quality bounds from valid-leaf percentiles, L2-normalized crop centroids from train leaves, leaf thresholds retaining at least 95% validation leaves while blocking negative fixtures, and reliability thresholds maximizing selective macro-F1 at coverage `>= 0.70`. Bump generated contracts to schema 2 but accept versions 1 and 2 in `validate_contract`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest ml.tests.test_calibration ml.tests.test_artifacts -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/calibration.py ml/artifacts.py ml/utils.py ml/tests
git commit -m "feat: version leaf and reliability calibration"
```

---

### Task 4: Reject unsuitable uploads before persistence

**Files:**
- Create: `backend/services/image_validation_service.py`
- Modify: `backend/services/storage_service.py`
- Modify: `backend/routes/upload.py`
- Create: `backend/tests/test_image_validation_service.py`
- Create: `backend/tests/test_upload_route.py`

**Interfaces:**
- Produces `ImageValidationError` carrying a structured result.
- Produces `ImageValidationService.validate(image_or_path) -> dict`.
- Produces `save_upload(...) -> tuple[UploadedImage, dict]` only for accepted images.

- [ ] **Step 1: Write failing service/route tests**

```python
def test_blank_image_is_rejected_before_embedding(self):
    service = ImageValidationService(
        contract=self.contract,
        embedding_provider=lambda _: self.fail("embedding should not run"),
    )
    result = service.validate(Image.new("RGB", (256, 256), "white"))
    self.assertEqual(result["status"], "rejected")
    self.assertEqual(result["reason_code"], "near_uniform")

def test_non_leaf_upload_returns_422_without_record(self):
    response = self.client.post("/upload", data={
        "image": (self.non_leaf_bytes, "dashboard.png")
    }, content_type="multipart/form-data")
    self.assertEqual(response.status_code, 422)
    self.assertEqual(response.json["reason_code"], "non_leaf")
    self.assertEqual(UploadedImage.query.count(), 0)
```

Also test missing, empty, spoofed, truncated, tiny, dark, bright, blurry, grayscale, transparent, rotated, animated, and valid diseased leaf files.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest backend.tests.test_image_validation_service backend.tests.test_upload_route -v`

Expected: FAIL because the service and 422 path do not exist.

- [ ] **Step 3: Implement decoding, quality, and embedding gates**

```python
class ImageValidationError(ValueError):
    def __init__(self, result):
        super().__init__(result["message"])
        self.result = result

def validate(self, image_or_path):
    image = self._load_rgb(image_or_path)
    metrics = quality_metrics(np.asarray(image))
    if failure := self._quality_failure(image.size, metrics):
        return failure | {"quality": metrics}
    score, crop = leaf_similarity(self.embedding_provider(image), self.reference)
    if score < self.reference["retry_threshold"]:
        status, code = "rejected", "non_leaf"
    elif score < self.reference["accept_threshold"]:
        status, code = "retry_required", "uncertain_leaf"
    else:
        status, code = "accepted", "valid_leaf"
    return {"status": status, "reason_code": code, "message": self._message(code), "guidance": self._guidance(code), "leaf_similarity": score, "inferred_crop": crop, "quality": metrics, "threshold_version": self.reference["version"]}
```

Use Pillow decoded format rather than extension, `ImageOps.exif_transpose`, minimum geometry, animation/decompression-bomb rejection, and rollback/deletion on partial persistence failure. Save and commit only accepted uploads. Return structured 422 for `ImageValidationError`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest backend.tests.test_image_validation_service backend.tests.test_upload_route -v`

Expected: PASS and rejected files leave no database/file residue.

- [ ] **Step 5: Commit**

```bash
git add backend/services backend/routes/upload.py backend/tests
git commit -m "feat: block unsuitable image uploads"
```

---

### Task 5: Persist completed or inconclusive analyses safely

**Files:**
- Create: `backend/services/schema_service.py`
- Create: `backend/services/analysis_service.py`
- Modify: `backend/models/db_models.py`
- Modify: `backend/services/inference_service.py`
- Modify: `backend/routes/predict.py`
- Modify: `backend/app.py`
- Create: `backend/tests/test_analysis_service.py`
- Create: `backend/tests/test_predict_route.py`

**Interfaces:**
- Produces: `ensure_schema_compatibility(engine) -> None` with additive columns only.
- Produces: `AnalysisService.build(prediction, image_validation, selected_crop, readings) -> dict`.
- Produces fields `analysis_status`, `reliability`, `image_validation`, `crop_consistency`, `observations`, `recommendations`.

- [ ] **Step 1: Write failing analysis/migration/predict tests**

```python
def test_ambiguous_prediction_is_inconclusive(self):
    analysis = AnalysisService.build(
        self.prediction([.27, .26, .25, .22]), self.accepted_leaf,
        "Tomato", self.readings,
    )
    self.assertEqual(analysis["analysis_status"], "inconclusive")
    self.assertNotIn("diagnosis", " ".join(analysis["recommendations"]).lower())

def test_predict_revalidates_upload(self):
    self.validator.return_value = {"status": "rejected", "reason_code": "non_leaf"}
    response = self.client.post("/predict", json={"upload_id": self.upload.id})
    self.assertEqual(response.status_code, 422)
    self.assertEqual(Prediction.query.count(), 0)
```

Create an old-schema SQLite fixture, run `ensure_schema_compatibility`, and assert that existing prediction rows remain while the new nullable columns appear.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest backend.tests.test_analysis_service backend.tests.test_predict_route -v`

Expected: FAIL because analysis status, revalidation, and additive migration are absent.

- [ ] **Step 3: Implement backward-compatible analysis**

Add nullable JSON/status columns to uploaded images and predictions. Use SQLAlchemy inspection and exact `ALTER TABLE ... ADD COLUMN` statements only for missing columns; call the migration after `db.create_all()`.

Update inference for dictionary outputs:

```python
raw = ModelSingleton.get().predict({
    "image": images, "sensor_sequence": sensors,
}, verbose=0)
stress = np.asarray(raw["stress_probabilities"][0], dtype="float32")
crop = np.asarray(raw["crop_probabilities"][0], dtype="float32")
```

Build calibrated analysis and crop consistency:

```python
reliability = classify_reliability(stress, contract["decision_policy"])
crop_index = int(np.argmax(crop))
inferred_crop = contract["crop_labels"][crop_index]
crop_consistency = {
    "selected": selected_crop, "inferred": inferred_crop,
    "matches": normalize_crop(selected_crop) == normalize_crop(inferred_crop),
    "confidence": float(crop[crop_index]),
}
recommendations = (
    ["Retake a clear centered leaf photo and verify the sensor readings."]
    if reliability["analysis_status"] == "inconclusive"
    else safe_recommendations(predicted_class, crop_consistency, readings)
)
```

Revalidate before inference and return 422 without persistence if blocked. Persist an inconclusive row with the audit top candidate in the legacy non-null class field, but expose `analysis_status` so clients never present it as the conclusion.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py' -v`

Expected: PASS including old-row preservation, completed, inconclusive, and revalidation paths.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/models backend/routes/predict.py backend/services backend/tests
git commit -m "feat: return trustworthy stress analyses"
```

---

### Task 6: Render blocked, inconclusive, and completed states

**Files:**
- Create: `frontend/src/utils/analysis.js`
- Create: `frontend/tests/analysis.test.js`
- Create: `frontend/tests/analysis-flow.spec.js`
- Create: `frontend/playwright.config.js`
- Modify: `frontend/package.json`
- Modify: `frontend/src/components/Primitives.jsx`
- Modify: `frontend/src/pages/Upload.jsx`
- Modify: `frontend/src/pages/Prediction.jsx`
- Modify: `frontend/src/utils/api.js`

**Interfaces:**
- Produces: `uploadErrorDetails(error)` and `isConclusive(analysis)`.
- Consumes structured 422 upload errors and Task 5 analysis fields.

- [ ] **Step 1: Write failing Node utility tests**

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { isConclusive, uploadErrorDetails } from '../src/utils/analysis.js';

test('maps non-leaf 422 to actionable blocked state', () => {
  const result = uploadErrorDetails({ response: { data: {
    status: 'rejected', reason_code: 'non_leaf',
    message: 'This does not look like a leaf.', guidance: ['Upload one leaf.'],
  } } });
  assert.equal(result.status, 'rejected');
  assert.equal(result.reasonCode, 'non_leaf');
  assert.deepEqual(result.guidance, ['Upload one leaf.']);
});

test('does not treat inconclusive analysis as a prediction', () => {
  assert.equal(isConclusive({ analysis_status: 'inconclusive' }), false);
});
```

- [ ] **Step 2: Verify RED**

Run: `node --test frontend/tests/analysis.test.js`

Expected: FAIL because `analysis.js` is absent.

- [ ] **Step 3: Implement state mapping and pages**

```javascript
export function uploadErrorDetails(error) {
  const data = error?.response?.data || {};
  return {
    status: data.status || 'error',
    reasonCode: data.reason_code || 'upload_failed',
    message: data.message || data.error || error?.message || 'Analysis failed.',
    guidance: Array.isArray(data.guidance) ? data.guidance : [],
  };
}
export const isConclusive = (analysis) => analysis?.analysis_status === 'completed';
```

On Upload, show validation progress and a red/amber alert with reason and guidance; retain an indicated preview; never call `/predict` or navigate after 422; revoke old object URLs. Replace the crop selector with the exact nine `CROP_LABELS` values so unsupported Soybean cannot create a false mismatch. On Prediction, show an amber “Analysis inconclusive” state with retry guidance and no primary stress badge. Completed analyses show reliability, crop consistency, observations, recommendations, quality, probabilities, activation, and telemetry evidence.

- [ ] **Step 4: Verify utility and build GREEN**

Run: `node --test frontend/tests/analysis.test.js`

Run: `npm run build --prefix frontend`

Expected: tests and production build pass.

- [ ] **Step 5: Add and run rendered browser flows**

```javascript
test('non-leaf alert blocks prediction navigation', async ({ page }) => {
  await page.route('**/upload', route => route.fulfill({ status: 422, json: {
    status: 'rejected', reason_code: 'non_leaf',
    message: 'This image does not look like a leaf.',
    guidance: ['Upload one clear leaf centered in the frame.'],
  }}));
  await page.goto('/upload');
  await page.setInputFiles('input[type=file]', 'src/assets/agrisense-ui-concept.png');
  await page.getByRole('button', { name: 'Analyze Now' }).click();
  await expect(page.getByText('This image does not look like a leaf.')).toBeVisible();
  await expect(page).toHaveURL(/\/upload$/);
});
```

Run: `cd frontend && npx playwright install chromium && npx playwright test`

Expected: blocked, accepted, and inconclusive flows pass with no console errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/playwright.config.js frontend/src frontend/tests
git commit -m "feat: show blocked and inconclusive analyses"
```

---

### Task 7: Calibrate, evaluate, retrain, and conditionally promote

**Files:**
- Modify: `ml/evaluate.py`
- Modify: `ml/train.py`
- Modify: `ml/tests/test_evaluate.py`
- Regenerate: `ml/reports/*`
- Generate locally: `ml/saved_model/agrisense_cnn_lstm.keras`
- Generate locally: `ml/saved_model/preprocessing.json`

**Interfaces:**
- Produces metric sections `normal`, `image_only`, `sensor_only`, `selective`, `leaf_validation`, `negative_suite`, `crop`.
- Produces `enforce_acceptance(metrics) -> None` covering all global gates.

- [ ] **Step 1: Write failing acceptance tests**

```python
def test_acceptance_requires_accuracy_improvement(self):
    metrics = self.accepted_metrics
    metrics["normal"]["accuracy"] = 0.8192090395480226
    with self.assertRaisesRegex(RuntimeError, "accuracy"):
        enforce_acceptance(metrics)

def test_acceptance_rejects_any_non_leaf_false_accept(self):
    metrics = self.accepted_metrics
    metrics["negative_suite"] = {"blocked": 4, "samples": 5, "all_blocked": False}
    with self.assertRaisesRegex(RuntimeError, "non-leaf"):
        enforce_acceptance(metrics)
```

Add independent tests for macro-F1, Brier, per-class recall collapse, modality reliance, leaf false rejection, missing crop classes, and invalid distributions.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest ml.tests.test_evaluate -v`

Expected: FAIL because complete baseline and validation gates are absent.

- [ ] **Step 3: Implement calibration/evaluation before promotion**

Extract train/validation embeddings and quality rows; generate deterministic solid/noise/checkerboard/document/UI negatives; fit leaf and decision policies; write schema-v2 staging contract; evaluate stress, crop, modalities, selective coverage, held-out valid leaves, and negatives.

```python
def enforce_acceptance(metrics):
    failures = []
    if metrics["normal"]["accuracy"] <= 0.8192090395480226:
        failures.append("accuracy did not beat baseline")
    if metrics["normal"]["f1_macro"] <= 0.7901860901925868:
        failures.append("macro-F1 did not beat baseline")
    if metrics["normal"]["brier_score"] > 0.2740994393825531:
        failures.append("Brier score regressed")
    if metrics["image_only"]["f1_macro"] <= metrics["sensor_only"]["f1_macro"]:
        failures.append("image-only F1 must exceed sensor-only F1")
    if metrics["leaf_validation"]["false_rejection_rate"] > .05:
        failures.append("leaf false rejection exceeds 5%")
    if not metrics["negative_suite"]["all_blocked"]:
        failures.append("non-leaf suite was not fully blocked")
    if failures:
        raise RuntimeError("Candidate rejected: " + "; ".join(failures))
```

Write rejected candidate metrics separately. Call `promote_bundle` only after acceptance.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest ml.tests.test_evaluate -v`

Expected: PASS.

- [ ] **Step 5: Retrain from scratch**

```bash
SSL_CERT_FILE=.venv/lib/python3.12/site-packages/certifi/cacert.pem \
.venv/bin/python ml/train.py --epochs 12 --fine-tune-epochs 12 --batch-size 16 --sequence-length 7
```

Expected: candidate report is always generated; promotion occurs only if every gate passes. If rejected, tune one documented variable per run without weakening gates.

- [ ] **Step 6: Evaluate promoted bundle and commit**

Run: `.venv/bin/python ml/evaluate.py --sequence-length 7 --batch-size 16`

Expected: improved schema-v2 bundle, all negative fixtures blocked, leaf false rejection at most 5%, and selective coverage reported.

```bash
git add ml/evaluate.py ml/train.py ml/tests/test_evaluate.py ml/reports
git commit -m "feat: promote calibrated trustworthy model"
```

---

### Task 8: Execute every edge path, document, and publish

**Files:**
- Modify: `backend/tests/smoke_test.py`
- Modify: `README.md`
- Modify: `dataset/README_dataset.md`
- Modify: `frontend/src/pages/Model.jsx`
- Modify: `frontend/src/pages/About.jsx`

**Interfaces:**
- Produces end-to-end accepted-leaf, rejected-non-leaf, and inconclusive smoke evidence.
- Documents actual promoted metrics only.

- [ ] **Step 1: Extend smoke coverage**

```python
valid = assert_status(valid_upload, 201, "valid-leaf-upload")
with (PROJECT_ROOT / "frontend/src/assets/agrisense-ui-concept.png").open("rb") as handle:
    non_leaf = client.post("/upload", data={"image": (handle, "dashboard.png")}, content_type="multipart/form-data")
assert_status(non_leaf, 422, "non-leaf-upload")
assert inconclusive["analysis_status"] == "inconclusive"
```

- [ ] **Step 2: Run full verification**

Run separately and require exit 0:

```bash
.venv/bin/python -m unittest discover -s ml/tests -v
.venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py' -v
.venv/bin/python backend/tests/smoke_test.py
node --test frontend/tests/analysis.test.js
npm run build --prefix frontend
cd frontend && npx playwright test
.venv/bin/python -m compileall -q ml backend dataset
git diff --check
```

- [ ] **Step 3: Update evidence-based documentation**

Read `ml/reports/evaluation_metrics.json` and document exact full-coverage accuracy/F1/Brier, selective coverage/accuracy, modality metrics, leaf false rejection, and negative-suite results. Document 422 errors, inconclusive analyses, schema-v2 promotion, retraining, crop evidence, and research limitations.

- [ ] **Step 4: Start and exercise the real app**

```bash
cd backend && ../.venv/bin/python -c "from app import app; app.run(host='127.0.0.1', port=5001, debug=False)"
cd frontend && VITE_API_BASE_URL=http://127.0.0.1:5001 npm run dev
```

Verify health, a valid leaf analysis, non-leaf alert, completed/inconclusive UI, and no browser console errors at `http://127.0.0.1:5173`.

- [ ] **Step 5: Commit and publish only after verification**

```bash
git add README.md dataset/README_dataset.md backend/tests/smoke_test.py frontend/src/pages/Model.jsx frontend/src/pages/About.jsx
git commit -m "docs: document trustworthy agrisense analysis"
git status --short --branch
git push origin main
```
