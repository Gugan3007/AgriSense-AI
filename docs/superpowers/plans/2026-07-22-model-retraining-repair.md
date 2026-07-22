# AgriSense Model Retraining Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and retrain AgriSense as an image-first, sensor-assisted classifier whose evaluation matches the one-image production flow and whose API exposes only bounded probabilities and sensor trend scores.

**Architecture:** A single current leaf image passes through a frozen MobileNetV2 feature extractor while seven normalized sensor readings pass through a small LSTM. Separate image and sensor probability heads are trained with auxiliary losses, and the production probability is a fixed 80/20 image/sensor mixture. Dataset construction forbids source-image reuse, an artifact contract travels with the saved model, and evaluation measures each modality independently before promotion.

**Tech Stack:** Python 3.12, TensorFlow/Keras 2.20, NumPy, pandas, scikit-learn, Pillow, Flask, React/Vite, built-in `unittest`.

## Global Constraints

- Preserve the public classes `Healthy`, `Low`, `Medium`, and `High` in that exact order.
- Treat leaf imagery as the primary modality and synthetic sensor telemetry as supporting evidence.
- Use one current leaf image and exactly seven sensor readings at the model boundary.
- Keep probabilities finite, bounded to `[0, 1]`, and summing to one.
- Keep sensor stress chart values bounded to `0–100`.
- Never replace the current working model until the staged model and contract pass validation.
- Continue documenting the system as an educational/research prototype with synthetic telemetry.

---

### Task 1: Dataset integrity and production-aligned preprocessing

**Files:**
- Create: `ml/tests/__init__.py`
- Create: `ml/tests/test_preprocessing.py`
- Modify: `dataset/build_sequences.py`
- Modify: `ml/preprocessing.py`

**Interfaces:**
- Produces: `validate_split_integrity(frame, splits) -> None`
- Produces: `make_windows(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]` where image paths have shape `(samples,)`, sensors `(samples, 7, 4)`, and labels `(samples,)`.
- Produces: a generated CSV in which `Image_Path` is globally unique.

- [ ] **Step 1: Write failing preprocessing tests**

```python
def test_make_windows_uses_only_the_current_leaf_image(self):
    images, sensors, labels = make_windows(self.frame, ["P1"], 7, self.normalization)
    self.assertEqual(images.shape, (1,))
    self.assertEqual(images[0], self.frame.iloc[-1]["Resolved_Image_Path"])
    self.assertEqual(sensors.shape, (1, 7, 4))

def test_split_integrity_rejects_source_image_overlap(self):
    with self.assertRaisesRegex(ValueError, "image leakage"):
        validate_split_integrity(self.overlapping_frame, self.splits)
```

- [ ] **Step 2: Verify the tests fail for the current sequence-shaped image output and missing validator**

Run: `.venv/bin/python -m unittest ml.tests.test_preprocessing -v`
Expected: FAIL because image paths currently have shape `(samples, 7)` and `validate_split_integrity` does not exist.

- [ ] **Step 3: Implement unique image selection and integrity validation**

Add per-class shuffled image pools in `dataset/build_sequences.py` and pop each selected image so a source path cannot be reused. Widen the synthetic sensor distributions so adjacent classes overlap and smooth each plant's readings over time. In `ml/preprocessing.py`, validate plant separation, source-image separation, and all-class coverage after splitting. Change each window to store only its last row's resolved image path while retaining all seven sensor rows.

- [ ] **Step 4: Rebuild the CSV and verify unit tests pass**

Run: `.venv/bin/python dataset/build_sequences.py`
Expected: JSON reports at least 60 plants, all 38 image classes, and zero reused source images.

Run: `.venv/bin/python -m unittest ml.tests.test_preprocessing -v`
Expected: PASS.

- [ ] **Step 5: Commit the isolated dataset/preprocessing repair**

```bash
git add dataset/build_sequences.py dataset/sequential_data.csv dataset/README_dataset.md ml/preprocessing.py ml/tests
git commit -m "fix: align dataset with production inference"
```

### Task 2: Versioned preprocessing contract and safe artifact promotion

**Files:**
- Create: `ml/artifacts.py`
- Create: `ml/tests/test_artifacts.py`
- Modify: `ml/utils.py`

**Interfaces:**
- Produces: `build_contract(statistics) -> dict`
- Produces: `load_contract(path=MODEL_CONTRACT_PATH) -> dict`
- Produces: `validate_contract(contract) -> None`
- Produces: `promote_bundle(staged_model, staged_contract) -> None`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_contract_rejects_wrong_sensor_order(self):
    contract = self.valid_contract | {"sensor_columns": list(reversed(SENSOR_COLUMNS))}
    with self.assertRaisesRegex(ValueError, "sensor column order"):
        validate_contract(contract)

def test_promotion_replaces_model_and_contract_together(self):
    promote_bundle(self.staged_model, self.staged_contract, self.model_path, self.contract_path)
    self.assertEqual(self.model_path.read_bytes(), b"new-model")
    self.assertEqual(load_contract(self.contract_path)["schema_version"], 1)
```

- [ ] **Step 2: Verify tests fail because the artifact module is absent**

Run: `.venv/bin/python -m unittest ml.tests.test_artifacts -v`
Expected: FAIL with an import error for `artifacts`.

- [ ] **Step 3: Implement the contract and atomic same-directory promotion**

The contract must include `schema_version`, `class_labels`, `sensor_columns`, `sensor_mean`, `sensor_std`, `image_size`, `sequence_length`, and input names `image` and `sensor_sequence`. Validate exact orders, positive standard deviations, and dimensions. Validate staged files before using `Path.replace` to promote them.

- [ ] **Step 4: Verify artifact tests pass**

Run: `.venv/bin/python -m unittest ml.tests.test_artifacts -v`
Expected: PASS.

- [ ] **Step 5: Commit contract support**

```bash
git add ml/artifacts.py ml/utils.py ml/tests/test_artifacts.py
git commit -m "feat: version model preprocessing artifacts"
```

### Task 3: Image-first hybrid model and training loop

**Files:**
- Create: `ml/tests/test_model.py`
- Modify: `ml/model.py`
- Modify: `ml/train.py`
- Modify: `ml/tune.py`

**Interfaces:**
- Produces: `build_models(sequence_length=7, image_weights="imagenet") -> tuple[training_model, inference_model]`
- Produces: training outputs named `stress_probabilities`, `image_probabilities`, and `sensor_probabilities`.
- Produces: inference output `stress_probabilities = 0.8 * image_probabilities + 0.2 * sensor_probabilities`.

- [ ] **Step 1: Write failing architecture tests using `image_weights=None`**

```python
def test_inference_model_uses_single_image_and_sensor_sequence(self):
    _, model = build_models(image_weights=None)
    self.assertEqual(tuple(model.get_layer("image").batch_shape[1:]), (128, 128, 3))
    self.assertEqual(tuple(model.get_layer("sensor_sequence").batch_shape[1:]), (7, 4))

def test_fused_output_is_an_image_first_probability_mixture(self):
    training, inference = build_models(image_weights=None)
    self.assertEqual(set(training.output_names), {
        "stress_probabilities", "image_probabilities", "sensor_probabilities"
    })
    probabilities = inference(self.sample_inputs, training=False).numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Verify tests fail because `build_models` is absent**

Run: `.venv/bin/python -m unittest ml.tests.test_model -v`
Expected: FAIL importing `build_models`.

- [ ] **Step 3: Implement the two-branch model**

Use MobileNetV2 without its classification top, global average pooling, and a 128-unit image feature layer. Use Gaussian noise, a 32-unit LSTM, and whole-modality dropout for sensor features. Add separate four-class softmax heads, scale them by `0.8` and `0.2` with serializable Keras layers, and add them into `stress_probabilities`. Compile the training model with categorical cross-entropy on all three outputs and loss weights `1.0`, `0.5`, and `0.1` respectively.

- [ ] **Step 4: Update training for balanced classes and staged output**

Calculate inverse-frequency class weights from training labels. Map each one-hot target to all three named training outputs. Monitor `val_stress_probabilities_loss`, restore the best weights, save the inference model to a staging file, write its staging contract, reload both with `compile=False`, and leave the current production bundle untouched until evaluation accepts the stage.

- [ ] **Step 5: Verify model tests and a one-step training QA run**

Run: `.venv/bin/python -m unittest ml.tests.test_model -v`
Expected: PASS.

Run: `.venv/bin/python ml/train.py --epochs 1 --max-steps 1 --qa-only`
Expected: one training step completes and staged QA artifacts load without changing the production model.

- [ ] **Step 6: Commit the model/training repair**

```bash
git add ml/model.py ml/train.py ml/tune.py ml/tests/test_model.py
git commit -m "fix: make hybrid model image first"
```

### Task 4: Honest evaluation and modality acceptance gate

**Files:**
- Create: `ml/tests/test_evaluate.py`
- Modify: `ml/evaluate.py`

**Interfaces:**
- Produces: `validate_probabilities(probabilities) -> None`
- Produces: `evaluate_predictions(y_true, probabilities) -> dict`
- Produces modality metrics `normal`, `image_only`, and `sensor_only` plus counterfactual sensitivity and calibration values.

- [ ] **Step 1: Write failing probability and modality tests**

```python
def test_probability_validation_rejects_negative_values(self):
    with self.assertRaisesRegex(ValueError, "outside"):
        validate_probabilities(np.array([[1.1, -0.1, 0.0, 0.0]], dtype="float32"))

def test_acceptance_requires_image_only_to_beat_sensor_only(self):
    with self.assertRaisesRegex(RuntimeError, "image-only"):
        enforce_acceptance({"image_only": {"f1_macro": 0.4}, "sensor_only": {"f1_macro": 0.8}})
```

- [ ] **Step 2: Verify tests fail because the validators are absent**

Run: `.venv/bin/python -m unittest ml.tests.test_evaluate -v`
Expected: FAIL importing the new validation functions.

- [ ] **Step 3: Implement evaluation modes and acceptance**

Evaluate normal inputs, image-only inputs with normalized sensors set to zero, and sensor-only inputs with neutral images. Compute accuracy, macro precision/recall/F1, multiclass Brier score, mean maximum confidence, and mean counterfactual probability change when one modality is permuted. Reject non-finite or invalid distributions, missing test classes, and staged models whose image-only macro F1 does not exceed sensor-only macro F1.

- [ ] **Step 4: Verify evaluation tests pass**

Run: `.venv/bin/python -m unittest ml.tests.test_evaluate -v`
Expected: PASS.

- [ ] **Step 5: Commit evaluation safeguards**

```bash
git add ml/evaluate.py ml/tests/test_evaluate.py
git commit -m "feat: evaluate model modality reliance"
```

### Task 5: Contract-driven inference and bounded sensor trends

**Files:**
- Create: `backend/tests/test_inference_service.py`
- Modify: `ml/inference.py`
- Modify: `backend/services/inference_service.py`
- Modify: `frontend/src/components/Charts.jsx`
- Modify: `frontend/src/pages/Prediction.jsx`

**Interfaces:**
- `predict_stress(image, sensor_sequence) -> dict`
- `InferenceService.lstm_trend(readings) -> {direction, slope, stress_score}` with seven scores in `[0, 100]`.

- [ ] **Step 1: Write failing inference-service tests**

```python
def test_temporal_scores_are_bounded(self):
    result = InferenceService.lstm_trend(self.readings)
    self.assertTrue(all(0 <= value <= 100 for value in result["stress_score"]))

def test_non_finite_sensor_value_is_rejected(self):
    readings = [self.readings[0] | {"Temperature": float("nan")}]
    with self.assertRaisesRegex(ValueError, "finite"):
        InferenceService.normalize_sensor_readings(readings)
```

- [ ] **Step 2: Verify tests fail against the negative `stress_proxy` implementation**

Run: `.venv/bin/python -m unittest backend.tests.test_inference_service -v`
Expected: FAIL because `stress_score` is absent and NaN is accepted.

- [ ] **Step 3: Implement contract-driven preprocessing and bounded scoring**

Load sensor normalization and input dimensions from the promoted contract. Accept one image instead of a repeated image list. Reject non-finite values and values outside schema ranges. Convert soil moisture, temperature, humidity, and light into a clipped `0–100` daily stress score, then calculate direction from its slope. Keep the response key `lstm_trend` while replacing `stress_proxy` with `stress_score`.

- [ ] **Step 4: Update activation probing and chart labels**

Probe the nested image encoder with the single image input. Render `stress_score` in the frontend and label the axis/card as `Sensor stress score (0–100)`.

- [ ] **Step 5: Verify backend tests and frontend build**

Run: `.venv/bin/python -m unittest backend.tests.test_inference_service -v`
Expected: PASS.

Run: `npm run build --prefix frontend`
Expected: Vite production build succeeds.

- [ ] **Step 6: Commit inference and UI corrections**

```bash
git add ml/inference.py backend/services/inference_service.py backend/tests/test_inference_service.py frontend/src/components/Charts.jsx frontend/src/pages/Prediction.jsx
git commit -m "fix: bound inference outputs and sensor trends"
```

### Task 6: Full retraining, report regeneration, documentation, and execution

**Files:**
- Modify: `README.md`
- Modify: `dataset/README_dataset.md`
- Regenerate: `ml/reports/*`
- Regenerate: `ml/saved_model/agrisense_cnn_lstm.keras`
- Generate locally (gitignored): `ml/saved_model/preprocessing.json`

**Interfaces:**
- Produces a promoted, loadable inference bundle and current evaluation reports.

- [ ] **Step 1: Run the complete unit suite before the expensive training run**

Run: `.venv/bin/python -m unittest discover -s ml/tests -v`
Expected: PASS.

Run: `.venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py' -v`
Expected: PASS.

- [ ] **Step 2: Retrain from scratch and promote only an accepted model**

Run: `.venv/bin/python ml/train.py --epochs 18 --batch-size 16 --sequence-length 7`
Expected: training restores its best validation weights, staged modality evaluation passes, the model and contract are promoted together, and all reports are regenerated.

- [ ] **Step 3: Inspect generated metrics and counterfactual behavior**

Run: `.venv/bin/python ml/evaluate.py --sequence-length 7 --batch-size 16`
Expected: valid probabilities, all test classes, and image-only macro F1 greater than sensor-only macro F1.

- [ ] **Step 4: Execute the real API smoke test and frontend build**

Run: `.venv/bin/python backend/tests/smoke_test.py`
Expected: upload, real prediction, model metadata, history, and contact checks succeed.

Run: `npm run build --prefix frontend`
Expected: Vite production build succeeds.

- [ ] **Step 5: Update documentation with the final measured results**

Document the single-image plus seven-sensor contract, synthetic-sensor limitation, safe artifact promotion, modality metrics, retraining command, and actual final accuracy/F1 values from `evaluation_metrics.json`.

- [ ] **Step 6: Run final verification and commit generated results**

Run: `git diff --check && git status --short`
Expected: no whitespace errors and only intended model-repair files are changed.

```bash
git add README.md dataset ml backend frontend/src ml/reports
git commit -m "feat: retrain reliable agrisense model"
```

The promoted `.keras` model and `preprocessing.json` remain local runtime artifacts under the gitignored `ml/saved_model/` directory; evaluation reports and metadata remain reviewable tracked artifacts.
