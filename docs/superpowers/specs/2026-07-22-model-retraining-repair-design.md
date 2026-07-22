# AgriSense Model Retraining Repair Design

## Objective

Retrain AgriSense so its stress prediction is driven primarily by real leaf-image evidence, uses the synthetic seven-reading sensor sequence only as supporting context, never exposes negative probability or trend values, and reports evaluation results that reflect production behavior.

## Confirmed root causes

The current saved model changes class almost entirely when the sensor profile changes and remains effectively unchanged when the leaf image changes. The sequence generator creates sensor readings directly from the target stress class, so the model can minimize its loss by learning that shortcut. The current evaluation repeats the same synthetic relationship and therefore reports high accuracy without demonstrating visual learning.

Training also consumes seven different leaf images per window, while the web application supplies one uploaded leaf. Repeating that upload during inference creates an input distribution the model did not see during training. Exact source images may also be reused by different simulated plants, so splitting only by virtual `Plant_ID` does not guarantee image isolation.

The negative values visible in the temporal chart are not softmax probabilities. They come from an unbounded hand-written stress-proxy formula in the backend.

## Selected approach

The repaired model will be image-first and sensor-assisted.

The image branch will consume the single current leaf image used by the application. A compact pretrained image encoder with global pooling will replace the current large flatten-based CNN. This removes redundant computation and aligns training, evaluation, and production with the same input contract.

The sensor branch will consume the seven normalized readings through an LSTM. Sensor noise and modality dropout will prevent the branch from becoming a guaranteed label lookup. Class balancing and early stopping will remain part of training. The two representations will be fused only after each branch has learned a bounded feature vector.

Image learning will be protected by an image-only auxiliary classification objective during training. The production model will expose only the fused four-class softmax output.

## Dataset and split integrity

Dataset construction will sample source images without replacement within a generated snapshot. Preprocessing will validate that no resolved image path appears in more than one of the train, validation, and test partitions. The split will continue to isolate virtual plants and will also verify that every partition contains every stress class.

Synthetic sensor telemetry will remain explicitly documented as simulated. Its class profiles will overlap more strongly, and training-time sensor perturbation will make it supporting evidence instead of a deterministic encoding of the label.

The generated dataset, schema, and dataset README will be rebuilt from the checked-in PlantVillage image folders before retraining.

## Training and artifact consistency

Training will write the best model and its exact preprocessing contract together. The contract will contain class order, sensor column order, sensor means and standard deviations, image size, sequence length, and model input names. Inference and evaluation will read this saved contract rather than silently relying on independent constants or a potentially stale report file.

A failed or interrupted training run must not replace the last working inference bundle. New artifacts will be written to a staging location and promoted only after model loading and contract validation succeed.

Reports will be regenerated from the promoted model. They will include the standard classification metrics plus modality diagnostics:

- normal test accuracy and macro F1;
- image-only performance with sensors replaced by training means;
- sensor-only performance with images neutralized;
- image counterfactual sensitivity at fixed sensors;
- sensor counterfactual sensitivity at fixed images;
- probability validity and calibration summary.

The model is accepted only when all probabilities are finite, each probability is within `[0, 1]`, every row sums to approximately one, all four classes occur in the test labels, and image-only macro F1 exceeds sensor-only macro F1. Exact metric values will be reported rather than hard-coded into the application.

## Inference and dashboard behavior

The backend will validate sensor values against the physical ranges in the dataset schema and reject non-finite values. Short and long sensor sequences will retain the existing pad/trim behavior.

The temporal visualization will use a documented stress score bounded to `0–100`. Its direction will be calculated from the bounded daily scores. The API will retain the `lstm_trend` object for compatibility, but its values will no longer be negative.

The prediction API will continue returning four class probabilities and confidence as fractions from zero to one. The frontend will continue converting only class probabilities to percentages; the temporal graph will be labeled as a sensor stress score so it cannot be confused with model probability.

## Testing strategy

Tests will be added before each behavior change. They will cover:

- source-image isolation across data partitions;
- all-class coverage and class-balancing calculations;
- the production-aligned single-image input shape;
- preprocessing-contract round trips and mismatch rejection;
- finite sensor validation and physical-range rejection;
- bounded `0–100` temporal scores and correct trend direction;
- probability bounds and sum-to-one behavior;
- loading the promoted model bundle;
- counterfactual modality diagnostics on the trained model;
- backend smoke behavior and frontend production build.

After unit tests pass, the dataset will be rebuilt, the model retrained from scratch, all reports regenerated, representative counterfactual predictions inspected, the backend smoke test run, and the frontend production build completed.

## Documentation and limitations

The main README and dataset documentation will describe the new input contract, retraining command, artifact promotion behavior, modality diagnostics, and the fact that sensor telemetry remains synthetic. The application will remain clearly identified as an educational/research prototype rather than a field-validated diagnostic tool.

## Out of scope

This repair will not claim agronomic validation, create real longitudinal sensor observations, add a seven-image upload interface, or diagnose the original PlantVillage disease class. Those require new real-world data or a separate product feature.
