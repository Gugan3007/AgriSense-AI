# AgriSense AI sequential dataset

This dataset combines real PlantVillage leaf images with a reproducible simulated
timeline and plausible plant-health telemetry for CNN-LSTM experimentation.

## Provenance and honesty

**Real:** image files, crop/disease folder labels, and the number of source images.

**Simulated:** virtual plant identity, calendar date, placement of independent
PlantVillage images into a plant progression, stress grouping, health status, and
all four sensor values. PlantVillage does not track individual plants over time and
does not include sensor readings. Therefore these sequences must not be described
as real longitudinal field observations or used for clinical/agronomic decisions.

The explicit disease-to-stress mapping and sensor distributions are in
`build_sequences.py`. A fixed random seed (42) makes generation reproducible.
Stress placement is non-decreasing by severity rank; sensor profile means move with
stress (not pure noise), with seeded Gaussian variation and safe physical bounds.

## Generated snapshot

- Virtual plants: 64
- Plant-day rows: 1593
- Sequence length: 20–30 days
- Crops used: Apple, Cherry_(including_sour), Corn_(maize), Grape, Peach, Pepper,_bell, Potato, Strawberry, Tomato
- Stress distribution: {"Healthy": 623, "High": 232, "Low": 215, "Medium": 523}
- Source images available: 54305
- Reused source images: 0 (must remain zero)

`Image_Path` references the original files beneath `dataset/plantvillage_raw/`;
images are not duplicated. Paths are relative to the project root for portability.

## Rebuild

From the project root:

```bash
python3 dataset/build_sequences.py
```

Use `--plants`, `--min-days`, `--max-days`, or `--seed` to change generation.
The script validates that the explicit mapping exactly matches discovered classes.
