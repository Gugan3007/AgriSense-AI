#!/usr/bin/env python3
"""Build honest, reproducible temporal sequences from PlantVillage images.

The source images and their disease labels are real. Plant IDs, timestamps, the
ordering of images into a progression, and sensor readings are simulated.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


SEED = 42
STRESS_LEVELS = ("Healthy", "Low", "Medium", "High")
SEVERITY_RANK = {label: rank for rank, label in enumerate(STRESS_LEVELS)}

# Explicit, reviewable mapping for all 38 PlantVillage classes in this dataset.
# Severity is a project-level stress grouping, not a clinical disease stage.
DISEASE_TO_STRESS = {
    "Apple___Apple_scab": "Medium",
    "Apple___Black_rot": "High",
    "Apple___Cedar_apple_rust": "Low",
    "Apple___healthy": "Healthy",
    "Blueberry___healthy": "Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Medium",
    "Cherry_(including_sour)___healthy": "Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Medium",
    "Corn_(maize)___Common_rust_": "Low",
    "Corn_(maize)___Northern_Leaf_Blight": "High",
    "Corn_(maize)___healthy": "Healthy",
    "Grape___Black_rot": "High",
    "Grape___Esca_(Black_Measles)": "High",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Medium",
    "Grape___healthy": "Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "High",
    "Peach___Bacterial_spot": "Medium",
    "Peach___healthy": "Healthy",
    "Pepper,_bell___Bacterial_spot": "Medium",
    "Pepper,_bell___healthy": "Healthy",
    "Potato___Early_blight": "Low",
    "Potato___Late_blight": "High",
    "Potato___healthy": "Healthy",
    "Raspberry___healthy": "Healthy",
    "Soybean___healthy": "Healthy",
    "Squash___Powdery_mildew": "Medium",
    "Strawberry___Leaf_scorch": "Medium",
    "Strawberry___healthy": "Healthy",
    "Tomato___Bacterial_spot": "Medium",
    "Tomato___Early_blight": "Low",
    "Tomato___Late_blight": "High",
    "Tomato___Leaf_Mold": "Medium",
    "Tomato___Septoria_leaf_spot": "Medium",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Low",
    "Tomato___Target_Spot": "Medium",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "High",
    "Tomato___Tomato_mosaic_virus": "High",
    "Tomato___healthy": "Healthy",
}

HEALTH_STATUS = {
    "Healthy": "Healthy",
    "Low": "Watch",
    "Medium": "Stressed",
    "High": "Critical",
}

SENSOR_PROFILES = {
    # Deliberately overlapping distributions keep synthetic telemetry from
    # becoming a deterministic encoding of the target class.
    # mean soil %, temperature C, humidity %, light lux; standard deviations
    "Healthy": ((58.0, 25.0, 62.0, 15000.0), (10.0, 3.0, 10.0, 3500.0)),
    "Low": ((53.0, 26.0, 59.0, 15500.0), (10.0, 3.0, 10.0, 3500.0)),
    "Medium": ((48.0, 28.0, 55.0, 16250.0), (10.0, 3.0, 10.0, 3500.0)),
    "High": ((43.0, 29.0, 51.0, 17000.0), (10.0, 3.0, 10.0, 3500.0)),
}


def locate_class_root(raw_root: Path) -> Path:
    """Support both downloaded layouts: raw/Class and raw/PlantVillage/Class."""
    nested = raw_root / "PlantVillage"
    candidates = [nested, raw_root] if nested.is_dir() else [raw_root]
    for candidate in candidates:
        if any(p.is_dir() and "___" in p.name for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"No PlantVillage class folders found in {raw_root} or {nested}"
    )


def inspect_dataset(class_root: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    extensions = {".jpg", ".jpeg", ".png"}
    for folder in sorted(p for p in class_root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
        )
        if images:
            classes[folder.name] = images

    actual, mapped = set(classes), set(DISEASE_TO_STRESS)
    if actual != mapped:
        missing = sorted(actual - mapped)
        stale = sorted(mapped - actual)
        raise ValueError(
            "DISEASE_TO_STRESS must exactly match the dataset. "
            f"Unmapped classes={missing}; mapping entries not found={stale}"
        )
    return classes


def plant_type(class_name: str) -> str:
    return class_name.split("___", 1)[0]


def build_crop_catalog(classes: dict[str, list[Path]]) -> dict[str, dict[str, list[str]]]:
    catalog: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for class_name in classes:
        catalog[plant_type(class_name)][DISEASE_TO_STRESS[class_name]].append(class_name)
    return {crop: dict(levels) for crop, levels in catalog.items()}


def progression(day_index: int, days: int, available: list[str], rng: random.Random) -> str:
    """Choose a non-decreasing stress rank, with a small plant-specific offset."""
    ranks = sorted(SEVERITY_RANK[level] for level in available)
    progress = day_index / max(days - 1, 1)
    target = min(3, int((progress + rng.uniform(-0.025, 0.025)) * 4))
    valid = [rank for rank in ranks if rank <= target]
    chosen = max(valid) if valid else min(ranks)
    return STRESS_LEVELS[chosen]


def sensor_reading(
    level: str,
    rng: random.Random,
    previous: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    means, deviations = SENSOR_PROFILES[level]
    values = [rng.gauss(mean, sd) for mean, sd in zip(means, deviations)]
    if previous is not None:
        # Environmental telemetry changes gradually. Smoothing also prevents
        # every row from acting like an independent class-labelled lookup.
        values = [0.65 * old + 0.35 * new for old, new in zip(previous, values)]
    soil = min(100.0, max(5.0, values[0]))
    temp = min(45.0, max(8.0, values[1]))
    humidity = min(100.0, max(10.0, values[2]))
    light = min(30000.0, max(1000.0, values[3]))
    return tuple(round(value, 2) for value in (soil, temp, humidity, light))


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_schema(path: Path) -> None:
    columns = {
        "Plant_ID": {"type": "string", "description": "Simulated virtual plant identifier"},
        "Image_Path": {"type": "string", "description": "Project-relative path to a real PlantVillage image"},
        "Timestamp": {"type": "date", "format": "YYYY-MM-DD", "description": "Simulated daily observation date"},
        "Plant_Type": {"type": "string", "description": "Crop name parsed from the source class"},
        "Soil_Moisture": {"type": "number", "unit": "percent", "range": [5, 100]},
        "Temperature": {"type": "number", "unit": "celsius", "range": [8, 45]},
        "Humidity": {"type": "number", "unit": "percent", "range": [10, 100]},
        "Light_Intensity": {"type": "number", "unit": "lux", "range": [1000, 30000]},
        "Health_Status": {"type": "category", "allowed_values": list(HEALTH_STATUS.values())},
        "Stress_Level": {"type": "category", "allowed_values": list(STRESS_LEVELS)},
        "Disease_Class": {"type": "category", "allowed_values": sorted(DISEASE_TO_STRESS)},
    }
    payload = {
        "dataset": "AgriSense AI hybrid sequential dataset",
        "schema_version": "1.0",
        "row_granularity": "one simulated plant-day observation",
        "columns": columns,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_readme(path: Path, stats: dict[str, object]) -> None:
    content = f"""# AgriSense AI sequential dataset

This dataset combines real PlantVillage leaf images with a reproducible simulated
timeline and plausible plant-health telemetry for image-first hybrid experimentation.

## Provenance and honesty

**Real:** image files, crop/disease folder labels, and the number of source images.

**Simulated:** virtual plant identity, calendar date, placement of independent
PlantVillage images into a plant progression, stress grouping, health status, and
all four sensor values. PlantVillage does not track individual plants over time and
does not include sensor readings. Therefore these sequences must not be described
as real longitudinal field observations or used for clinical/agronomic decisions.

The explicit disease-to-stress mapping and sensor distributions are in
`build_sequences.py`. A fixed random seed ({SEED}) makes generation reproducible.
Stress placement is non-decreasing by severity rank; sensor profile means move with
stress (not pure noise), with seeded Gaussian variation and safe physical bounds.

## Generated snapshot

- Virtual plants: {stats['plant_count']}
- Plant-day rows: {stats['row_count']}
- Sequence length: {stats['min_days']}–{stats['max_days']} days
- Crops used: {', '.join(stats['crops'])}
- Stress distribution: {json.dumps(stats['stress_distribution'], sort_keys=True)}
- Source images available: {stats['source_image_count']}
- Reused source images: {stats['reused_source_images']} (must remain zero)

`Image_Path` references the original files beneath `dataset/plantvillage_raw/`;
images are not duplicated. Paths are relative to the project root for portability.

## Rebuild

From the project root:

```bash
python3 dataset/build_sequences.py
```

Use `--plants`, `--min-days`, `--max-days`, or `--seed` to change generation.
The script validates that the explicit mapping exactly matches discovered classes.
"""
    path.write_text(content, encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.plants < 60:
        raise ValueError("--plants must be at least 60")
    if not (20 <= args.min_days <= args.max_days <= 30):
        raise ValueError("Require 20 <= --min-days <= --max-days <= 30")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    raw_root = (project_root / args.raw_root).resolve()
    class_root = locate_class_root(raw_root)
    classes = inspect_dataset(class_root)
    catalog = build_crop_catalog(classes)
    # A temporal stress demonstration needs healthy and at least one disease class.
    eligible_crops = sorted(
        crop for crop, levels in catalog.items() if "Healthy" in levels and len(levels) >= 2
    )
    if not eligible_crops:
        raise ValueError("No crop has both healthy and stressed images")

    rng = random.Random(args.seed)
    image_rng = random.Random(args.seed + 1)
    image_pools = {class_name: list(images) for class_name, images in classes.items()}
    for images in image_pools.values():
        image_rng.shuffle(images)
    fieldnames = [
        "Plant_ID", "Image_Path", "Timestamp", "Plant_Type", "Soil_Moisture",
        "Temperature", "Humidity", "Light_Intensity", "Health_Status",
        "Stress_Level", "Disease_Class",
    ]
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    base_date = date(2024, 1, 1)

    for index in range(args.plants):
        crop = eligible_crops[index % len(eligible_crops)]
        levels = catalog[crop]
        days = rng.randint(args.min_days, args.max_days)
        lengths.append(days)
        last_rank = 0
        previous_sensor_reading = None
        for day_index in range(days):
            proposed = progression(day_index, days, list(levels), rng)
            rank = max(last_rank, SEVERITY_RANK[proposed])
            available_ranks = sorted(SEVERITY_RANK[level] for level in levels)
            rank = max(r for r in available_ranks if r <= rank) if any(r <= rank for r in available_ranks) else min(available_ranks)
            last_rank = rank
            level = STRESS_LEVELS[rank]
            disease_class = rng.choice(levels[level])
            if not image_pools[disease_class]:
                raise ValueError(
                    f"Not enough unique images in {disease_class} for {args.plants} plants. "
                    "Reduce --plants or provide more source images."
                )
            image_path = image_pools[disease_class].pop()
            current_reading = sensor_reading(level, rng, previous_sensor_reading)
            previous_sensor_reading = current_reading
            soil, temp, humidity, light = current_reading
            rows.append({
                "Plant_ID": f"PLANT_{index + 1:03d}",
                "Image_Path": relative_path(image_path, project_root),
                "Timestamp": (base_date + timedelta(days=index * 3 + day_index)).isoformat(),
                "Plant_Type": crop,
                "Soil_Moisture": soil,
                "Temperature": temp,
                "Humidity": humidity,
                "Light_Intensity": light,
                "Health_Status": HEALTH_STATUS[level],
                "Stress_Level": level,
                "Disease_Class": disease_class,
            })

    output = script_dir / "sequential_data.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    stress_counts = Counter(str(row["Stress_Level"]) for row in rows)
    reused_source_images = len(rows) - len({str(row["Image_Path"]) for row in rows})
    stats = {
        "class_root": str(class_root),
        "class_count": len(classes),
        "source_image_count": sum(len(images) for images in classes.values()),
        "plant_count": args.plants,
        "row_count": len(rows),
        "min_days": min(lengths),
        "max_days": max(lengths),
        "crops": eligible_crops,
        "stress_distribution": dict(sorted(stress_counts.items())),
        "reused_source_images": reused_source_images,
        "output": str(output),
    }
    write_schema(script_dir / "schema.json")
    write_readme(script_dir / "README_dataset.md", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="dataset/plantvillage_raw")
    parser.add_argument("--plants", type=int, default=64)
    parser.add_argument("--min-days", type=int, default=20)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
