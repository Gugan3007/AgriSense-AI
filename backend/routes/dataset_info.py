"""Dataset information endpoint computed from the real generated CSV."""

from __future__ import annotations

from flask import Blueprint, jsonify
import pandas as pd

from config import DATASET_CSV

dataset_info_bp = Blueprint("dataset_info", __name__)


@dataset_info_bp.get("/dataset-info")
def dataset_info():
    if not DATASET_CSV.is_file():
        return jsonify({"error": "dataset/sequential_data.csv not found."}), 404

    frame = pd.read_csv(DATASET_CSV)
    timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce")
    sample_rows = frame.head(8).copy()
    sample_rows["image_url"] = sample_rows["Image_Path"].apply(lambda path: f"/dataset-image/{path}")
    return jsonify({
        "row_count": int(len(frame)),
        "image_count": int(frame["Image_Path"].nunique()),
        "plant_count": int(frame["Plant_ID"].nunique()),
        "plant_types": sorted(frame["Plant_Type"].dropna().unique().tolist()),
        "stress_distribution": frame["Stress_Level"].value_counts().to_dict(),
        "disease_class_distribution": frame["Disease_Class"].value_counts().to_dict(),
        "date_range": {
            "start": timestamps.min().date().isoformat(),
            "end": timestamps.max().date().isoformat(),
        },
        "sample_rows": sample_rows.to_dict(orient="records"),
    })
