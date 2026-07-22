"""Small additive SQLite migration for analysis metadata."""

from __future__ import annotations

from sqlalchemy import inspect, text


ADDITIVE_COLUMNS = {
    "uploaded_images": {"validation_result": "JSON"},
    "predictions": {
        "analysis_status": "VARCHAR(32)",
        "reliability": "JSON",
        "image_validation": "JSON",
        "crop_consistency": "JSON",
        "observations": "JSON",
        "recommendations": "JSON",
    },
}


def ensure_schema_compatibility(engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table, definitions in ADDITIVE_COLUMNS.items():
            if table not in tables:
                continue
            existing = {item["name"] for item in inspect(engine).get_columns(table)}
            for column, sql_type in definitions.items():
                if column not in existing:
                    connection.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                    ))
