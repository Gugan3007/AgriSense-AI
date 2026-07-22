from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.schema_service import ensure_schema_compatibility  # noqa: E402


class SchemaServiceTests(unittest.TestCase):
    def test_additive_migration_preserves_old_rows(self) -> None:
        engine = create_engine("sqlite://")
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE uploaded_images (id VARCHAR(36) PRIMARY KEY)"))
            connection.execute(text("CREATE TABLE predictions (id VARCHAR(36) PRIMARY KEY)"))
            connection.execute(text("INSERT INTO predictions (id) VALUES ('old')"))

        ensure_schema_compatibility(engine)

        prediction_columns = {item["name"] for item in inspect(engine).get_columns("predictions")}
        upload_columns = {item["name"] for item in inspect(engine).get_columns("uploaded_images")}
        self.assertIn("analysis_status", prediction_columns)
        self.assertIn("validation_result", upload_columns)
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar(), 1)


if __name__ == "__main__":
    unittest.main()
