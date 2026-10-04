import csv
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def provenance(**extra):
    return {"timestamp": datetime.now(UTC).isoformat(), "python": platform.python_version(), "platform": platform.platform(), **extra}


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0, 1]
    z, proportion = 1.96, successes / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0, center - radius), min(1, center + radius)]


def write_table(path, rows: list[dict], title: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to report")
    columns = list(rows[0])
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(value):
        if value is None:
            return "pending"
        return f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "/")

    markdown = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    markdown += ["| " + " | ".join(cell(row[column]) for column in columns) + " |" for row in rows]
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
