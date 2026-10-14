"""Чтение параметров, проверка бюджета и сохранение результатов."""

import csv
import json
import math
import platform
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Нельзя сохранить пустую таблицу результатов")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    values = list(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Статистика требует непустого набора конечных значений")
    return {"count": len(values), "minimum": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "maximum": max(values)}


def validate_config(config):
    for key in ("population", "budget", "runs"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Параметр {key} должен быть положительным целым числом")
    if config["population"] < 30 or config["budget"] < config["population"]:
        raise ValueError("Нужны хотя бы 30 особей и бюджет не меньше популяции")
    if config["runs"] < 20:
        raise ValueError("Для итогового опыта нужно не меньше 20 запусков")
    for key in ("mutation_probability", "crossover_probability"):
        if key in config and not 0 <= config[key] <= 1:
            raise ValueError(f"Вероятность {key} должна лежать от 0 до 1")


def environment(config):
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "config": config, "variant": read_json(ROOT / "variant.json")}


def table(summary, names):
    lines = ["| Метод | Лучшее | Среднее | Медиана | Станд. откл. | Худшее |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, stats in summary.items():
        lines.append("| " + names.get(key, key) + " | " + " | ".join(f"{stats[k]:.6g}" for k in
                     ("minimum", "mean", "median", "std", "maximum")) + " |")
    return "\n".join(lines)
