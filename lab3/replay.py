"""Точное повторение траектории поиска по журналу измеренных критериев."""

import argparse
import json
import math
from pathlib import Path

from evolab.common import ROOT,read_json
from evolab.pareto import nondominated_sort,nsga2
from lab3.run import operators,weighted_ga


def replay(method,seed,results):
    config = read_json(results / "environment.json")["config"]
    records = []
    with (results / "evaluations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["method"] == method and item["seed"] == seed:
                records.append(item)
    if len(records) != config["budget"]:
        raise ValueError("В журнале нет полной серии для выбранного метода и состояния генератора")
    position = 0
    def evaluate(genome):
        nonlocal position
        record = records[position]
        if len(genome) != len(record["genome"]) or not all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
                                                                       for a,b in zip(genome,record["genome"])):
            raise AssertionError(f"Траектория разошлась на оценке {position+1}")
        position += 1
        return tuple(record["objectives"])
    initialize,vary = operators(config)
    if method == "nsga2":
        result = nsga2(initialize,evaluate,vary,population_size=config["population"],budget=config["budget"],seed=seed)
    else:
        index = seed-config["first_seed"]
        weights = config["weight_vectors"][index % len(config["weight_vectors"])]
        result = weighted_ga(initialize,evaluate,vary,config,seed,weights)
    expected = next(f for f in read_json(results / "final_fronts.json") if f["method"] == method and f["seed"] == seed)
    actual = {(tuple(result["population"][i]),tuple(result["objectives"][i])) for i in nondominated_sort(result["objectives"])[0]}
    saved = {(tuple(r["genome"]),tuple(r["objectives"])) for r in expected["front"]}
    def matches(a,b):
        return a[1] == b[1] and all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(a[0],b[0]))
    # Разные системные математические библиотеки могут давать различие последнего разряда.
    if not all(any(matches(a,b) for b in saved) for a in actual) or not all(any(matches(a,b) for a in actual) for b in saved):
        raise AssertionError("Финальный фронт отличается от записанного")
    return position


def main():
    parser = argparse.ArgumentParser(description="Повторение поиска по журналу измерений")
    parser.add_argument("--method",choices=["nsga2","weighted"],default="nsga2")
    parser.add_argument("--seed",type=int,default=3000)
    parser.add_argument("--results",type=Path,default=ROOT / "lab3/results")
    args = parser.parse_args()
    count = replay(args.method,args.seed,args.results)
    print(f"Траектория и финальный фронт совпали: {count} оценок")


if __name__ == "__main__":
    main()
