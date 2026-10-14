"""Проверка целостности приложенных опытов и готовых моделей."""

import csv
import unittest

import numpy as np

from evolab.common import ROOT,read_json
from evolab.pareto import dominates
from lab2.problem import ScheduleProblem
from lab3.model import accuracy,load_data
from lab3.replay import replay


class SavedResultsTests(unittest.TestCase):
    def test_series_and_equal_budgets(self):
        for lab,methods in [("lab1",3),("lab2",4),("lab3",2)]:
            config = read_json(ROOT / lab / "config.json")
            with (ROOT / lab / "results/runs.csv").open(encoding="utf-8",newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),methods*config["runs"])
            self.assertEqual(len({r["method"] for r in rows}),methods)
            self.assertTrue(all(int(r["evaluations"]) == config["budget"] for r in rows))
            for method in {r["method"] for r in rows}:
                seeds = {int(r["seed"]) for r in rows if r["method"] == method}
                self.assertEqual(seeds,set(range(config["first_seed"],config["first_seed"]+config["runs"])))

    def test_all_saved_schedules_are_feasible(self):
        problem = ScheduleProblem(read_json(ROOT / "lab2/data.json"))
        for item in read_json(ROOT / "lab2/results/best.json").values():
            self.assertEqual(problem.violations(item["schedule"]),[])
            self.assertEqual(problem.soft_cost(item["schedule"]),item["soft_cost"])
        examples = read_json(ROOT / "lab2/results/examples.json")
        self.assertEqual([bool(problem.violations(r["schedule"])) for r in examples],[False,False,True,True])

    def test_archive_is_nondominated(self):
        archive = read_json(ROOT / "lab3/results/pareto.json")
        self.assertGreaterEqual(len(archive),3)
        for i,a in enumerate(archive):
            for j,b in enumerate(archive):
                if i != j:
                    self.assertFalse(dominates(a["objectives"],b["objectives"]))

    def test_saved_models_match_test_accuracy(self):
        data,_,_ = load_data(ROOT / "lab3/data.csv",include_test=True)
        representatives = read_json(ROOT / "lab3/results/representatives.json")
        self.assertEqual(len({tuple(r["genome"]) for r in representatives}),3)
        for index,item in enumerate(representatives,1):
            saved = read_json(ROOT / f"lab3/results/model-{index}.json")
            model = {key:np.asarray(value) if key != "b2" else float(value) for key,value in saved["weights"].items()}
            self.assertAlmostEqual(accuracy(model,data["test"]),item["test_accuracy"],places=12)
            self.assertAlmostEqual(accuracy(model,data["validation"]),1-item["objectives"][0],places=12)

    def test_replay_both_algorithms(self):
        for method in ("nsga2","weighted"):
            self.assertEqual(replay(method,3000,ROOT / "lab3/results"),320)


if __name__ == "__main__":
    unittest.main()
