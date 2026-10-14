"""Проверка каждого ограничения и восстановления после разрушающих операторов."""

import random
import unittest

from evolab.common import ROOT,read_json
from lab2.problem import ScheduleProblem
from lab2.run import operators


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / "lab2/config.json")
        self.problem = ScheduleProblem(read_json(ROOT / "lab2/data.json"))
        self.good = self.problem.data["feasible_witness"][:]

    def test_witness_and_instance_scale(self):
        self.assertEqual(self.problem.violations(self.good),[])
        self.assertGreaterEqual(len(self.problem.events),20)
        self.assertGreaterEqual(len(self.problem.conflicts),10)
        self.assertGreaterEqual(len(self.problem.employees),5)

    def test_each_constraint_detected(self):
        bad = self.good[:]; bad[2] = 4
        self.assertTrue(any("допуска" in v for v in self.problem.violations(bad)))
        bad = self.good[:]; bad[1] = bad[0]
        self.assertIn("Конфликт смен 0 и 1",self.problem.violations(bad))
        bad = self.good[:]; bad[3] = bad[2]
        self.assertIn("Конфликт смен 2 и 3",self.problem.violations(bad))
        event = next(e for e in self.problem.events if e["unavailable"])
        bad = self.good[:]; bad[event["id"]] = event["unavailable"][0]
        self.assertTrue(any("недоступен" in v for v in self.problem.violations(bad)))
        bad = [0]*21
        self.assertTrue(any("недельного" in v for v in self.problem.violations(bad)))
        bad = self.good[:]; bad[0] = 100
        self.assertTrue(any("неверный сотрудник" in v for v in self.problem.violations(bad)))

    def test_repair_varied_invalid_genomes(self):
        rng = random.Random(19)
        for _ in range(50):
            bad = [rng.randrange(8) for _ in range(21)]
            repaired = self.problem.repair(bad,rng)
            self.assertEqual(self.problem.violations(repaired),[])
        self.assertEqual(self.problem.repair(self.good,rng),self.good)

    def test_both_mutations_preserve_feasibility_after_repair(self):
        rng = random.Random(20)
        for swap in (False,True):
            initial,vary = operators(self.problem,self.config,True,swap)
            for _ in range(50):
                child = vary(initial(rng),initial(rng),rng,.5)
                self.assertEqual(self.problem.violations(child),[])

    def test_penalty_separates_feasibility(self):
        rng = random.Random(21)
        for _ in range(50):
            x = self.problem.initialize(rng)
            self.assertLessEqual(self.problem.soft_cost(x),300)
            if self.problem.violations(x):
                self.assertGreater(self.problem.soft_cost(x)+1000*len(self.problem.violations(x)),300)


if __name__ == "__main__":
    unittest.main()
