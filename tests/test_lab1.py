"""Проверки бюджета, воспроизводимости и области допустимых вещественных решений."""

import math
import random
import unittest

from evolab.common import ROOT,read_json
from evolab.ga import minimize
from lab1.problem import component,michalewicz,numerical_reference,reflect
from lab1.run import operators


class ContinuousTests(unittest.TestCase):
    def test_exact_budget_and_reproducibility(self):
        calls = []
        def objective(x):
            calls.append(x[:])
            return sum(v*v for v in x)
        def initial(rng):
            return [rng.uniform(-5,5) for _ in range(3)]
        def variation(a,b,rng,progress):
            return [(x+y)/2+rng.gauss(0,.1) for x,y in zip(a,b)]
        first = minimize(initial,objective,variation,population_size=30,budget=107,seed=11)
        self.assertEqual(len(calls),107)
        self.assertEqual(first.evaluations,107)
        second = minimize(initial,objective,variation,population_size=30,budget=107,seed=11)
        self.assertEqual(first.genome,second.genome)
        self.assertEqual(first.history,second.history)
        self.assertTrue(all(a["best"] >= b["best"] for a,b in zip(first.history,first.history[1:])))

    def test_function_known_values(self):
        self.assertEqual(michalewicz([0]*9),0)
        self.assertAlmostEqual(component(math.pi/2,2),-1,places=12)
        with self.assertRaises(ValueError):
            michalewicz([-1]*9)

    def test_large_reflections(self):
        for x in [-1000*math.pi-.2,-.3,0,math.pi,math.pi+.3,10000]:
            self.assertGreaterEqual(reflect(x),0)
            self.assertLessEqual(reflect(x),math.pi)
        self.assertAlmostEqual(reflect(-.3),.3)
        self.assertAlmostEqual(reflect(math.pi+.3),math.pi-.3)

    def test_all_offspring_feasible(self):
        config = read_json(ROOT / "lab1/config.json")
        rng = random.Random(9)
        for adaptive in (False,True):
            initial,vary = operators(config,adaptive)
            for _ in range(100):
                child = vary(initial(rng),initial(rng),rng,rng.random())
                self.assertEqual(len(child),9)
                self.assertTrue(all(0 <= x <= math.pi for x in child))

    def test_independent_reference(self):
        reference = numerical_reference(2,3000)
        self.assertAlmostEqual(reference["value"],-1.8013034101,places=7)


if __name__ == "__main__":
    unittest.main()
