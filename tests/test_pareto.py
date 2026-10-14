"""Ручные примеры доминирования, разнообразия и покрытия фронта."""

import math
import unittest

from evolab.pareto import additive_epsilon,crowding_distance,dominates,nondominated_sort,nsga2,select_indices


class ParetoTests(unittest.TestCase):
    def test_manual_fronts_and_duplicates(self):
        points = [(0,3),(1,2),(2,1),(3,0),(2,3),(3,3),(0,3)]
        fronts = nondominated_sort(points)
        self.assertEqual(set(fronts[0]),{0,1,2,3,6})
        self.assertEqual(set(fronts[1]),{4})
        self.assertEqual(set(fronts[2]),{5})
        self.assertFalse(dominates((1,1),(1,1)))
        self.assertTrue(dominates((1,1),(1,2)))

    def test_crowding_extremes_and_constant_objectives(self):
        points = [(0,3),(1,2),(2,1),(3,0)]
        distance = crowding_distance([0,1,2,3],points)
        self.assertTrue(math.isinf(distance[0]))
        self.assertTrue(math.isinf(distance[3]))
        self.assertEqual(set(select_indices(points,2)),{0,3})
        self.assertEqual(crowding_distance([0,1,2],[(1,1)]*3),{0:0,1:0,2:0})

    def test_epsilon_known_coverage(self):
        reference = [(0,2),(1,1),(2,0)]
        self.assertEqual(additive_epsilon(reference,reference,[1,1]),0)
        self.assertEqual(additive_epsilon([(1,1)],reference,[1,1]),1)

    def test_nsga_exact_budget_and_reproducibility(self):
        calls = []
        def initial(rng):
            return [rng.random()]
        def evaluate(x):
            calls.append(x)
            return (x[0],1-x[0])
        def vary(a,b,rng,progress):
            return [rng.random()]
        first = nsga2(initial,evaluate,vary,population_size=30,budget=103,seed=4)
        self.assertEqual(len(calls),103)
        second = nsga2(initial,evaluate,vary,population_size=30,budget=103,seed=4)
        self.assertEqual(first["population"],second["population"])
        self.assertEqual(first["objectives"],second["objectives"])


if __name__ == "__main__":
    unittest.main()
