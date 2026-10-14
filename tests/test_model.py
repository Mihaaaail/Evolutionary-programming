"""Проверки воспроизводимости обучения и отсутствия утечки нормировки."""

import random
import unittest

import numpy as np

from evolab.common import ROOT,read_json
from lab3.model import accuracy,load_data,parameter_count,train_model
from lab3.run import operators


class ClassifierTests(unittest.TestCase):
    def test_reproducible_training_and_parameter_count(self):
        data,_,_ = load_data(ROOT / "lab3/data.csv")
        parameters = {"hidden":8,"epochs":30,"learning_rate":.2,"l2":.001}
        first = train_model(*data["train"],parameters)
        second = train_model(*data["train"],parameters)
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])
        total = sum(np.asarray(value).size for value in first.values())
        self.assertEqual(total,parameter_count(8))
        self.assertTrue(0 <= accuracy(first,data["validation"]) <= 1)

    def test_test_part_absent_during_search(self):
        data,mean,std = load_data(ROOT / "lab3/data.csv")
        self.assertNotIn("test",data)
        np.testing.assert_allclose(data["train"][0].mean(axis=0),np.zeros(4),atol=1e-12)
        full,other_mean,other_std = load_data(ROOT / "lab3/data.csv",include_test=True)
        self.assertEqual(len(full["test"][0]),80)
        np.testing.assert_array_equal(mean,other_mean)
        np.testing.assert_array_equal(std,other_std)

    def test_structure_operators_respect_domains(self):
        config = read_json(ROOT / "lab3/config.json")
        initialize,vary = operators(config)
        rng = random.Random(5)
        for _ in range(100):
            x = vary(initialize(rng),initialize(rng),rng,.5)
            self.assertTrue(config["hidden_min"] <= x[0] <= config["hidden_max"])
            self.assertTrue(config["epochs_min"] <= x[1] <= config["epochs_max"])
            self.assertTrue(config["log10_learning_rate_min"] <= x[2] <= config["log10_learning_rate_max"])
            self.assertIn(x[3],(0,1))


if __name__ == "__main__":
    unittest.main()
