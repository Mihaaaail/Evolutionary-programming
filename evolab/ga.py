"""Генетический алгоритм минимизации с точным учётом вычислений функции."""

import random
import time
from dataclasses import dataclass


@dataclass
class Result:
    genome: list
    fitness: float
    evaluations: int
    history: list
    seconds: float


def tournament(population, scores, rng, size=3):
    indices = [rng.randrange(len(population)) for _ in range(size)]
    return population[min(indices, key=lambda i: scores[i])][:]


def minimize(initialize, objective, vary, *, population_size, budget, seed, elitism=2):
    if population_size < 2 or budget < population_size or not 1 <= elitism < population_size:
        raise ValueError("Некорректные размер популяции, бюджет или элитизм")
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [initialize(rng) for _ in range(population_size)]
    scores = [objective(x) for x in population]
    evaluations = population_size
    index = min(range(population_size), key=lambda i: scores[i])
    best, best_score = population[index][:], scores[index]
    generation = 0
    history = [{"generation": 0, "evaluations": evaluations, "best": best_score}]
    while evaluations < budget:
        ordered = sorted(range(len(population)), key=lambda i: scores[i])
        next_population = [population[i][:] for i in ordered[:elitism]]
        next_scores = [scores[i] for i in ordered[:elitism]]
        while len(next_population) < population_size and evaluations < budget:
            a = tournament(population, scores, rng)
            b = tournament(population, scores, rng)
            child = vary(a, b, rng, evaluations / budget)
            value = objective(child)
            evaluations += 1
            next_population.append(child)
            next_scores.append(value)
            if value < best_score:
                best, best_score = child[:], value
        population, scores = next_population, next_scores
        generation += 1
        history.append({"generation": generation, "evaluations": evaluations, "best": best_score})
    return Result(best, best_score, evaluations, history, time.perf_counter() - started)


def random_search(initialize, objective, *, budget, seed, checkpoint=58):
    rng = random.Random(seed)
    started = time.perf_counter()
    best, best_score = None, float("inf")
    history = []
    for i in range(1, budget + 1):
        genome = initialize(rng)
        score = objective(genome)
        if score < best_score:
            best, best_score = genome[:], score
        if i % checkpoint == 0 or i == budget:
            history.append({"generation": len(history), "evaluations": i, "best": best_score})
    return Result(best, best_score, budget, history, time.perf_counter() - started)
