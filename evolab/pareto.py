"""Собственная недоминируемая сортировка и расстояние до скопления NSGA-II."""

import math
import random
import time


def dominates(a,b):
    return all(x <= y for x,y in zip(a,b)) and any(x < y for x,y in zip(a,b))


def nondominated_sort(objectives):
    dominated = [[] for _ in objectives]
    counts = [0]*len(objectives)
    for i in range(len(objectives)):
        for j in range(i+1,len(objectives)):
            if dominates(objectives[i],objectives[j]):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(objectives[j],objectives[i]):
                dominated[j].append(i)
                counts[i] += 1
    front = [i for i,n in enumerate(counts) if n == 0]
    fronts = []
    while front:
        fronts.append(front)
        following = []
        for i in front:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        front = following
    return fronts


def crowding_distance(front,objectives):
    distances = {i:0.0 for i in front}
    if len(front) <= 2:
        return {i:math.inf for i in front}
    for criterion in range(len(objectives[0])):
        ordered = sorted(front,key=lambda i:objectives[i][criterion])
        low,high = objectives[ordered[0]][criterion],objectives[ordered[-1]][criterion]
        # Постоянная цель не должна произвольно выделять крайние по номеру особи.
        if high == low:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = math.inf
        for position in range(1,len(ordered)-1):
            before,after = ordered[position-1],ordered[position+1]
            distances[ordered[position]] += (objectives[after][criterion]-objectives[before][criterion])/(high-low)
    return distances


def select_indices(objectives,size):
    chosen = []
    for front in nondominated_sort(objectives):
        if len(chosen)+len(front) <= size:
            chosen.extend(front)
        else:
            distances = crowding_distance(front,objectives)
            chosen.extend(sorted(front,key=lambda i:(-distances[i],i))[:size-len(chosen)])
            break
    return chosen


def ranks_and_distances(objectives):
    ranks,distances = {},{}
    for rank,front in enumerate(nondominated_sort(objectives)):
        ranks.update({i:rank for i in front})
        distances.update(crowding_distance(front,objectives))
    return ranks,distances


def nsga2(initialize,evaluate,vary,*,population_size,budget,seed):
    if population_size < 2 or budget < population_size:
        raise ValueError("Недостаточные размер популяции или бюджет")
    started = time.perf_counter()
    rng = random.Random(seed)
    population = [initialize(rng) for _ in range(population_size)]
    objectives = [evaluate(x) for x in population]
    evaluations = len(population)
    generation = 0
    history = []
    while True:
        front = nondominated_sort(objectives)[0]
        history.append({"generation":generation,"evaluations":evaluations,"front_size":len(front),
                        "best_error":min(o[0] for o in objectives)})
        if evaluations == budget:
            break
        ranks,distances = ranks_and_distances(objectives)
        def parent():
            i,j = rng.randrange(len(population)),rng.randrange(len(population))
            winner = min((i,j),key=lambda k:(ranks[k],-distances[k]))
            return population[winner][:]
        children = []
        values = []
        for _ in range(min(population_size,budget-evaluations)):
            child = vary(parent(),parent(),rng,evaluations/budget)
            children.append(child)
            values.append(evaluate(child))
            evaluations += 1
        combined = population+children
        combined_values = objectives+values
        selected = select_indices(combined_values,population_size)
        population = [combined[i] for i in selected]
        objectives = [combined_values[i] for i in selected]
        generation += 1
    return {"population":population,"objectives":objectives,"history":history,"evaluations":evaluations,
            "seconds":time.perf_counter()-started}


def additive_epsilon(approximation,reference,scales):
    """Аддитивный ε-индикатор: покрытие общего эмпирического фронта, меньше лучше."""
    if not approximation or not reference:
        raise ValueError("Для индикатора нужны оба непустых множества")
    return max(min(max((a[k]-r[k])/scales[k] for k in range(len(scales)))
                   for a in approximation) for r in reference)
