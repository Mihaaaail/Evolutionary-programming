"""Полные серии опытов лабораторной № 1 и формирование отчёта."""

import argparse
import math
from pathlib import Path

from evolab.common import ROOT, describe, environment, read_json, table, validate_config, write_csv, write_json
from evolab.ga import minimize, random_search
from evolab.plots import chart, convergence
from lab1.problem import michalewicz, numerical_reference, reflect

NAMES = {"fixed": "ГА, постоянный масштаб", "adaptive": "ГА, убывающий масштаб", "random": "Случайная выборка"}


def operators(config, adaptive=False):
    def initialize(rng):
        return [rng.uniform(0, math.pi) for _ in range(config["dimension"])]
    def vary(a, b, rng, progress):
        child = a[:]
        if rng.random() < config["crossover_probability"]:
            for i in range(len(child)):
                lo, hi = sorted((a[i], b[i]))
                margin = config["blx_alpha"] * (hi-lo)
                child[i] = rng.uniform(lo-margin, hi+margin)
        fraction = config["fixed_sigma_fraction"]
        if adaptive:
            start, end = config["adaptive_initial_sigma_fraction"], config["adaptive_final_sigma_fraction"]
            fraction = end + (start-end)*(1-progress)**2
        for i in range(len(child)):
            if rng.random() < config["mutation_probability"]:
                child[i] += rng.gauss(0, fraction*math.pi)
            child[i] = reflect(child[i])
        return child
    return initialize, vary


def run(config, output):
    validate_config(config)
    if config["dimension"] != 9:
        raise ValueError("Индивидуальный вариант 10 требует размерности 9")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", environment(config))
    reference = numerical_reference(9, config["reference_grid"])
    write_json(output / "reference.json", reference)
    results, histories, bests, summaries = [], [], {}, {}
    for method in NAMES:
        runs = []
        for seed in range(config["first_seed"], config["first_seed"]+config["runs"]):
            initialize, vary = operators(config, method == "adaptive")
            if method == "random":
                result = random_search(initialize, michalewicz, budget=config["budget"], seed=seed)
            else:
                result = minimize(initialize, michalewicz, vary, population_size=config["population"],
                                  budget=config["budget"], seed=seed, elitism=config["elitism"])
            results.append({"method": method, "seed": seed, "fitness": result.fitness,
                            "gap_to_reference": result.fitness-reference["value"],
                            "evaluations": result.evaluations, "seconds": result.seconds})
            histories.extend({"method": method, "seed": seed, **row} for row in result.history)
            runs.append(result)
        best = min(runs, key=lambda r: r.fitness)
        bests[method] = {"point": best.genome, "value": best.fitness,
                         "seed": results[-config["runs"]+runs.index(best)]["seed"]}
        summaries[method] = describe(r.fitness for r in runs)
        convergence(output / f"convergence-{method}.svg", [r.history for r in runs], NAMES[method])
        print(f"Лабораторная 1: {NAMES[method]}, завершено {len(runs)} запусков", flush=True)
    write_csv(output / "runs.csv", results)
    write_csv(output / "history.csv", histories)
    write_json(output / "best.json", bests)
    write_json(output / "summary.json", summaries)
    means = []
    for method in NAMES:
        rows = [r for r in histories if r["method"] == method]
        groups = {}
        for row in rows:
            groups.setdefault(row["evaluations"], []).append(row["best"])
        means.append((NAMES[method], [(n, sum(v)/len(v)) for n,v in sorted(groups.items())]))
    chart(output / "comparison.svg", means, "Сравнение при одинаковом бюджете", "Вычисления функции", "Среднее лучшее значение")
    fixed = [r["fitness"] for r in results if r["method"] == "fixed"]
    adaptive = [r["fitness"] for r in results if r["method"] == "adaptive"]
    differences = [a-b for a,b in zip(adaptive, fixed)]
    write_json(output / "paired_difference.json", describe(differences))
    better = min(("fixed", "adaptive"), key=lambda m: summaries[m]["mean"])
    conclusion = ("Убывающий масштаб улучшил среднее значение относительно постоянного. " if better == "adaptive" else
                  "В этой серии постоянный масштаб оказался лучше по среднему значению. ")
    report = rf"""# Лабораторная работа № 1

## Постановка и индивидуальный вариант

N = 28, группа 517, Q = 1, Y = 26, V = 10. Размерность d = 5 + (10 mod 6) = 9.
Минимизируется функция Михалевича:

$$f(x) = -\sum_{{i=1}}^9 \sin(x_i)\,[\sin(i x_i^2 / \pi)]^{{20}},\qquad 0\le x_i\le\pi.$$

Функция разделима по координатам, но содержит узкие долины и множество локальных
минимумов. Случайной выборке трудно одновременно попасть в хорошие долины девяти
координат. Локальный поиск зависит от начальной точки. Разделимость также означает,
что для этой конкретной функции существуют более подходящие методы, чем общий ГА.

Определение, область и параметр m = 10 сверены с
[каталогом Университета Саймона Фрейзера](https://www.sfu.ca/~ssurjano/michal.html).
В [научной публикации](https://pmc.ncbi.nlm.nih.gov/articles/PMC4772519/)
приведены ориентиры: d = 2 — −1,8013; d = 5 — −4,687658;
d = 10 — −9,66015. Эти значения относятся к другим размерностям,
поэтому непосредственно сравнивать с ними девятимерный результат нельзя.
Численный ориентир для d = 9 отдельно вычислен покоординатным просмотром
{reference['grid_per_coordinate']} интервалов и уточнением золотым сечением:
**f ≈ {reference['value']:.10f}**. Координаты в `results/reference.json`.
Это независимый численный ориентир, не строгая гарантия; общий нижний предел f ≥ −9
следует из границ каждого слагаемого. ГА не получает координаты ориентира.

## Представление и алгоритм

Генотип и фенотип совпадают: список из девяти вещественных чисел.
Инициализация равномерная в допустимом кубе. Турнир из трёх особей выбирает родителей.
BLX-α с α = {config['blx_alpha']} создаёт каждую координату из расширенного интервала
между родителями. Он учитывает вещественное представление и допускает исследование
соседней области. Гауссовская мутация выполняется независимо по координатам.
Выход за границы исправляется отражением, включая большие скачки.
Две элитные особи переносятся без повторной оценки. Остановка точно по бюджету,
включая при необходимости неполное последнее поколение.

```mermaid
flowchart TD
    A[Равномерная инициализация] --> B[Оценка популяции]
    B --> C{{Бюджет исчерпан?}}
    C -- Нет --> D[Турнирный отбор и сохранение элиты]
    D --> E[BLX и гауссовская мутация]
    E --> F[Отражение границ и оценка потомков]
    F --> C
    C -- Да --> G[Лучшее решение и журнал]
```

## Параметры и сравнение

Популяция {config['population']}; бюджет {config['budget']} вычислений на запуск;
вероятность скрещивания {config['crossover_probability']}; вероятность мутации одной
координаты 1/9; элитизм {config['elitism']}; начальные состояния генератора
{config['first_seed']}–{config['first_seed']+config['runs']-1}.
Полные параметры в `config.json`, фактически применённые — в `results/environment.json`.

Две конфигурации ГА различаются **только законом масштаба мутации**:
постоянный σ = {config['fixed_sigma_fraction']}π и убывающий
σ(t) = π·({config['adaptive_final_sigma_fraction']} +
({config['adaptive_initial_sigma_fraction']}-{config['adaptive_final_sigma_fraction']})(1−t)²),
где t — доля израсходованного бюджета.
Убывающий масштаб является дополнительным заданием; он задаётся заранее,
не зависит от качества популяции и не является самоадаптацией.
Случайная выборка имеет тот же бюджет. Оценки начальной популяции включены.
Оценки элиты повторно не считаются. Отдельное вычисление ориентира исключено.

## Результаты

Для каждого метода выполнено {config['runs']} независимых запусков.
Стандартное отклонение выборочное (делитель n−1). Меньшее значение лучше.

{table(summaries, NAMES)}

![Средние траектории](results/comparison.svg)

![Минимум среднее максимум постоянного масштаба](results/convergence-fixed.svg)

![Минимум среднее максимум убывающего масштаба](results/convergence-adaptive.svg)

![Минимум среднее максимум случайной выборки](results/convergence-random.svg)

Средняя парная разность «убывающий − постоянный» при совпадающих начальных
состояниях генератора: {sum(differences)/len(differences):.6g}.
После различий в операторах траектории случайных чисел могут расходиться.
CSV содержит также время и отклонение от ориентира. Координаты лучшего решения и
его начальное состояние генератора записаны в `results/best.json`.

## Выводы

{conclusion}Это вывод о заданном бюджете и выбранных параметрах, а не универсальная
гарантия. Лучший результат ГА имеет отклонение от численного ориентира
{min(summaries['fixed']['minimum'],summaries['adaptive']['minimum'])-reference['value']:.6g}.
Среднее значение лучшей конфигурации равно {summaries[better]['mean']:.6g},
случайной выборки — {summaries['random']['mean']:.6g}.
Результат следует оценивать и по отклонению от ориентира, и по разбросу серии.
Даже при выигрыше у случайной выборки отсутствие строгого сертификата не позволяет
объявить найденную точку доказанным глобальным минимумом.

"""
    (output.parent / "REPORT.md").write_text(report.rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Оптимизация функции Михалевича")
    parser.add_argument("--config", type=Path, default=ROOT / "lab1/config.json", help="Файл параметров")
    parser.add_argument("--output", type=Path, default=ROOT / "lab1/results", help="Папка результатов")
    args = parser.parse_args()
    run(read_json(args.config), args.output)


if __name__ == "__main__":
    main()
