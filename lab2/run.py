"""Сравнение штрафов, ремонта, мутаций и случайного допустимого поиска."""

import argparse
from pathlib import Path

from evolab.common import ROOT, describe, environment, read_json, table, validate_config, write_csv, write_json
from evolab.ga import minimize, random_search
from evolab.plots import chart, convergence
from lab2.problem import DAYS, SHIFTS, ScheduleProblem, generate_data

NAMES = {"repair_assignment": "Ремонт и переназначение", "penalty_assignment": "Штраф и переназначение",
         "repair_swap": "Ремонт и обмен", "random_feasible": "Случайный допустимый"}


def operators(problem, config, repair=True, swap=False):
    def initialize(rng):
        genome = problem.initialize(rng)
        return problem.repair(genome, rng) if repair else genome
    def vary(a,b,rng,progress):
        child = a[:]
        if rng.random() < config["crossover_probability"]:
            child = [x if rng.random() < .5 else y for x,y in zip(a,b)]
        for i in range(len(child)):
            if rng.random() < config["mutation_probability"]:
                if swap:
                    j = rng.randrange(len(child))
                    child[i], child[j] = child[j], child[i]
                else:
                    child[i] = rng.choice(problem.domains[i])
        return problem.repair(child,rng) if repair else child
    return initialize,vary


def run(config, data_path, output):
    validate_config(config)
    if config["penalty"] <= 300:
        raise ValueError("Коэффициент штрафа должен превосходить верхнюю границу мягкой стоимости 300")
    if not data_path.exists():
        generate_data(data_path, config["data_seed"])
    data = read_json(data_path)
    if data["seed"] != config["data_seed"]:
        raise ValueError("Начальное состояние генерации данных не совпадает с конфигурацией")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", environment(config))
    rows, histories, summaries, bests, success, repairs = [], [], {}, {}, {}, {}
    for method in NAMES:
        problem = ScheduleProblem(data,config["imbalance_weight"],config["repair_node_limit"])
        initialize,vary = operators(problem,config,method != "penalty_assignment",method == "repair_swap")
        def objective(genome):
            return problem.soft_cost(genome) + config["penalty"]*len(problem.violations(genome))
        runs = []
        for seed in range(config["first_seed"],config["first_seed"]+config["runs"]):
            if method == "random_feasible":
                result = random_search(initialize,objective,budget=config["budget"],seed=seed)
            else:
                result = minimize(initialize,objective,vary,population_size=config["population"],
                                  budget=config["budget"],seed=seed,elitism=config["elitism"])
            hard = len(problem.violations(result.genome))
            rows.append({"method":method,"seed":seed,"fitness":result.fitness,"soft_cost":problem.soft_cost(result.genome),
                         "violations":hard,"feasible":int(hard == 0),"evaluations":result.evaluations,"seconds":result.seconds})
            histories.extend({"method":method,"seed":seed,**h} for h in result.history)
            runs.append(result)
        feasible_runs = [r for r in runs if not problem.violations(r.genome)]
        success[method] = len(feasible_runs)/len(runs)
        repairs[method] = problem.fallbacks
        if feasible_runs:
            best = min(feasible_runs,key=lambda r:r.fitness)
            bests[method] = {"schedule":best.genome,"soft_cost":problem.soft_cost(best.genome),
                             "seed":config["first_seed"]+runs.index(best)}
            summaries[method] = describe(problem.soft_cost(r.genome) for r in feasible_runs)
        convergence(output / f"convergence-{method}.svg",[r.history for r in runs],NAMES[method])
        print(f"Лабораторная 2: {NAMES[method]}, допустимых запусков {len(feasible_runs)}/{len(runs)}",flush=True)
    if not bests:
        raise RuntimeError("Серия не дала ни одного допустимого решения")
    problem = ScheduleProblem(data,config["imbalance_weight"],config["repair_node_limit"])
    greedy = problem.greedy()
    overall_method = min(bests,key=lambda m:bests[m]["soft_cost"])
    overall = bests[overall_method]
    write_csv(output / "runs.csv",rows)
    write_csv(output / "history.csv",histories)
    write_json(output / "summary.json",{"feasible_soft_cost":summaries,"feasible_fraction":success,"repair_fallbacks":repairs,
                                        "greedy_cost":problem.soft_cost(greedy)})
    write_json(output / "best.json",bests)
    write_csv(output / "best_schedule.csv",problem.rows(overall["schedule"]))
    write_json(output / "greedy.json",{"schedule":greedy,"cost":problem.soft_cost(greedy)})
    good1 = data["feasible_witness"]
    good2 = overall["schedule"]
    bad1 = good1[:]; bad1[2] = 4
    bad2 = good1[:]; bad2[1] = bad2[0]
    examples = [{"label":label,"schedule":x,"violations":problem.violations(x)} for label,x in
                [("Допустимое 1",good1),("Допустимое 2",good2),("Недопустимое 1",bad1),("Недопустимое 2",bad2)]]
    write_json(output / "examples.json",examples)
    comparisons = []
    for method in NAMES:
        groups = {}
        for row in histories:
            if row["method"] == method:
                groups.setdefault(row["evaluations"],[]).append(row["best"])
        comparisons.append((NAMES[method],[(n,sum(v)/len(v)) for n,v in sorted(groups.items())]))
    chart(output / "comparison.svg",comparisons,"Динамика стоимости с учётом штрафов","Вычисления функции","Среднее лучшее значение")
    schedule_table = "| День | Утро | День | Ночь |\n|---|---|---|---|\n"
    for day,name in enumerate(DAYS):
        schedule_table += "| " + name + " | " + " | ".join(problem.employees[overall["schedule"][3*day+s]]["name"] for s in range(3)) + " |\n"
    rates = "\n".join(f"- {NAMES[m]}: {success[m]:.0%}; аварийных возвратов к свидетелю {repairs[m]}." for m in NAMES)
    method_a,method_b = "repair_assignment","repair_swap"
    mutation_conclusion = ("Переназначение дало меньшее среднее значение." if summaries[method_a]["mean"] < summaries[method_b]["mean"]
                           else "Обмен дал меньшее или равное среднее значение.")
    report = rf"""# Лабораторная работа № 2

## Постановка и данные

Индивидуальный вариант 10: план дежурств на неделю, N = 28, группа 517.
Назначаются восемь сотрудников на 21 смену (по три в каждый из семи дней).
В каждой смене ровно один сотрудник; сотрудники 1–4 имеют право на ночные смены.
Все смены одинаковой длительности, отдых после ночной смены запрещает только
утреннюю смену следующего дня. Неделя рассматривается отдельно: переход к следующему
понедельнику не моделируется. Это явные допущения учебного экземпляра.

Входной файл `data.json` с начальным состоянием генерации {config['data_seed']} содержит
доступность, цены нежелательности назначений и 27 конфликтующих пар.
Данные синтетические: сначала задаётся допустимый свидетель; затем с вероятностью
0,17 запрещаются прочие назначения, стоимости независимо выбираются от 1 до 10.
Свидетель не является оптимумом и не добавляется в исходную популяцию.
Он используется для проверки входных данных и аварийного восстановления после
{config['repair_node_limit']} узлов поиска. Число таких случаев фиксируется отдельно.
Ресурсов восемь, событий 21, конфликтов 27: минимальный масштаб задания соблюдён.

## Формальная модель

Генотип и фенотип — вектор a длины 21, aᵢ ∈ {{0,…,7}}. Позиция задаёт смену,
значение — сотрудника. Каждая позиция всегда содержит ровно одно назначение.
Жёсткие условия: доступность, допуск к ночной смене, не более одной смены в день,
отдых после ночи, не более четырёх смен на сотрудника в неделю.

Мягкая стоимость:

$$C(a)=\sum_i c_{{i,a_i}} + {config['imbalance_weight']}\sum_{{e=0}}^7 |n_e-3|.$$

Здесь c — нежелательность назначения, n — недельная загрузка. Желаемые три смены
для каждого недостижимы одновременно (всего 21, а не 24), поэтому дисбаланс не может
обнулиться; минимальный вклад дисбаланса равен шести. Минимизируется C среди
допустимых решений. Для штрафного метода F = C + {config['penalty']}·H,
где H — число нарушений, превышение недельного лимита считается по каждой лишней смене.
Для корректных векторов C ≤ 300, поэтому любое допустимое решение лучше недопустимого
при указанном штрафе. Ошибки формата отвергаются до вычисления мягкой стоимости.

## Алгоритм и операторы

Турнир из трёх особей, равномерное скрещивание векторов назначений, элитизм двух
особей. Сравниваются переназначение сотрудника из разрешённого для смены множества
и обмен значений двух позиций. Обмен сохраняет недельные количества, но может
нарушить доступность или отдых. Скрещивание тоже может создавать конфликты.

Ремонт решает только задачу допустимости: сначала назначается наиболее ограниченная
смена; варианты упорядочены по сохранению гена, загрузке и стоимости; при тупике
выполняется возврат. Оптимальное расписание этим поиском не вычисляется.
Штрафный метод сохраняет нарушения и оценивает их явно.

```mermaid
flowchart TD
    A[Случайные назначения] --> B[Ремонт либо сохранение нарушений]
    B --> C[Стоимость и жёсткие нарушения]
    C --> D{{Остался бюджет?}}
    D -- Да --> E[Турнир и равномерное скрещивание]
    E --> F[Переназначение либо обмен]
    F --> B
    D -- Нет --> G[Проверка лучшего расписания]
```

## Эксперимент

По {config['runs']} запусков каждого метода; начальные состояния
{config['first_seed']}–{config['first_seed']+config['runs']-1}; популяция {config['population']};
бюджет {config['budget']} оценок; вероятность скрещивания {config['crossover_probability']};
вероятность изменения каждой позиции {config['mutation_probability']}.
Переназначение может выбрать того же сотрудника; обмен может выбрать ту же позицию.

Штраф и ремонт сравниваются при одинаковом операторе переназначения.
Две мутации сравниваются при одинаковом ремонте. Случайный допустимый поиск
использует такое же восстановление и тот же бюджет оценок стоимости.
Детерминированная жадная эвристика дополнительно даёт C = {problem.soft_cost(greedy)}
за одно построение; её бюджет меньше, это дополнительный ориентир.
Бюджет равен по оценкам, но ремонт требует дополнительного времени; время каждого
запуска приведено в CSV и не скрывается.

## Результаты

Статистика мягкой стоимости только по допустимым финальным решениям:

{table(summaries,NAMES)}

Доля допустимых финальных решений и возвраты к свидетелю:

{rates}

![Сравнение методов](results/comparison.svg)

Графики отдельных серий с минимумом, средним и максимумом находятся в `results/`.
Штрафные значения не следует смешивать с мягкими стоимостями допустимых решений.
Полные нарушения и стоимости сохранены в `runs.csv`.

## Лучшее расписание

Метод «{NAMES[overall_method]}», начальное состояние генератора {overall['seed']},
мягкая стоимость {overall['soft_cost']}; жёстких нарушений **0**.

{schedule_table}

Полная предметная таблица находится в `results/best_schedule.csv`.
Два допустимых и два недопустимых вектора приведены в `results/examples.json`.
Первый недопустимый пример нарушает допуск к ночи, второй — одну смену в день.
Причины определяются проверяющей функцией; примеры не подменяют проверку остальных условий.

## Выводы

{mutation_conclusion}Ремонт гарантирует допустимость после вариации, но может уменьшать
разнообразие, поскольку несколько исходных векторов переходят в одно расписание.
Штрафный поиск допускает промежуточные конфликты; его устойчивость оценивается по
доле допустимых финальных решений и разбросу стоимости.
Результат приемлем как допустимый недельный план с явно измеренной нежелательностью.
Его качество сравнивается с жадным построением ({problem.soft_cost(greedy)}) и случайным
допустимым поиском; строгая глобальная оптимальность не установлена.

## К защите

1. Дискретная позиция хромосомы непосредственно обозначает смену, декодирование тождественно.
2. Все жёсткие условия централизованы в `violations`; предпочтения относятся к мягкой цели.
3. Ремонт сохраняет допустимость, обмен сам по себе её не гарантирует.
4. Большой штраф отделяет допустимость от цены предпочтений, но не гарантирует её достижение за конечный бюджет.
5. При росте числа смен восстановление становится дороже; уместны локальные обмены цепочками и отдельный учёт времени.
"""
    (output.parent / "REPORT.md").write_text(report,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="План дежурств на неделю")
    parser.add_argument("--config",type=Path,default=ROOT / "lab2/config.json",help="Файл параметров")
    parser.add_argument("--data",type=Path,default=ROOT / "lab2/data.json",help="Входные данные")
    parser.add_argument("--output",type=Path,default=ROOT / "lab2/results",help="Папка результатов")
    args = parser.parse_args()
    run(read_json(args.config),args.data,args.output)


if __name__ == "__main__":
    main()
