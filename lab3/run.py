"""NSGA-II и взвешенная сумма на трёх критериях малого классификатора."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from evolab.common import ROOT,describe,environment,read_json,table,validate_config,write_csv,write_json
from evolab.pareto import additive_epsilon,dominates,nondominated_sort,nsga2
from evolab.plots import chart
from lab3.model import Evaluator,accuracy,decode,generate_data,linear_baseline,load_data,parameter_count,train_model

NAMES = {"nsga2":"NSGA-II","weighted":"Взвешенная сумма"}


def operators(config):
    def initialize(rng):
        return [rng.randint(config["hidden_min"],config["hidden_max"]),
                rng.randint(config["epochs_min"],config["epochs_max"]),
                rng.uniform(config["log10_learning_rate_min"],config["log10_learning_rate_max"]),rng.randrange(2)]
    def vary(a,b,rng,progress):
        child = a[:]
        if rng.random() < config["crossover_probability"]:
            child = [x if rng.random() < .5 else y for x,y in zip(a,b)]
        if rng.random() < config["mutation_probability"]:
            child[0] += rng.choice([-2,-1,1,2])
        if rng.random() < config["mutation_probability"]:
            child[1] += rng.choice([-10,-5,5,10])
        if rng.random() < config["mutation_probability"]:
            child[2] += rng.gauss(0,.2)
        if rng.random() < config["mutation_probability"]:
            child[3] = 1-child[3]
        child[0] = max(config["hidden_min"],min(config["hidden_max"],child[0]))
        child[1] = max(config["epochs_min"],min(config["epochs_max"],child[1]))
        child[2] = max(config["log10_learning_rate_min"],min(config["log10_learning_rate_max"],child[2]))
        return child
    return initialize,vary


def weighted_ga(initialize,evaluate,vary,config,seed,weights):
    rng = random.Random(seed)
    start = time.perf_counter()
    population = [initialize(rng) for _ in range(config["population"])]
    objectives = [evaluate(x) for x in population]
    evaluations = len(population)
    history = []
    generation = 0
    def score(value):
        return sum(w*v/s for w,v,s in zip(weights,value,config["objective_scales"]))
    while True:
        history.append({"generation":generation,"evaluations":evaluations,
                        "front_size":len(nondominated_sort(objectives)[0]),"best_error":min(o[0] for o in objectives)})
        if evaluations == config["budget"]:
            break
        def parent():
            i,j = rng.randrange(len(population)),rng.randrange(len(population))
            return population[min((i,j),key=lambda k:score(objectives[k]))][:]
        children,values = [],[]
        for _ in range(min(config["population"],config["budget"]-evaluations)):
            child = vary(parent(),parent(),rng,evaluations/config["budget"])
            children.append(child)
            values.append(evaluate(child))
            evaluations += 1
        combined,combined_values = population+children,objectives+values
        chosen = sorted(range(len(combined)),key=lambda i:score(combined_values[i]))[:config["population"]]
        population = [combined[i] for i in chosen]
        objectives = [combined_values[i] for i in chosen]
        generation += 1
    return {"population":population,"objectives":objectives,"evaluations":evaluations,"history":history,
            "seconds":time.perf_counter()-start}


def empirical_archive(records):
    # Сначала одинаковые настройки объединяются: выбирается медианное время повторных оценок.
    # Качество при одной настройке детерминировано; медиана уменьшает шум измерений.
    import statistics
    grouped = {}
    for record in records:
        key = tuple(record["genome"])
        grouped.setdefault(key,[]).append(record)
    unique = []
    for items in grouped.values():
        record = dict(items[0])
        record["objectives"] = [record["objectives"][0],statistics.median(r["objectives"][1] for r in items),record["objectives"][2]]
        unique.append(record)
    return [unique[i] for i in nondominated_sort([r["objectives"] for r in unique])[0]]


def run(config,data_path,output):
    validate_config(config)
    if config["training_repeats"] < 3:
        raise ValueError("Для устойчивого измерения времени нужны минимум три повтора обучения")
    if not data_path.exists():
        generate_data(data_path,config["data_seed"])
    data,mean,std = load_data(data_path,include_test=False)
    output.mkdir(parents=True,exist_ok=True)
    env = environment(config)
    env["numpy"] = np.__version__
    env["standardization"] = {"mean":mean.tolist(),"std":std.tolist()}
    write_json(output / "environment.json",env)
    # Разогрев библиотеки выполняется до первого измерения и одинаков для методов.
    train_model(*data["train"],{"hidden":4,"epochs":5,"learning_rate":.1,"l2":0},config["model_seed"])
    initialize,vary = operators(config)
    rows,histories,front_records = [],[],[]
    finals = []
    log_path = output / "evaluations.jsonl"
    with log_path.open("w",encoding="utf-8") as log:
        for method in NAMES:
            for index in range(config["runs"]):
                seed = config["first_seed"]+index
                evaluator = Evaluator(data,config)
                weights = config["weight_vectors"][index % len(config["weight_vectors"])]
                if method == "nsga2":
                    result = nsga2(initialize,evaluator,vary,population_size=config["population"],budget=config["budget"],seed=seed)
                else:
                    result = weighted_ga(initialize,evaluator,vary,config,seed,weights)
                front = nondominated_sort(result["objectives"])[0]
                seen = set()
                unique_front = []
                for i in front:
                    record = {"method":method,"seed":seed,"genome":result["population"][i],
                              "parameters":decode(result["population"][i]),"objectives":list(result["objectives"][i])}
                    key = (tuple(record["genome"]),tuple(record["objectives"]))
                    if key not in seen:
                        seen.add(key)
                        unique_front.append(record)
                front_records.extend(unique_front)
                finals.append({"method":method,"seed":seed,"front":unique_front})
                rows.append({"method":method,"seed":seed,"best_validation_accuracy":1-min(o[0] for o in result["objectives"]),
                             "front_size":len(unique_front),"evaluations":result["evaluations"],
                             "training_calls":result["evaluations"]*config["training_repeats"],"seconds":result["seconds"],
                             "weights":json.dumps(weights) if method == "weighted" else ""})
                histories.extend({"method":method,"seed":seed,**h} for h in result["history"])
                for record in evaluator.records:
                    enriched = {"method":method,"seed":seed,**record}
                    log.write(json.dumps(enriched,ensure_ascii=False)+"\n")
                if (index+1) % 5 == 0:
                    print(f"Лабораторная 3: {NAMES[method]}, завершено {index+1}/{config['runs']} запусков",flush=True)
    archive = empirical_archive(front_records)
    write_json(output / "pareto.json",archive)
    write_json(output / "final_fronts.json",finals)
    reference = [r["objectives"] for r in archive]
    for row,final in zip(rows,finals):
        row["epsilon"] = additive_epsilon([r["objectives"] for r in final["front"]],reference,config["objective_scales"])
    summaries = {method:describe(r["epsilon"] for r in rows if r["method"] == method) for method in NAMES}
    qualities = {method:describe(1-r["best_validation_accuracy"] for r in rows if r["method"] == method) for method in NAMES}
    write_csv(output / "runs.csv",rows)
    write_csv(output / "history.csv",histories)
    write_json(output / "summary.json",{"epsilon":summaries,"best_validation_error":qualities,
                                        "archive_size":len(archive),"reference":"Общий эмпирический фронт финальных популяций"})
    write_csv(output / "pareto.csv",[{"method":r["method"],"seed":r["seed"],"validation_accuracy":1-r["objectives"][0],
                                     "training_ms":r["objectives"][1],"parameters":r["objectives"][2],**r["parameters"]} for r in archive])
    for xindex,yindex,name,xlabel,ylabel in [(2,0,"quality-complexity","Число параметров","Ошибка на проверочных данных"),
                                            (1,0,"quality-time","Время обучения, мс","Ошибка на проверочных данных"),
                                            (2,1,"time-complexity","Число параметров","Время обучения, мс")]:
        series = [(NAMES[m],[(r["objectives"][xindex],r["objectives"][yindex]) for r in archive if r["method"] == m]) for m in NAMES]
        chart(output / f"pareto-{name}.svg",series,"Проекция общего эмпирического фронта",xlabel,ylabel,scatter=True)
    # Три модели выбираются исключительно по проверочным критериям.
    selected = []
    used = set()
    priorities = [("Наиболее точная",lambda r:(r["objectives"][0],r["objectives"][2],r["objectives"][1])),
                  ("Наиболее компактная",lambda r:(r["objectives"][2],r["objectives"][0],r["objectives"][1])),
                  ("Сбалансированная",lambda r:sum(o/s for o,s in zip(r["objectives"],config["objective_scales"])))]
    for label,key in priorities:
        candidates = [r for r in archive if tuple(r["genome"]) not in used]
        if not candidates:
            raise RuntimeError("Недостаточно различных моделей во фронте для трёх компромиссов")
        chosen = min(candidates,key=key)
        used.add(tuple(chosen["genome"]))
        selected.append({"label":label,**chosen})
    # Испытательные метки загружаются впервые только после фиксации трёх моделей.
    full_data,_,_ = load_data(data_path,include_test=True)
    for index,item in enumerate(selected):
        model = train_model(*full_data["train"],item["parameters"],config["model_seed"])
        item["test_accuracy"] = accuracy(model,full_data["test"])
        item["train_accuracy"] = accuracy(model,full_data["train"])
        saved = {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in model.items()}
        write_json(output / f"model-{index+1}.json",{"label":item["label"],"parameters":item["parameters"],
                                                   "standardization":env["standardization"],"weights":saved})
    baseline = linear_baseline(full_data)
    write_json(output / "representatives.json",selected)
    write_json(output / "linear_baseline.json",baseline)
    representative_table = "| Решение | Скрытых узлов | Эпох | Параметров | Точность проверки | Точность испытания | Время, мс |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for item in selected:
        representative_table += f"| {item['label']} | {item['parameters']['hidden']} | {item['parameters']['epochs']} | {item['objectives'][2]} | {1-item['objectives'][0]:.3f} | {item['test_accuracy']:.3f} | {item['objectives'][1]:.3f} |\n"
    winning = min(summaries,key=lambda m:summaries[m]["mean"])
    report = f"""# Лабораторная работа № 3

## Постановка и направление

Вариант 10, направление А: подбор параметров малого классификатора.
N = 28, группа 517, Q = 1, Y = 26. Требуется максимизировать качество и
минимизировать время обучения и число параметров. Выбрано направление А;
в задании отсутствует дополнительное правило выбора между А и Б.

Используется нейронная сеть с четырьмя входами, одним скрытым слоем tanh и одним
логистическим выходом. Оптимизируются число скрытых узлов {config['hidden_min']}–{config['hidden_max']},
число эпох {config['epochs_min']}–{config['epochs_max']}, десятичный логарифм скорости обучения
{config['log10_learning_rate_min']}–{config['log10_learning_rate_max']} и включение L2-регуляризации 0 либо 0,001.
Генотип — смешанный вектор этих четырёх параметров; фенотип — обученная сеть.
Обучение реализовано вручную полным пакетным градиентным спуском по перекрёстной
энтропии. NumPy используется для матричных операций, готовой модели и оптимизатора нет.

## Данные и цели

360 синтетических наблюдений с четырьмя признаками из равномерного распределения
[−2,2]. Метка равна 1 при x₁x₂ + 0,55x₃ − 0,25x₄ + ε > 0,
ε — гауссовский шум с σ = 0,15. Начальное состояние генерации {config['data_seed']}.
Строки 0–199 обучающие, 200–279 проверочные, 280–359 испытательные.
Наблюдения независимы и одинаково распределены; файл `data.csv` включён в репозиторий.
Среднее и масштаб вычисляются только по обучающей части.
Противоречие критериев: более широкая сеть и длительное обучение могут улучшать
разделение нелинейных классов, но увеличивают затраты и число параметров.

Вектор минимизируемых целей:

1. f₁ = 1 − доля правильных ответов на проверочных данных.
2. f₂ = медианное фактически измеренное время обучения в миллисекундах
   по {config['training_repeats']} повторным обучениям с одинаковой инициализацией.
3. f₃ = число обучаемых параметров = 6h + 1, включая смещения.

Начальное состояние инициализации весов {config['model_seed']} одинаково для всех
оценок одной архитектуры, поэтому качество при фиксированных настройках повторяется.
Измерение времени включает создание весов и обучение, исключает подготовку данных
и проверку точности. До опыта выполняется разогрев библиотеки.
Кэширование оценок не применяется. Число обращений к функции и число реальных
обучений указаны отдельно.

## Алгоритм

Самостоятельно реализованы попарное доминирование, недоминируемая сортировка,
нормированное расстояние до скопления, бинарный турнир по рангу и расстоянию,
выбор следующего поколения из объединения родителей и потомков.
Доминирование требует неухудшения всех целей и строгого улучшения хотя бы одной.
Скрещивание равномерное по четырём генам; мутация изменяет ширину, число эпох,
логарифм скорости или переключает регуляризацию; границы восстанавливаются обрезанием.
Новый относительно первых двух работ механизм — сортировка по Парето и сохранение
разнообразия фронта. Описание исходного алгоритма:
[Deb и соавторы, 2002](https://doi.org/10.1109/4235.996017).

```mermaid
flowchart TD
    A[Параметры моделей] --> B[Обучение и три критерия]
    B --> C[Ранги Парето и расстояния]
    C --> D{{Исчерпан бюджет?}}
    D -- Нет --> E[Турнир и вариация]
    E --> F[Обучение потомков]
    F --> G[Объединение и элитный выбор]
    G --> C
    D -- Да --> H[Фронты и выбор трёх компромиссов]
    H --> I[Однократное испытание выбранных моделей]
```

## Сравнение и параметры

NSGA-II сравнивается с ГА взвешенной суммы. Операторы, исходные распределения,
популяция {config['population']} и бюджет {config['budget']} одинаковы.
Различаются только правила отбора родителей и элитного выбора.
Взвешенная сумма использует нормировки {config['objective_scales']} и циклически
назначенные по номеру запуска веса {config['weight_vectors']}.
Это анализ трёх предпочтений при суммарных {config['runs']} запусках, а не отдельные
серии по 20 для каждого вектора весов; точные веса каждого запуска сохранены в CSV.
Вероятность скрещивания {config['crossover_probability']}, вероятность мутации каждого гена
{config['mutation_probability']}, начальные состояния поиска
{config['first_seed']}–{config['first_seed']+config['runs']-1}.

Бюджет каждого метода: {config['budget']} оценок и
{config['budget']*config['training_repeats']} обучений за запуск. Полученные множества
сравниваются по аддитивному ε-индикатору к общему эмпирическому фронту из финальных
популяций обоих методов. Разности критериев делятся на указанные фиксированные масштабы.
ε показывает худшее необходимое ухудшение для покрытия общей опорной выборки,
меньше лучше. Опорная выборка не является известным истинным фронтом.

## Результаты

По {config['runs']} независимых запусков двух алгоритмов.
Статистика ε-индикатора:

{table(summaries,NAMES)}

Статистика наименьшей проверочной ошибки в финальной популяции:

{table(qualities,NAMES)}

Общий эмпирический фронт содержит {len(archive)} различных настроек.
При объединении одинаковых настроек используется медиана наблюдавшегося времени;
затем снова выполняется недоминируемая сортировка.

![Точность и сложность](results/pareto-quality-complexity.svg)

![Точность и время](results/pareto-quality-time.svg)

![Время и сложность](results/pareto-time-complexity.svg)

## Три компромиссных решения

Настройки выбираются до загрузки испытательных меток. Первое решение минимизирует
проверочную ошибку; второе — число параметров среди оставшихся; третье — сумму
нормированных целей среди оставшихся. Исключение уже выбранных настроек обеспечивает
три различных предметных примера. Все они принадлежат эмпирическому фронту.

{representative_table}

Наиболее точная модель подходит при приоритете качества; компактная — при ограничении
памяти и вычислений; сбалансированная отражает равноценность нормированных целей.
Обученные веса и нормировка сохранены в `results/model-1.json`–`model-3.json`.
Собственная линейная логистическая модель, обученная 200 эпох со скоростью 0,1,
имеет проверочную точность {baseline['validation']:.3f} и испытательную {baseline['test']:.3f}.
Она служит дополнительным ориентиром качества, а не равнобюджетной серией поиска.

## Выводы и ограничения

В этой серии {NAMES[winning]} имеет меньший средний ε-индикатор.
Взвешенная сумма зависит от заданных весов и масштаба критериев и может пропускать
участки невыпуклого фронта; NSGA-II сохраняет разные компромиссы без единой свёртки.
Практическую приемлемость каждой модели показывают испытательная точность и её
стоимость, а не только лучший ответ на проверочных данных.

Время обучения зависит от оборудования, нагрузки и реализации матричных операций.
Медиана трёх повторов уменьшает шум, но не устраняет его; при перезапуске с тем же
состоянием генератора сортировка по времени может менять дальнейшую траекторию.
В журнале `evaluations.jsonl` сохранены все времена и настройки, а `replay.py`
воспроизводит исходные решения отбора по записанным критериям без нового измерения.
Проверочное качество и веса при одинаковых параметрах воспроизводятся независимо
от времени. Выводы о превосходстве по скорости ограничены этой средой выполнения.
Опыты последовательны; серии методов могли испытывать разную фоновую нагрузку.
Нормировка времени 10 мс является явно заданной шкалой предпочтений, не верхним пределом.
Для серьёзного сравнения уместны чередование методов и повторение в изолированной среде.

Общая опорная выборка построена из результатов самих алгоритмов; ε оценивает их
взаимное покрытие, не расстояние до истинного глобального фронта. Применение модели
к реальным данным требует отдельной проверки; текущая задача является синтетической.

## К защите

1. Три цели сравниваются по доминированию, а не одному числу.
2. Крайние решения получают большое расстояние до скопления для сохранения диапазона компромиссов.
3. Число параметров включает веса обоих слоёв и все смещения.
4. Повторное использование испытательных меток для выбора настроек исказило бы итоговую оценку.
5. Инициализация модели и случайность поиска разделены; шум времени фиксируется в журнале.
"""
    (output.parent / "REPORT.md").write_text(report,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Многокритериальная настройка классификатора")
    parser.add_argument("--config",type=Path,default=ROOT / "lab3/config.json",help="Файл параметров")
    parser.add_argument("--data",type=Path,default=ROOT / "lab3/data.csv",help="Входные данные")
    parser.add_argument("--output",type=Path,default=ROOT / "lab3/results",help="Папка результатов")
    args = parser.parse_args()
    run(read_json(args.config),args.data,args.output)


if __name__ == "__main__":
    main()
