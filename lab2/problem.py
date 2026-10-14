"""Данные, все жёсткие ограничения и восстановление допустимости расписания."""

import random
from collections import Counter

from evolab.common import read_json, write_json

DAYS = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
SHIFTS = ["Утро", "День", "Ночь"]


def generate_data(path, seed=260010):
    rng = random.Random(seed)
    # Свидетель допустимости нужен только для проверки данных и аварийного ремонта.
    witness = []
    for day in range(7):
        witness.extend([4+day % 4, 4+(day+1) % 4, day % 4])
    employees = [{"id": i, "name": f"Сотрудник {i+1}", "max_shifts": 4,
                  "night_qualified": i < 4} for i in range(8)]
    events = []
    for i in range(21):
        domain = list(range(4)) if i % 3 == 2 else list(range(8))
        unavailable = [e for e in domain if e != witness[i] and rng.random() < .17]
        events.append({"id": i, "day": i // 3, "shift": i % 3,
                       "unavailable": unavailable, "preference_cost": [rng.randint(1,10) for _ in range(8)]})
    conflicts = []
    for day in range(7):
        a = 3*day
        conflicts.extend([[a,a+1], [a,a+2], [a+1,a+2]])
    conflicts.extend([[3*day+2,3*(day+1)] for day in range(6)])
    data = {"seed": seed, "employees": employees, "events": events, "conflicts": conflicts,
            "feasible_witness": witness, "description": "21 смена, 8 сотрудников, 27 пар конфликтов"}
    write_json(path, data)
    return data


class ScheduleProblem:
    def __init__(self, data, imbalance_weight=2, node_limit=5000):
        self.data = data
        self.events = data["events"]
        self.employees = data["employees"]
        self.conflicts = data["conflicts"]
        self.weight = imbalance_weight
        self.node_limit = node_limit
        self.fallbacks = 0
        self.neighbors = [set() for _ in self.events]
        for a,b in self.conflicts:
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)
        self.domains = [[e["id"] for e in self.employees if e["id"] not in event["unavailable"]
                         and (event["shift"] != 2 or e["night_qualified"])] for event in self.events]
        if any(not d for d in self.domains) or self.violations(data["feasible_witness"]):
            raise ValueError("Входные данные не содержат корректного свидетеля допустимости")

    def violations(self, schedule):
        if len(schedule) != len(self.events):
            return ["Неверное число смен"]
        errors = []
        for i,e in enumerate(schedule):
            if not isinstance(e, int) or isinstance(e, bool) or not 0 <= e < len(self.employees):
                errors.append(f"Смена {i}: неверный сотрудник")
                continue
            event = self.events[i]
            if e in event["unavailable"]:
                errors.append(f"Смена {i}: сотрудник недоступен")
            if event["shift"] == 2 and not self.employees[e]["night_qualified"]:
                errors.append(f"Смена {i}: нет допуска к ночному дежурству")
        for a,b in self.conflicts:
            if schedule[a] == schedule[b]:
                errors.append(f"Конфликт смен {a} и {b}")
        counts = Counter(schedule)
        for employee in self.employees:
            excess = max(0, counts[employee["id"]]-employee["max_shifts"])
            errors.extend([f"Превышение недельного предела сотрудника {employee['id']}"]*excess)
        return errors

    def soft_cost(self, schedule):
        if len(schedule) != len(self.events) or any(not isinstance(e,int) or not 0 <= e < len(self.employees) for e in schedule):
            raise ValueError("Некорректный вектор назначений")
        counts = Counter(schedule)
        preferences = sum(event["preference_cost"][schedule[i]] for i,event in enumerate(self.events))
        imbalance = sum(abs(counts[e["id"]]-3) for e in self.employees)
        return preferences + self.weight*imbalance

    def initialize(self, rng):
        return [rng.choice(domain) for domain in self.domains]

    def repair(self, schedule, rng):
        if not self.violations(schedule):
            return schedule[:]
        result = [-1]*len(self.events)
        counts = [0]*len(self.employees)
        nodes = 0
        def candidates(i):
            return [e for e in self.domains[i] if counts[e] < self.employees[e]["max_shifts"]
                    and all(result[j] != e for j in self.neighbors[i])]
        def search(remaining):
            nonlocal nodes
            nodes += 1
            if not remaining:
                return True
            if nodes > self.node_limit:
                return False
            available = [(i,candidates(i)) for i in remaining]
            i, options = min(available, key=lambda pair: (len(pair[1]),-len(self.neighbors[pair[0]]),pair[0]))
            if not options:
                return False
            # Сначала сохраняется ген родителя, затем учитываются загрузка и предпочтения.
            tie = {e: rng.random() for e in options}
            options.sort(key=lambda e: (e != schedule[i], counts[e], self.events[i]["preference_cost"][e], tie[e]))
            for e in options:
                result[i] = e
                counts[e] += 1
                if search([j for j in remaining if j != i]):
                    return True
                counts[e] -= 1
                result[i] = -1
            return False
        if not search(list(range(len(self.events)))):
            self.fallbacks += 1
            result = self.data["feasible_witness"][:]
        if self.violations(result):
            raise RuntimeError("Восстановление не обеспечило допустимость")
        return result

    def greedy(self):
        result = [-1]*len(self.events)
        counts = [0]*len(self.employees)
        for i in sorted(range(len(self.events)), key=lambda j: (len(self.domains[j]),-len(self.neighbors[j]),j)):
            options = [e for e in self.domains[i] if counts[e] < self.employees[e]["max_shifts"]
                       and all(result[j] != e for j in self.neighbors[i])]
            if not options:
                return self.repair(result, random.Random(0))
            e = min(options, key=lambda e: self.events[i]["preference_cost"][e] + self.weight*counts[e])
            result[i] = e
            counts[e] += 1
        return result

    def rows(self, schedule):
        return [{"day": DAYS[event["day"]], "shift": SHIFTS[event["shift"]],
                 "employee": self.employees[schedule[i]]["name"],
                 "preference_cost": event["preference_cost"][schedule[i]]} for i,event in enumerate(self.events)]
