"""Функция варианта 10 и отдельный численный ориентир её минимума."""

import math


def component(x, index):
    return -math.sin(x) * math.sin(index * x*x / math.pi) ** 20


def michalewicz(x):
    if not x or not all(math.isfinite(value) and 0 <= value <= math.pi for value in x):
        raise ValueError("Координаты должны лежать в [0, π]")
    return sum(component(value, i) for i,value in enumerate(x, 1))


def reflect(value, low=0.0, high=math.pi):
    # Отражение корректно и при пересечении нескольких ширин области.
    width = high - low
    remainder = (value-low) % (2*width)
    return low + (remainder if remainder <= width else 2*width-remainder)


def numerical_reference(dimension=9, grid=20000):
    """Просмотр всех долин по координатам с золотым сечением в каждой долине.

    Это численный ориентир, а не сертификат глобальной оптимальности.
    Его вычисления не входят в сравнительный бюджет методов поиска.
    """
    coordinates = []
    for index in range(1, dimension+1):
        values = [component(math.pi*j/grid, index) for j in range(grid+1)]
        intervals = [j for j in range(1, grid) if values[j] <= values[j-1] and values[j] <= values[j+1]]
        candidates = [0.0, math.pi]
        for j in intervals:
            low, high = math.pi*(j-1)/grid, math.pi*(j+1)/grid
            ratio = (math.sqrt(5)-1)/2
            a, b = high-ratio*(high-low), low+ratio*(high-low)
            fa, fb = component(a, index), component(b, index)
            for _ in range(65):
                if fa < fb:
                    high, b, fb = b, a, fa
                    a = high-ratio*(high-low)
                    fa = component(a, index)
                else:
                    low, a, fa = a, b, fb
                    b = low+ratio*(high-low)
                    fb = component(b, index)
            candidates.append((low+high)/2)
        coordinates.append(min(candidates, key=lambda x: component(x, index)))
    return {"point": coordinates, "value": michalewicz(coordinates), "grid_per_coordinate": grid,
            "status": "Численный ориентир по разделимой функции, без строгого сертификата"}
