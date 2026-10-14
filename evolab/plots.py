"""Автономные графики SVG без зависимости от графических библиотек."""

from html import escape
from pathlib import Path
from statistics import mean

COLORS = ["#1769aa", "#d35400", "#2b8b57", "#9c3d99"]


def chart(path, series, title, xlabel, ylabel, scatter=False):
    width, height = 960, 560
    left, top, right, bottom = 100, 70, 30, 90
    all_points = [point for _, points in series for point in points]
    if not all_points:
        raise ValueError("Для графика нужны точки")
    xmin, xmax = min(p[0] for p in all_points), max(p[0] for p in all_points)
    ymin, ymax = min(p[1] for p in all_points), max(p[1] for p in all_points)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + 1
    span = ymax - ymin
    ymin, ymax = ymin - .05 * span, ymax + .05 * span
    def xy(x, y):
        return (left + (x-xmin)/(xmax-xmin)*(width-left-right),
                height-bottom-(y-ymin)/(ymax-ymin)*(height-top-bottom))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560" role="img">',
             f'<title>{escape(title)}</title>', '<rect width="960" height="560" fill="white"/>',
             '<g font-family="Arial, sans-serif" font-size="14" fill="#222">',
             f'<text x="480" y="30" text-anchor="middle" font-size="20">{escape(title)}</text>']
    for i in range(6):
        x = xmin + i*(xmax-xmin)/5
        px, _ = xy(x, ymin)
        y = ymin + i*(ymax-ymin)/5
        _, py = xy(xmin, y)
        parts += [f'<path d="M {px:.2f} {top} V {height-bottom}" stroke="#e3e7eb"/>',
                  f'<path d="M {left} {py:.2f} H {width-right}" stroke="#e3e7eb"/>',
                  f'<text x="{px:.2f}" y="{height-bottom+25}" text-anchor="middle">{x:.4g}</text>',
                  f'<text x="{left-12}" y="{py+5:.2f}" text-anchor="end">{y:.4g}</text>']
    for i, (name, points) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        if scatter:
            parts += [f'<circle cx="{xy(x,y)[0]:.2f}" cy="{xy(x,y)[1]:.2f}" r="3.5" fill="{color}" fill-opacity="0.65"/>' for x,y in points]
        else:
            coords = " ".join(f"{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}" for x,y in points)
            parts += [f'<polyline points="{coords}" stroke="{color}" stroke-width="2" fill="none"/>']
        lx = left + (i % 4)*205
        parts += [f'<rect x="{lx}" y="{height-30}" width="15" height="8" fill="{color}"/>',
                  f'<text x="{lx+22}" y="{height-21}">{escape(name)}</text>']
    parts += [f'<text x="480" y="{height-45}" text-anchor="middle">{escape(xlabel)}</text>',
              f'<text transform="translate(23 270) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>', '</g></svg>']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def convergence(path, histories, title):
    # Последняя неполная генерация тоже включается; бюджет всех запусков одинаков.
    count = min(len(h) for h in histories)
    stats = []
    for i in range(count):
        values = [h[i]["best"] for h in histories]
        stats.append((histories[0][i]["generation"], min(values), mean(values), max(values)))
    chart(path, [(label, [(p[0], p[j]) for p in stats]) for j,label in enumerate(
        ["Минимум", "Среднее", "Максимум"], 1)], title, "Поколение", "Лучшее значение функции")
