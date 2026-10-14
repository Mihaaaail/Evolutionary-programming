"""Применить сохранённую модель к одному наблюдению с четырьмя признаками."""

import argparse
from pathlib import Path

import numpy as np

from evolab.common import ROOT,read_json
from lab3.model import predict


def main():
    parser = argparse.ArgumentParser(description="Классификация сохранённой моделью")
    parser.add_argument("--model",type=Path,default=ROOT / "lab3/results/model-1.json",help="Файл обученной модели")
    parser.add_argument("--features",type=float,nargs=4,required=True,help="Четыре исходных признака")
    args = parser.parse_args()
    if not all(np.isfinite(args.features)):
        parser.error("Признаки должны быть конечными числами")
    saved = read_json(args.model)
    x = (np.array([args.features])-np.array(saved["standardization"]["mean"]))/np.array(saved["standardization"]["std"])
    model = {key:np.asarray(value) if key != "b2" else float(value) for key,value in saved["weights"].items()}
    print(f"Предсказанный класс: {int(predict(model,x)[0])}")


if __name__ == "__main__":
    main()
