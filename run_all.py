"""Последовательно выполнить три лабораторные и проверить проект."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    for module in ("lab1.run","lab2.run","lab3.run"):
        subprocess.run([sys.executable,"-m",module],cwd=root,env=environment,check=True)
    subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=root,env=environment,check=True)


if __name__ == "__main__":
    main()
