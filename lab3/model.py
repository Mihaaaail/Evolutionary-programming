"""Синтетические данные и малый классификатор без готовых средств обучения."""

import csv
import math
import random
import statistics
import time

import numpy as np

from evolab.common import write_csv


def generate_data(path,seed=260310):
    rng = random.Random(seed)
    rows = []
    for i in range(360):
        x = [rng.uniform(-2,2) for _ in range(4)]
        score = x[0]*x[1] + .55*x[2] - .25*x[3] + rng.gauss(0,.15)
        label = int(score > 0)
        rows.append({"id":i,"split":"train" if i < 200 else "validation" if i < 280 else "test",
                     **{f"x{j+1}":value for j,value in enumerate(x)},"label":label})
    write_csv(path,rows)


def load_data(path,include_test=False):
    groups = {"train":[],"validation":[]}
    if include_test:
        groups["test"] = []
    with path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in groups:
                groups[row["split"]].append(([float(row[f"x{i}"]) for i in range(1,5)],int(row["label"])))
    arrays = {key:(np.array([r[0] for r in rows]),np.array([r[1] for r in rows],dtype=float)) for key,rows in groups.items()}
    if any(len(x) == 0 for x,y in arrays.values()):
        raise ValueError("Каждая используемая часть данных должна быть непустой")
    train,_ = arrays["train"]
    mean,std = train.mean(axis=0),train.std(axis=0)
    std = np.where(std > 0,std,1)
    return {key:((x-mean)/std,y) for key,(x,y) in arrays.items()},mean,std


def decode(genome):
    hidden,epochs,log_rate,l2_flag = genome
    return {"hidden":int(hidden),"epochs":int(epochs),"learning_rate":10**log_rate,
            "l2":.001 if l2_flag else 0.0}


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))


def train_model(x,y,parameters,model_seed=261010):
    """Полный пакетный градиентный спуск по перекрёстной энтропии."""
    rng = np.random.default_rng(model_seed)
    hidden = parameters["hidden"]
    w1 = rng.normal(0,1/math.sqrt(x.shape[1]),size=(x.shape[1],hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0,1/math.sqrt(hidden),size=hidden)
    b2 = 0.0
    rate,l2 = parameters["learning_rate"],parameters["l2"]
    for _ in range(parameters["epochs"]):
        activation = np.tanh(x @ w1 + b1)
        predicted = sigmoid(activation @ w2 + b2)
        residual = (predicted-y)/len(y)
        dw2 = activation.T @ residual + l2*w2
        db2 = float(residual.sum())
        hidden_delta = residual[:,None]*w2[None,:]*(1-activation*activation)
        dw1 = x.T @ hidden_delta + l2*w1
        db1 = hidden_delta.sum(axis=0)
        w1 -= rate*dw1
        b1 -= rate*db1
        w2 -= rate*dw2
        b2 -= rate*db2
    return {"w1":w1,"b1":b1,"w2":w2,"b2":b2}


def predict(model,x):
    return (sigmoid(np.tanh(x @ model["w1"] + model["b1"]) @ model["w2"] + model["b2"]) >= .5).astype(int)


def accuracy(model,data):
    x,y = data
    return float(np.mean(predict(model,x) == y))


def parameter_count(hidden,features=4):
    return features*hidden + hidden + hidden + 1


class Evaluator:
    def __init__(self,data,config):
        self.data = data
        self.config = config
        self.records = []

    def __call__(self,genome):
        parameters = decode(genome)
        x,y = self.data["train"]
        timings = []
        model = None
        # Не применяется кэш: каждое обращение действительно обучает модель.
        for _ in range(self.config["training_repeats"]):
            start = time.perf_counter_ns()
            model = train_model(x,y,parameters,self.config["model_seed"])
            timings.append((time.perf_counter_ns()-start)/1e6)
        error = 1-accuracy(model,self.data["validation"])
        objectives = (error,statistics.median(timings),parameter_count(parameters["hidden"]))
        self.records.append({"evaluation":len(self.records)+1,"genome":genome[:],"parameters":parameters,
                             "objectives":list(objectives),"training_ms_samples":timings})
        return objectives


def linear_baseline(data,epochs=200,rate=.1):
    x,y = data["train"]
    w,b = np.zeros(x.shape[1]),0.0
    for _ in range(epochs):
        residual = (sigmoid(x @ w + b)-y)/len(y)
        w -= rate*(x.T @ residual)
        b -= rate*residual.sum()
    result = {}
    for split,(features,labels) in data.items():
        result[split] = float(np.mean((sigmoid(features @ w+b) >= .5) == labels))
    return result
