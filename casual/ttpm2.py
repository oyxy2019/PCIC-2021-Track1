#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : ttpm.py
@Time        : 2021-06-22 16:55:26
@Author      : dreamhomes
@Description : TTPM model with topology.
"""
import os
import time
import matplotlib.pyplot as plt

import networkx as nx
import numexpr as ne
import numpy as np
import pandas as pd
from castle.algorithms import TTPM
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from loguru import logger

from pyvis.network import Network

ne.set_vml_num_threads(12)
cnt = 0


def preprocessing(alarm_data):
    """alarm data preprocessing.

    Args:
        alarm_data ([type]): [description]
    """

    # 设置列名
    X = alarm_data.iloc[:, 0:4]
    X.columns = ["event", "node", "timestamp", "timestamp2"]
    X = X.reindex(columns=["event", "timestamp", "timestamp2", "node"])

    X = dataclean(X)
    return X

    # event_names = np.array(list(set(X['event'])))
    # event_names.sort()
    # N = len(event_names)
    # print(event_names)
    # freq_table = np.zeros((N, N))
    #
    # for v1 in event_names:
    #     for v2 in event_names:
    #         if v1 != v2:
    #             print(v1, v2)
    #             v1_t1s = list(X.loc[X['event'] == v1]['timestamp'])
    #             v2_t1s = list(X.loc[X['event'] == v2]['timestamp'])
    #             print(v1_t1s)
    #             print(v2_t1s)
    # return X

    # 数据可视化
    event_names = np.array(list(set(X['event'])))
    event_names.sort()
    print("events: ", event_names)
    print(type(X))
    print("length of alram_data:", len(X))

    # print(X.head())

    def plot_fun(x):
        print(x[:10])
        t1 = x['timestamp']
        t2 = x['timestamp2']
        # t1 = x['timestamp'][0:10]
        # t2 = x['timestamp2'][0:10]

        t1 = list(t1)
        t2 = list(t2)
        # print("--------------------------")
        # print(list1)
        # print(list2)

        flag = False
        for i in range(len(t1) - 1):

            # if t1[i] == t2[i]:                            # 一条数据前后时间戳相等的, 注意for不要减1：6278
            # if t1[i] == t1[i+1] and t2[i] == t2[i+1]:     # 两条数据完全重叠的：86
            # if t1[i] < t1[i+1] < t2[i+1] < t2[i]:         # 两条数据, 一条包围另一条的：54504
            if t1[i] < t1[i + 1] <= t2[i] < t2[i + 1]:  # 两条数据有交集的：53224, 56463

                global cnt
                cnt += 1
                flag = True

                # 显示一个例子
                # print("%%%%%%%%%%%%%%%%%%%%%%%")
                # print(i)
                # print(t1[i], t2[i])
                # print(t1[i+1], t2[i+1])
                # print("************************")
                # plt.plot([t1[i], t1[i], t2[i], t2[i]], [0, 1, 1, 0])
                # plt.plot([t1[i+1], t1[i+1], t2[i+1], t2[i+1]], [0, 1, 1, 0])
                # plt.show(block=True)

        # if flag:
        #     plt.plot([t1, t1, t2, t2], [0, 0.5, 0.5, 0])
        #     plt.show(block=True)

    group = X.groupby(['event', 'node']).apply(plot_fun)

    print("cnt: ", cnt)
    print('percent: {:.2%}'.format(cnt / len(X)))

    logger.info("Data preprocessing finished.")
    return X


def model_fit(train_data, topo_matrix, iters):
    """model train.

    Args:
        train_data (pd.dataframe): alert data.
        topo_matrix (np.array): device topology.
    """
    model_time_start = time.time()

    ttpm = TTPM(topo_matrix, max_iter=iters, max_hop=2)

    dag_matrix = np.load(dag_path)
    ttpm.learn(train_data, dag_matrix)  # 迭代时间非常长...

    est_causal_matrix = ttpm.causal_matrix

    model_time_end = time.time()

    logger.info(f"Model fitting finished. Elapsed time: {model_time_end - model_time_start}")

    return est_causal_matrix


def evaluate(est_causal_matrix, true_graph_matrix):
    """evaluation.

    Args:
        est_causal_matrix (np.array): estimate casual graph
        true_graph_matrix (np.array): true graph
    """
    g_score = MetricsDAG(est_causal_matrix, true_graph_matrix).metrics["gscore"]
    logger.info(f"g-score: {g_score}")

    TP = []
    FP = []
    FN = []
    for i in range(len(est_causal_matrix)):
        for j in range(len(est_causal_matrix)):
            if est_causal_matrix[i][j] == 1 and true_graph_matrix[i][j] == 1:
                TP.append((i, j))
            if est_causal_matrix[i][j] == 1 and true_graph_matrix[i][j] == 0:
                FP.append((i, j))
            if est_causal_matrix[i][j] == 0 and true_graph_matrix[i][j] == 1:
                FN.append((i, j))
    logger.info("TP {}".format(len(TP)))
    logger.info("FP {}".format(len(FP)))
    logger.info("FN {}".format(len(FN)))

    _g_score = max(0.0, (len(TP) - len(FP))) / (len(TP) + len(FN))
    logger.info(f"g-score(Ref): {_g_score}")  # 源码 False Positives + Reversed Edges

    return g_score


def Draw_graph(graph_matrix, name="graph"):
    """draw graph.

    Args:
        graph_matrix ([type]): [description]
    """
    net = Network("500px", "900px", notebook=False, directed=True, layout=False)
    g = nx.from_numpy_matrix(graph_matrix)
    net.from_nx(g)

    os.makedirs("../output/draw_graphs/", exist_ok=True)
    net.show(f"../output/draw_graphs/{name}.html")  # 打开html路径有点问题，请手动打开html


if __name__ == "__main__":
    dataset_name = "18V_55N_Wireless"
    dataset_path = "../datasets/alarm/25V_474N_Microwave"
    alarm_path = f"{dataset_path}/Alarm.csv"
    # topo_path = f"{dataset_path}/Topology.npy"
    dag_path = f"{dataset_path}/DAG.npy"
    draw_graph = False
    iters = 1

    logger.info("---start---")

    os.makedirs("../output", exist_ok=True)
    # 历史告警
    alarm_data = pd.read_csv(alarm_path, encoding="utf")
    # 拓扑图
    # topo_matrix = np.load(topo_path)

    alarm_data = preprocessing(alarm_data)

    # est_causal_matrix = model_fit(alarm_data, topo_matrix, iters)

    # os.makedirs("../output/est_graphs", exist_ok=True)
    # np.save(f"../output/est_graphs/{iters}-{dataset_name}.npy", est_causal_matrix)
    #
    # if dag_path:
    #     # 因果图
    #     dag_matrix = np.load(dag_path)
    #     evaluate(est_causal_matrix, dag_matrix)
    #     GraphDAG(est_causal_matrix, dag_matrix)
    #     if draw_graph:
    #         Draw_graph(est_causal_matrix, f"est-{dataset_name}")
    #         Draw_graph(dag_matrix, f"true-{dataset_name}")

    logger.info("---finished---")
