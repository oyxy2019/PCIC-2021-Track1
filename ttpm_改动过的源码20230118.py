# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from castle.common import BaseLearner, Tensor
from castle.metrics import MetricsDAG

plt.ion()

class TTPM(BaseLearner):
    """
    TTPM Algorithm.

    A causal structure learning algorithm based on Topological Hawkes process
     for spatio-temporal event sequences.

    Parameters
    ----------
    topology_matrix: np.matrix
        Interpreted as an adjacency matrix to generate the graph.
        It should have two dimensions, and should be square.

    delta: float, default=0.1
            Time decaying coefficient for the exponential kernel.

    epsilon: int, default=1
        BIC penalty coefficient.

    max_hop: positive int, default=6
        The maximum considered hops in the topology,
        when ``max_hop=0``, it is divided by nodes, regardless of topology.

    penalty: str, default=BIC
        Two optional values: 'BIC' or 'AIC'.
        
    max_iter: int
        Maximum number of iterations.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset
    >>> from castle.algorithms import TTPM
    # Data Simulation for TTPM
    >>> X, true_causal_matrix, topology_matrix = load_dataset('THP_Test')
    >>> ttpm = TTPM(topology_matrix, max_hop=2)
    >>> ttpm.learn(X)
    >>> causal_matrix = ttpm.causal_matrix
    # plot est_dag and true_dag
    >>> GraphDAG(ttpm.causal_matrix, true_causal_matrix)
    # calculate accuracy
    >>> ret_metrix = MetricsDAG(ttpm.causal_matrix, true_causal_matrix)
    >>> ret_metrix.metrics
    """

    def __init__(self, topology_matrix, delta=0.1, epsilon=1,
                 max_hop=0, penalty='BIC', max_iter=20):
        BaseLearner.__init__(self)
        assert isinstance(topology_matrix, np.ndarray),\
            'topology_matrix should be np.matrix object'
        assert topology_matrix.ndim == 2,\
            'topology_matrix should be two dimension'
        assert topology_matrix.shape[0] == topology_matrix.shape[1],\
            'The topology_matrix should be square.'
        self._topo = nx.from_numpy_matrix(topology_matrix,
                                          create_using=nx.Graph)
        # initialize instance variables
        self._penalty = penalty
        self._delta = delta
        self._max_hop = max_hop
        self._epsilon = epsilon
        self._max_iter = max_iter

    def learn(self, tensor, true_dag_matrix, *args, **kwargs):
        """
        Set up and run the TTPM algorithm.

        Parameters
        ----------
        tensor:  pandas.DataFrame
            (V 1.0.0, we'll eliminate this constraint in the next version)
            The tensor is supposed to contain three cols:
                ['event', 'timestamp', 'node']

            Description of the three columns:
                event: event name (type).
                timestamp: occurrence timestamp of event, i.e., '1615962101.0'.
                node: topological node where the event happened.
        """
        self.true_dag_matrix = true_dag_matrix      #oyxy加的2，用于训练过程中评估，以确定迭代次数

        # data type judgment
        if not isinstance(tensor, pd.DataFrame):
            raise TypeError('The tensor type is not correct,'
                            'only receive pd.DataFrame type currently.')

        cols_list = ['event', 'timestamp', 'node']
        for col in cols_list:
            if col not in tensor.columns:
                raise ValueError(
                    "The data tensor should contain column with name {}".format(
                        col))

        # initialize needed values
        self._start_init(tensor)

        # Generate causal matrix (DAG)
        _, raw_causal_matrix = self._hill_climb()
        self._causal_matrix = Tensor(raw_causal_matrix,
                                     index=self._matrix_names,
                                     columns=self._matrix_names)

    # def _start_init(self, tensor):
    #     """
    #     Generates some required initial values.
    #     """
    #     tensor.dropna(axis=0, how='any', inplace=True)
    #     tensor['timestamp'] = tensor['timestamp'].astype(float)
    #
    #     tensor = tensor.groupby(
    #         ['event', 'timestamp', 'node']).apply(len).reset_index()
    #     tensor.columns = ['event', 'timestamp', 'node', 'times']
    #     tensor = tensor.reindex(columns=['node', 'timestamp', 'event', 'times'])
    #     print(tensor)
    #     tensor = tensor.sort_values(['node', 'timestamp'])
    #     self.tensor = tensor[tensor['node'].isin(self._topo.nodes)]
    #
    #     # calculate considered events
    #     self._event_names = np.array(list(set(self.tensor['event'])))
    #     self._event_names.sort()
    #     self._N = len(self._event_names)
    #     self._matrix_names = list(self._event_names.astype(str))
    #
    #     # map event name to corresponding index value
    #     self._event_indexes = self._map_event_to_index(
    #         self.tensor['event'].values, self._event_names)
    #     self.tensor['event'] = self._event_indexes
    #
    #     self._g = self._topo.subgraph(self.tensor['node'].unique())
    #     self._ne_grouped = self.tensor.groupby('node')
    #
    #     self._decay_effects = np.zeros(
    #         [len(self._event_names), self._max_hop+1])  # will be used in EM.
    #
    #     self._max_s_t = tensor['timestamp'].max()
    #     self._min_s_t = tensor['timestamp'].min()
    #
    #     for k in range(self._max_hop+1):
    #         self._decay_effects[:, k] = tensor.groupby('event').apply(
    #             lambda i: ((((1 - np.exp(
    #                 -self._delta * (self._max_s_t - i['timestamp']))) / self._delta)
    #                         * i['times']) * i['node'].apply(
    #                 lambda j: len(self._k_hop_neibors(j, k)))).sum())
    #         # print(tensor.groupby('event').apply(lambda i: ( i['node'].apply(
    #         #         lambda j: len(self._k_hop_neibors(j, k)))).sum()))
    #     # |V|x|T|
    #     self._T = (self._max_s_t - self._min_s_t) * len(tensor['node'].unique())

    def _start_init(self, tensor):
        """
        Generates some required initial values.
        """
        # 读取数据清洗过后的数据集
        # tensor = pd.read_csv("../datasets/with_topology/2/dataclean-2-Alarm.csv")
        tensor['timestamp'] = tensor['timestamp'].astype(float)
        tensor['timestamp2'] = tensor['timestamp2'].astype(float)
        print(tensor)

        # 源码统计了次数times
        # tensor = tensor.groupby(
        #     ['event', 'timestamp', 'timestamp2', 'node']).apply(len).reset_index()
        # tensor.columns = ['event', 'timestamp', 'timestamp2', 'node', 'times']
        # tensor = tensor.reindex(columns=['node', 'timestamp', 'timestamp2', 'event', 'times'])

        tensor = tensor.sort_values(['node', 'timestamp'])
        tensor = tensor.drop(columns='timestamp2')		# THP暂时没有用到end_stamp，先丢弃
        self.tensor = tensor[tensor['node'].isin(self._topo.nodes)]
        print(f"删去不在设备拓扑图中的结点的数据{len(self.tensor)-len(tensor)}条")

        # calculate considered events
        self._event_names = np.array(list(set(self.tensor['event'])))
        self._event_names.sort()
        self._N = len(self._event_names)
        self._matrix_names = list(self._event_names.astype(str))

        # map event name to corresponding index value
        self._event_indexes = self._map_event_to_index(
            self.tensor['event'].values, self._event_names)
        self.tensor['event'] = self._event_indexes

        self._g = self._topo.subgraph(self.tensor['node'].unique())
        self._ne_grouped = self.tensor.groupby('node')

        self._decay_effects = np.zeros(
            [len(self._event_names), self._max_hop + 1])  # will be used in EM.

        self._max_s_t = tensor['timestamp'].max()
        self._min_s_t = tensor['timestamp'].min()
        print("emmmmmmmmm数据清洗后执行变慢了")
        for k in range(self._max_hop + 1):
            self._decay_effects[:, k] = tensor.groupby('event').apply(
                lambda i: ((((1 - np.exp(
                    -self._delta * (self._max_s_t - i['timestamp']))) / self._delta)
                            * i['times']) * i['node'].apply(
                    lambda j: len(self._k_hop_neibors(j, k)))).sum())
            # print(tensor.groupby('event').apply(lambda i: ( i['node'].apply(
            #         lambda j: len(self._k_hop_neibors(j, k)))).sum()))
        print("emmmmmmmmm数据清洗后执行变慢了")
        print(self._decay_effects)
        # |V|x|T|
        self._T = (self._max_s_t - self._min_s_t) * len(tensor['node'].unique())

    def _k_hop_neibors(self, node, k):

        if k == 0:
            return {node}
        else:
            return set(nx.single_source_dijkstra_path_length(
                self._g, node, k).keys()) - set(
                nx.single_source_dijkstra_path_length(
                    self._g, node, k - 1).keys())

    @staticmethod
    def _map_event_to_index(event_names, base_event_names):
        """
        Maps the event name to the corresponding index value.

        Parameters
        ----------
        event_names: np.ndarray, shape like (52622,)
            All occurred event names sorted by node and timestamp.
        base_event_names: np.ndarray, shape like (10,)
            All deduplicated and sorted event names

        Returns
        -------
        np.ndarray: All occurred event names mapped to their corresponding index 
         in base_event_names.
        """
        return np.array(list(map(lambda event_name:
                                 np.where(base_event_names == event_name)[0][0],
                                 event_names)))


    def _hill_climb(self):
        """
        Search the best causal graph, then generate the causal matrix (DAG).

        Returns
        -------
        result: tuple, (likelihood, alpha matrix, events vector)
            likelihood: used as the score criteria for searching the
                causal structure.
            alpha matrix: the intensity of causal effect from event v’ to v.
            events vector: the exogenous base intensity of each event.
        edge_mat: np.ndarray
            Causal matrix.
        """

        self._get_effect_tensor_decays()

        # Initialize the adjacency matrix
        edge_mat = np.eye(self._N, self._N)     # 虽然从结果来看对角线上01没有意义，但初始化必须为1，否则更新不动参数

        # 这里egde_mat可以加载已经训练了的npy继续训练
        # edge_mat = np.load("../output/est_graphs/20-3.npy")

        result = self._em(edge_mat)
        l_ret = result[0]

        maxl_edge_mat_list = []     # oyxy加的
        x = []
        y = []
        for num_iter in range(self._max_iter):
            # print(self._max_iter)
            logging.info('[iter {}]: likelihood_score = {}'.format(num_iter, l_ret))

            # oyxy加的2
            g_score = MetricsDAG(edge_mat, self.true_dag_matrix).metrics["gscore"]
            logging.info(f"g-score: {g_score}")
            x.append(num_iter)
            y.append(g_score)
            plt.clf()  # 清除之前画的图
            plt.plot(x, y, "r-o")  # 画出当前x列表和y列表中的值的图形
            plt.pause(0.05)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.ioff()  # 关闭画图窗口

            if num_iter!=0 and num_iter%10==0:
                np.save(f"../output/est_graphs/{num_iter}-2-data_clean.npy", edge_mat)

            stop_tag = True
            edge_mat_last_iter = edge_mat.copy()
            for new_edge_mat in tqdm(list(
                    self._one_step_change_iterator(edge_mat))):

                new_result = self._em(new_edge_mat)
                new_l = new_result[0]
                # Termination condition:
                #   no adjacency matrix with higher likelihood appears
                # print("#############")
                # print(f"{num_iter} {new_l} {l_ret} {stop_tag}")
                if new_l > l_ret:
                    # print("?????????????????")
                    if edge_mat_last_iter.sum() >= new_edge_mat.sum():    # oyxy加的3，研究为什么g分数下降
                        MyGraphDAG(num_iter, edge_mat_last_iter, new_edge_mat)
                    result = new_result
                    l_ret = new_l
                    stop_tag = False
                    edge_mat = new_edge_mat

            MyGraphDAG(num_iter, edge_mat)
            edge_mat_last_iter = edge_mat.copy()

            # maxl_edge_mat_list.append(edge_mat.copy())   # oyxy加的

            if stop_tag:
                break

        # self._plot_dag_change(maxl_edge_mat_list)  # oyxy加的
        plt.show()

        return result, edge_mat

    @staticmethod
    def _plot_dag_change(dag_list):
        """
        oyxy加的，打印出因果图的变化
        """
        fig, ax = plt.subplots()

        ims = []
        first = 1
        for dag in dag_list:
            # trans diagonal element into 0
            for i in range(len(dag)):
                if dag[i][i] == 1:
                    dag[i][i] = 0
            im = ax.imshow(dag, cmap='Greys', interpolation='none')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
        # ani.save("search_movie.mp4")
        plt.show()

    def _get_effect_tensor_decays(self):

        self._effect_tensor_decays = np.zeros([self._max_hop+1,
                                               len(self.tensor),
                                               len(self._event_names)])
        for k in range(self._max_hop+1):
            self._get_effect_tensor_decays_each_hop(k)

    def _get_effect_tensor_decays_each_hop(self, k):

        j = 0
        pre_effect = np.zeros(self._N)
        tensor_array = self.tensor.values
        for item_ind in range(len(self.tensor)):
            sub_n, start_t, ala_i, times = tensor_array[
                item_ind, [0, 1, 2, 3]]
            last_sub_n, last_start_t, last_ala_i, last_times = \
                tensor_array[item_ind - 1, [0, 1, 2, 3]]
            if (last_sub_n != sub_n) or (last_start_t > start_t):
                j = 0
                pre_effect = np.zeros(self._N)
                try:
                    k_hop_neighbors_ne = self._k_hop_neibors(sub_n, k)
                    neighbors_table = pd.concat(
                        [self._ne_grouped.get_group(i)
                         for i in k_hop_neighbors_ne])
                    neighbors_table = neighbors_table.sort_values(
                        'timestamp')
                    neighbors_table_value = neighbors_table.values
                except ValueError as e:
                    k_hop_neighbors_ne = []

                if len(k_hop_neighbors_ne) == 0:
                    continue

            cur_effect = pre_effect * np.exp(
                (np.min((last_start_t - start_t, 0))) * self._delta)
            while 1:
                try:
                    nei_sub_n, nei_start_t, nei_ala_i, nei_times \
                        = neighbors_table_value[j, :]
                except:
                    break
                if nei_start_t < start_t:
                    cur_effect[int(nei_ala_i)] += nei_times * np.exp(
                        (nei_start_t - start_t) * self._delta)
                    j += 1
                else:
                    break
            pre_effect = cur_effect

            self._effect_tensor_decays[k, item_ind] = pre_effect

    def _em(self, edge_mat):
        """
        E-M module, used to find the optimal parameters.

        Parameters
        ----------
        edge_mat： np.ndarray
            Adjacency matrix.

        Returns
        -------
        likelihood: used as the score criteria for searching the
            causal structure.
        alpha matrix: the intensity of causal effect from event v’ to v.
        events vector: the exogenous base intensity of each event.
        """

        causal_g = nx.from_numpy_matrix((edge_mat - np.eye(self._N, self._N)),
                                        create_using=nx.DiGraph)

        if not nx.is_directed_acyclic_graph(causal_g):
            return -100000000000000, \
                   np.zeros([len(self._event_names), len(self._event_names)]), \
                   np.zeros(len(self._event_names))

        # Initialize alpha:(nxn)，mu:(nx1) and L
        alpha = np.ones([self._max_hop+1, len(self._event_names),
                         len(self._event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self._event_names))
        l_init = 0

        for i in range(len(self._event_names)):
            pa_i = set(np.where(edge_mat[:, i] == 1)[0])
            li = -100000000000
            ind = np.where(self._event_indexes == i)
            x_i = self.tensor['times'].values[ind]
            x_i_all = np.zeros_like(self.tensor['times'].values)
            x_i_all[ind] = x_i
            while 1:
                # Calculate the first part of the likelihood
                lambda_i_sum = (self._decay_effects
                                * alpha[:, :, i].T).sum() + mu[i] * self._T

                # Calculate the second part of the likelihood
                lambda_for_i = np.zeros(len(self.tensor)) + mu[i]
                for k in range(self._max_hop+1):
                    lambda_for_i += np.matmul(
                        self._effect_tensor_decays[k, :],
                        alpha[k, :, i].T)
                lambda_for_i = lambda_for_i[ind]
                x_log_lambda = (x_i * np.log(lambda_for_i)).sum()
                new_li = -lambda_i_sum + x_log_lambda

                # Iteration termination condition
                delta = new_li - li
                if delta < 0.1:
                    li = new_li
                    l_init += li
                    pa_i_alpha = dict()
                    for j in pa_i:
                        pa_i_alpha[j] = alpha[:, j, i]
                    break
                li = new_li
                # update mu
                mu[i] = ((mu[i] / lambda_for_i) * x_i).sum() / self._T
                # update alpha
                for j in pa_i:
                    for k in range(self._max_hop+1):
                        upper = ((alpha[k, j, i] * (
                            self._effect_tensor_decays[k, :, j])[ind]
                                  / lambda_for_i) * x_i).sum()
                        lower = self._decay_effects[j, k]
                        if lower == 0:
                            alpha[k, j, i] = 0
                            continue
                        alpha[k, j, i] = upper / lower
            i += 1

        if self._penalty == 'AIC':
            return l_init - (len(self._event_names)
                             + self._epsilon * edge_mat.sum()
                             * (self._max_hop+1)), alpha, mu
        elif self._penalty == 'BIC':
            return l_init - (len(self._event_names)
                             + self._epsilon * edge_mat.sum()
                             * (self._max_hop+1)) * np.log(
                self.tensor['times'].sum()) / 2, alpha, mu
        else:
            raise ValueError("The penalty's value should be BIC or AIC.")

    def _one_step_change_iterator(self, edge_mat):

        return map(lambda e: self._one_step_change(edge_mat, e),
                   product(range(len(self._event_names)),
                           range(len(self._event_names))))
                    # product((0,1,2,3..,N),(0,1,2,3..,N))
                    # (0,0)(0,1)(0,2)...(0,N)
                    # (1,0)...
                    # .

    @staticmethod
    def _one_step_change(edge_mat, e):
        """
        Changes the edge value in the edge_mat.

        Parameters
        ----------
        edge_mat: np.ndarray
            Adjacency matrix.
        e: tuple_like (j,i)

        Returns
        -------
        new_edge_mat: np.ndarray
            new value of edge
        """
        j, i = e
        if j == i:
            return edge_mat
        new_edge_mat = edge_mat.copy()

        if new_edge_mat[j, i] == 1:
            new_edge_mat[j, i] = 0
            return new_edge_mat
        else:
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0
            return new_edge_mat


import numpy as np
import matplotlib.pyplot as plt


class MyGraphDAG(object):
    '''
    Visualization for causal discovery learning results.

    Parameters
    ----------
    est_dag: np.ndarray
        The DAG matrix to be estimated.
    true_dag: np.ndarray
        The true DAG matrix.
    show: bool
        Select whether to display pictures.
    save_name: str
        The file name of the image to be saved.
    '''

    def __init__(self, iters, est_dag, true_dag=None, show=True, save_name=None):

        self.iters = iters
        # self.est_dag = est_dag.copy()
        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name

        if not isinstance(est_dag, np.ndarray):
            raise TypeError("Input est_dag is not numpy.ndarray!")

        if true_dag is not None and not isinstance(true_dag, np.ndarray):
            raise TypeError("Input true_dag is not numpy.ndarray!")

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        MyGraphDAG._plot_dag(self.iters, self.est_dag, self.true_dag, self.show, self.save_name)

    @staticmethod
    def _plot_dag(iters, est_dag, true_dag, show=True, save_name=None):
        """
        Plot the estimated DAG and the true DAG.

        Parameters
        ----------
        est_dag: np.ndarray
            The DAG matrix to be estimated.
        true_dag: np.ndarray
            The True DAG matrix.
        show: bool
            Select whether to display pictures.
        save_name: str
            The file name of the image to be saved.
        """

        if isinstance(true_dag, np.ndarray):
            est_dag = np.copy(est_dag)
            true_dag = np.copy(true_dag)

            # trans diagonal element into 0
            for i in range(len(true_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0
                if true_dag[i][i] == 1:
                    true_dag[i][i] = 0

            # set plot size
            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            ax1.set_title('edge_mat')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            ax2.set_title('new_edge_mat')
            map2 = ax2.imshow(true_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map2, ax=ax2)

            ax1.text(0.1, 0.7, f"from iter {iters} to {iters+1}")

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        elif isinstance(est_dag, np.ndarray):

            est_dag = np.copy(est_dag)

            # trans diagonal element into 0
            for i in range(len(est_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0

            # set plot size
            fig, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

            ax1.set_title('est_graph_every_iter')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            ax1.text(0.1, 0.7, f"from iter {iters} to {iters + 1}")
            fig.colorbar(map1, ax=ax1)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()
