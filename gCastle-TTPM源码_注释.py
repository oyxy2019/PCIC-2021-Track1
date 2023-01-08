# coding=utf-8

import logging
import pandas as pd
import numpy as np
import networkx as nx
from itertools import product

from castle.common import BaseLearner, Tensor


class TTPM(BaseLearner):
    """
    TTPM Algorithm.
        A causal structure learning algorithm based on Topological Hawkes process
        for spatio-temporal event sequences.
        一种基于拓扑霍克斯过程的因果结构学习算法，用于时空事件序列
    Parameters
    ----------
    topology_matrix: np.Matrix
        Interpreted as an adjacency matrix to generate the graph.
        It should have two dimensions, and should be square.
        网络拓扑的二进制对称邻接矩阵
        它应该是二维的，而且应该是方阵。
    delta: float, default=0.1
        Time decaying coefficient for the exponential kernel.
        指数核的时间衰减系数
    epsilon: int, default=1
        BIC penalty coefficient.
        BIC惩罚系数
    max_hop: positive int, default=6
        The maximum considered hops in the topology,
        when ``max_hop=0``, it is divided by nodes, regardless of topology.
        拓扑中考虑的最大跳数，
        当``max_hop=0``时，它由节点划分，而不考虑拓扑结构。
    penalty: str, default=BIC
        Two optional values: 'BIC' or 'AIC'.
        两种可选的惩罚方式: 'BIC' or 'AIC'
    max_iter: int
        Maximum number of iterations.
        迭代次数
    """

    def __init__(self, topology_matrix, delta=0.1, epsilon=1,
                 max_hop=0, penalty='BIC', max_iter=20):
        BaseLearner.__init__(self)

        # 加载拓扑图
        assert isinstance(topology_matrix, np.ndarray), \
            'topology_matrix should be np.matrix object'  # 断言topology_matrix应该是一个np.matrix类型
        assert topology_matrix.ndim == 2, \
            'topology_matrix should be two dimension'  # 断言topology_matrix的维度为2
        assert topology_matrix.shape[0] == topology_matrix.shape[1], \
            'The topology_matrix should be square.'  # 断言topology_matrix为一个方阵
        self._topo = nx.from_numpy_matrix(topology_matrix, create_using=nx.Graph)

        # 初始化实例变量
        self._penalty = penalty
        self._delta = delta
        self._max_hop = max_hop
        self._epsilon = epsilon
        self._max_iter = max_iter

    def learn(self, tensor, *args, **kwargs):
        """
        Set up and run the TTPM algorithm.
        设置并运行TTPM算法
        Parameters
        ----------
        tensor:  pandas.DataFrame
            tensor应该包含三个cols:
                ['event', 'timestamp', 'node']
            描述如下:
                event: 事件类型
                timestamp: 事件发生的时间戳，例如'1615962101.0'
                node: 事件发生的拓扑节点
        """

        # 数据类型检查
        if not isinstance(tensor, pd.DataFrame):  # 检查tensor类型是否为DataFrame
            raise TypeError('The tensor type is not correct,'
                            'only receive pd.DataFrame type currently.')

        cols_list = ['event', 'timestamp', 'node']  # 检查列名在tensor中是否存在
        for col in cols_list:
            if col not in tensor.columns:
                raise ValueError("The data tensor should contain column with name {}".format(col))

        # 变量初始化，根据tensor
        self._start_init(tensor)

        # 生成因果图 (DAG)
        _, raw_causal_matrix = self._hill_climb()  # 爬山法
        self._causal_matrix = Tensor(raw_causal_matrix,
                                     index=self._matrix_names,
                                     columns=self._matrix_names)  # 转为Tensor(numpy)类型

    def _start_init(self, tensor):
        """
        Generates some required initial values.
        生成一些所需的初始值。
        """
        tensor.dropna(axis=0, how='any', inplace=True)  # 删去有缺失值的那些行
        tensor['timestamp'] = tensor['timestamp'].astype(float)  # timestamp转为float类型

        # 增加一列，出现相同行的次数，'times'
        tensor = tensor.groupby(
            ['event', 'timestamp', 'node']).apply(len).reset_index()
        tensor.columns = ['event', 'timestamp', 'node', 'times']
        tensor = tensor.reindex(columns=['node', 'timestamp', 'event', 'times'])

        # 排序后，只保留拓扑图中出现的节点，得到最终的self.tensor
        # （问题：对于没有拓扑图的数据集，拓扑矩阵应该是一个单位阵还是零矩阵？）
        tensor = tensor.sort_values(['node', 'timestamp'])
        self.tensor = tensor[tensor['node'].isin(self._topo.nodes)]

        # calculate considered events
        # 得到_event_names并排序，用于最终的因果图邻接矩阵的行和列
        self._event_names = np.array(list(set(self.tensor['event'])))
        self._event_names.sort()
        self._N = len(self._event_names)
        self._matrix_names = list(self._event_names.astype(str))

        # map event name to corresponding index value
        # 调用_map_event_to_index静态方法，将_event_names映射到相应的索引值，并取代
        self._event_indexes = self._map_event_to_index(
            self.tensor['event'].values, self._event_names)
        self.tensor['event'] = self._event_indexes

        # 拓扑图的子图，只包含'node'及其边，用于_k_hop_neibors方法
        self._g = self._topo.subgraph(self.tensor['node'].unique())

        # ？？，用于，应该也是用来计算拓扑邻居
        self._ne_grouped = self.tensor.groupby('node')

        # ？？，EM算法用到
        self._decay_effects = np.zeros([len(self._event_names), self._max_hop + 1])  # will be used in EM.

        # 最大时间戳，最小时间戳
        self._max_s_t = tensor['timestamp'].max()
        self._min_s_t = tensor['timestamp'].min()

        # 求self._decay_effects，过程看不懂
        for k in range(self._max_hop + 1):
            self._decay_effects[:, k] = tensor.groupby('event').apply(
                lambda i: (
                        (((1 - np.exp(-self._delta * (self._max_s_t - i['timestamp']))) / self._delta) * i['times'])
                        * i['node'].apply(lambda j: len(self._k_hop_neibors(j, k)))
                ).sum()
            )

        # |V|x|T|
        # 等于node数量×时间宽度，EM算法用到
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
        爬山法搜索最优因果图
        Returns：返回值
        -------
        result: tuple, (likelihood, alpha matrix, events vector)
            likelihood: used as the score criteria for searching the causal structure.
                用于搜索因果结构的得分标准。
            alpha matrix: the intensity of causal effect from event v’ to v.
                从事件v'到事件v的因果效应的强度。
            events vector: the exogenous base intensity of each event.
                每个事件的外部基础强度。
        edge_mat: np.ndarray
            Causal matrix.
                因果图矩阵
        """
        self._get_effect_tensor_decays()

        # Initialize the adjacency matrix
        # 初始化邻接矩阵edge_mat，和初始likelihood_score
        edge_mat = np.eye(self._N, self._N)
        result = self._em(edge_mat)
        l_ret = result[0]

        for num_iter in range(self._max_iter):

            logging.info('[iter {}]: likelihood_score = {}'.format(num_iter, l_ret))

            stop_tag = True
            # 对当前的因果图作出一步改变，得到一些新因果图
            for new_edge_mat in list(self._one_step_change_iterator(edge_mat)):
                new_result = self._em(new_edge_mat)  # EM算法，最大化似然函数
                new_l = new_result[0]  # 得到新图的似然
                # 循环结束条件：新似然不再变大
                # no adjacency matrix with higher likelihood appears
                if new_l > l_ret:
                    result = new_result
                    l_ret = new_l
                    stop_tag = False
                    edge_mat = new_edge_mat  # 更新因果图

            if stop_tag:
                return result, edge_mat

        return result, edge_mat

    def _get_effect_tensor_decays(self):

        self._effect_tensor_decays = np.zeros([self._max_hop + 1,
                                               len(self.tensor),
                                               len(self._event_names)])
        for k in range(self._max_hop + 1):
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
        E-M算法，用于找到最优参数
        Parameters
        ----------
        edge_mat： np.ndarray
            Adjacency matrix.
        Returns
        -------
        likelihood: used as the score criteria for searching the causal structure.
        alpha matrix: the intensity of causal effect from event v’ to v.
        events vector: the exogenous base intensity of each event.
        """

        causal_g = nx.from_numpy_matrix((edge_mat - np.eye(self._N, self._N)),
                                        create_using=nx.DiGraph)

        # 因果图必须是一个DAG（有向无环图）
        if not nx.is_directed_acyclic_graph(causal_g):
            return -100000000000000, \
                   np.zeros([len(self._event_names), len(self._event_names)]), \
                   np.zeros(len(self._event_names))

        # 初始化：
        # alpha:(nxn)，mu:(nx1) and L
        alpha = np.ones([self._max_hop + 1, len(self._event_names), len(self._event_names)])
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
                # 计算公式9的第一部分
                lambda_i_sum = (self._decay_effects
                                * alpha[:, :, i].T).sum() + mu[i] * self._T

                # Calculate the second part of the likelihood
                # 计算公式9的第二部分
                lambda_for_i = np.zeros(len(self.tensor)) + mu[i]
                for k in range(self._max_hop + 1):
                    lambda_for_i += np.matmul(
                        self._effect_tensor_decays[k, :],
                        alpha[k, :, i].T)
                lambda_for_i = lambda_for_i[ind]
                x_log_lambda = (x_i * np.log(lambda_for_i)).sum()

                # 公式9第一部分+第二部分
                new_li = -lambda_i_sum + x_log_lambda

                # Iteration termination condition
                # while(1)循环停止条件：当L与上一次L的差小于0.1，则认为收敛
                delta = new_li - li
                if delta < 0.1:
                    li = new_li
                    l_init += li
                    pa_i_alpha = dict()
                    for j in pa_i:
                        pa_i_alpha[j] = alpha[:, j, i]
                    break

                # update L
                li = new_li
                # update mu，公式11μ
                mu[i] = ((mu[i] / lambda_for_i) * x_i).sum() / self._T
                # update alpha，公式11α
                for j in pa_i:
                    for k in range(self._max_hop + 1):
                        upper = ((alpha[k, j, i] * (
                            self._effect_tensor_decays[k, :, j])[ind]
                                  / lambda_for_i) * x_i).sum()
                        lower = self._decay_effects[j, k]
                        if lower == 0:
                            alpha[k, j, i] = 0
                            continue
                        alpha[k, j, i] = upper / lower
            i += 1  # ？这里没太看懂

        # 加上BIC惩罚，公式10，返回三个值
        if self._penalty == 'AIC':
            return l_init - (len(self._event_names) + self._epsilon * edge_mat.sum() * (self._max_hop + 1)), \
                   alpha, mu
        elif self._penalty == 'BIC':
            return l_init - (len(self._event_names) + self._epsilon * edge_mat.sum() * (self._max_hop + 1)
                             ) * np.log(self.tensor['times'].sum()) / 2, \
                   alpha, mu
        else:
            raise ValueError("The penalty's value should be BIC or AIC.")

    def _one_step_change_iterator(self, edge_mat):
        # 返回一系列的新图，类型为numpy矩阵
        return map(
            lambda e: self._one_step_change(edge_mat, e),
            product(range(len(self._event_names)), range(len(self._event_names)))
        )

    @staticmethod
    def _one_step_change(edge_mat, e):
        """
        Changes the edge value in the edge_mat.
        改变 edge_mat 其中一条边
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

        # j和i相等，返回原图
        # ？？那这样对角线上的1不是一直都得不到更新吗
        # 更新：确实得不到更新，GraphDAG方法画图的时候直接令对角线上的元素为0了，对角线上是1或0也没有意义
        if j == i:
            return edge_mat

        new_edge_mat = edge_mat.copy()

        # 修改（j, i）这条边：如果原来是相连的就修改为断开，反之相连
        if new_edge_mat[j, i] == 1:
            new_edge_mat[j, i] = 0
            return new_edge_mat
        else:
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0  # 因果图中，两个结点（事件类型）不可能同时存在两条有向边：一个是因一个是果
            return new_edge_mat
