import pandas as pd


def dataclean(data, dataset_path):
    print("\n!!!!!!!!!!!!数据清洗!!!!!!!!!!!!!!!")
    data.dropna(axis=0, how='any', inplace=True)  # 删去有缺失值的那些行
    # 排序
    data = data.sort_values(['event', 'node', 'timestamp', 'timestamp2'], ignore_index=True)
    print("before:\n", data)
    events = list(data['event'])
    nodes = list(data['node'])
    t1s = list(data['timestamp'])
    t2s = list(data['timestamp2'])
    drop_idx = []
    for i in range(len(events) - 1):  # 遍历每一行
        event = events[i]
        node = nodes[i]
        t1 = t1s[i]
        t2 = t2s[i]
        next_event = events[i + 1]
        next_node = nodes[i + 1]
        next_t1 = t1s[i + 1]
        next_t2 = t2s[i + 1]
        # if t1 == t2:  # 一条数据前后时间戳相等的, 注意for不要减1
        #     drop_idx.append(i)
        #     continue
        if event == next_event and node == next_node:
            if t1 == next_t1 and t2 == next_t2:  # 两条数据完全重叠的
                drop_idx.append(i + 1)
            elif t1 < next_t1 < next_t2 < t2:  # 两条数据, 一条包围另一条的
                drop_idx.append(i + 1)
                data.loc[i + 1, :] = data.loc[i, :]  # 这里由于i+1滚动，为了删到非相邻的数据，所以要把第二行赋值为第一行
                # print("$$$$$$$$$$$$$$$$$")
                # print(i)
                # print(data.iloc[i:i+2, :])
            elif t1 < next_t1 <= t2 < next_t2:  # 两条数据有交集的
                # t2 <= next_t2
                # print("$$$$$$$$$$$$$$$$$")
                # print(i)
                # print(data.iloc[i:i+2, :])
                data.loc[i, 'timestamp2'] = next_t2
                data.loc[i + 1, :] = data.loc[i, :]
                # print(data.iloc[i:i+2, :])
                drop_idx.append(i + 1)

    data.drop(drop_idx, inplace=True)
    # print("after_drop:\n", data)

    # # THP源码求了times
    # data = data.groupby(
    #     ['event', 'timestamp', 'timestamp2', 'node']).apply(len).reset_index()
    # data.columns = ['event', 'timestamp', 'timestamp2', 'node', 'times']
    # data = data.reindex(columns=['node', 'timestamp', 'event', 'times', 'timestamp2'])
    # data = data.sort_values(['event', 'node', 'timestamp', 'timestamp2'], ignore_index=True)

    print("\n!!!!!!!!!!!!数据清洗完毕!!!!!!!!!!!!!!!")
    print("new_data:\n", data)
    data.to_csv(f"{dataset_path}/dataclean.csv")
    return data


if __name__ == '__main__':
    # 设置数据集路径
    dataset_name = "25V_474N_Microwave"
    dataset_path = f"../datasets/alarm/{dataset_name}"

    # 一个数据集中三个文件的路径
    alarm_path = f"{dataset_path}/Alarm.csv"
    # topo_path = f"{dataset_path}/Topology.npy"
    # dag_path = f"{dataset_path}/DAG.npy"

    alarm_data = pd.read_csv(alarm_path, encoding="utf")

    # 设置列名
    X = alarm_data.iloc[:, 0:4]
    X.columns = ["event", "node", "timestamp", "timestamp2"]
    # X = X.reindex(columns=["event", "timestamp", "timestamp2", "node"])

    X = dataclean(X, dataset_path)
