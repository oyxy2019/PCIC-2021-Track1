## THP代码复现

环境安装：(python3.8)
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

原Readme文件如下：

## PCIC 2021 Track1: Causal Discovery

AIOps 相关：多告警间因果关系图学习，并辅助定位根因。

竞赛地址：[https://competition.huaweicloud.com/information/1000041487/introduction](https://competition.huaweicloud.com/information/1000041487/introduction)

赛题解读：[PCIC 2021 | 华为 & 北京大学因果推理挑战赛](https://dreamhomes.top/posts/202106211024.html)

---

Baselines:

- TTPM (with topology)
  - Cite: [THP: Topological Hawkes Processes for Learning Granger Causality on Event Sequences](https://arxiv.org/abs/2105.10884)
  - Code: [https://nbviewer.jupyter.org/github/dreamhomes/PCIC-2021-Track1/blob/master/notebooks/baseline.ipynb](https://nbviewer.jupyter.org/github/dreamhomes/PCIC-2021-Track1/blob/master/notebooks/TTPM.ipynb)


