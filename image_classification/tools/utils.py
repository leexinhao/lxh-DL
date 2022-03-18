r"""
Module used to assist in understanding the training process.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
from typing import List
from typing import Optional as Op




class Accumulator: 
    """在`n`个变量上累加。"""
    def __init__(self, n, names:Op[List[str]]=None):
        self.data = [0.0] * n
        if names is not None:
            assert len(names) == n, f"len(names):{len(names)} must equal to n:{n}"
            self.name_dict = {k:v for v, k in enumerate(names)}

    def add(self, *args):
        r"""
        support accumlator.add(v1,..., vn) or accumlator.add({k1:v1,..., kn:vn})
        """
        if type(args[0]) is dict:
            assert len(args) == 1, "You can only add one dict at a time"
            for k, v in args[0].items():
                self.data[self.name_dict[k]] += v
        else:
            self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __str__(self):
        return ", ".join([f"{k}({v}):{self.data[v]}" for k, v in self.name_dict.items()])
        
    def __getitem__(self, idx):
        if type(idx) is int:
            return self.data[idx]
        else:
            return self.data[self.name_dict[idx]]

    def __setitem__(self, idx, value):
        if type(idx) is int:
            self.data[idx] = value
        else:
            self.data[self.name_dict[idx]] = value

class Timer: 
    """计时器，用于记录多次运行时间，以秒为单位。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        if len(self.times) == 0:
            return -1
        else:
            return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

class Animator:
    r"""
    在动画中绘制数据，用于实时显示训练过程
    Example: 
    animator = Animator(legends=[('train loss',), ('train acc', 'valid acc')],
                    nrows=1, ncols=2, figsize=(16, 8),
                    xlabels=['epoch' for _ in range(2)],
                    ylabels=['loss', 'acc'],
                    xlims=[(0, num_epochs) for _ in range(2)],
                    ylims=[None, (0, 1)])
    """

    def __init__(self, legends: List[List[str]],  # 必传参数
                 nrows, ncols, figsize=(12, 8),
                 xlabels: Op[List[Op[str]]] = None, ylabels: Op[List[Op[str]]] = None,
                 xlims: Op[List[Op[List]]] = None, ylims: Op[List[Op[List]]] = None,
                 xscales: Op[List[Op[str]]] = None, yscales: Op[List[Op[str]]] = None,
                 fontsize=16
                 ):

        self.n = nrows * ncols
        self.legends = legends
        self.fontsize = fontsize
        self.Xs = [None for _ in range(self.n)]
        self.Ys = [None for _ in range(self.n)]
        self.fmts = ('-', 'm--', 'g-.', 'r:')  # TODO 线条样式先写死
        if xlabels is None:
            self.xlabels = [None for _ in range(self.n)]
        else:
            self.xlabels = xlabels
        if ylabels is None:
            self.ylabels = [None for _ in range(self.n)]
        else:
            self.ylabels = ylabels
        if xlims is None:
            self.xlims = [None for _ in range(self.n)]
        else:
            self.xlims = xlims
        if ylims is None:
            self.ylims = [None for _ in range(self.n)]
        else:
            self.ylims = ylims
        if xscales is None:
            self.xscales = [None for _ in range(self.n)]
        else:
            self.xscales = xscales
        if yscales is None:
            self.yscales = [None for _ in range(self.n)]
        else:
            self.yscales = yscales

        assert len(self.legends) == len(self.xlabels) == len(self.ylabels) == len(self.xlims) == len(
            self.ylims) == len(self.xscales) == len(self.yscales) == self.n, f"Wrong parameters were received"

        self.use_svg_display()
        self.nrows, self.ncols = nrows, ncols

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        self.axes = self.axes.reshape((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                self.set_ax(self.axes[i, j], i*ncols+j)

    def add(self, row, col, x, y):
        r"""
        向第(row, col)个图表中添加多个数据点, 从而增量地绘制多条线
        """
        row, col = row - 1, col - 1  # 从0开始数
        idx = row * self.ncols + col
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)  # 曲线数量
        assert n <= 4, "The number of curve in a plot cannot exceed 4"
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.Xs[idx] is None:
            self.Xs[idx] = [[] for _ in range(n)]
        if self.Ys[idx] is None:
            self.Ys[idx] = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.Xs[idx][i].append(a)
                self.Ys[idx][i].append(b)

        self.axes[row, col].cla()
        for x, y, fmt in zip(self.Xs[idx], self.Ys[idx], self.fmts):
            self.axes[row, col].plot(x, y, fmt)
        self.set_ax(self.axes[row, col], idx)  # 得放plot后面不然legend显示不出来

        display.display(self.fig)
        display.clear_output(wait=True)

    def use_svg_display(self):
        r"""使⽤svg格式在Jupyter中显⽰绘图。"""
        display.set_matplotlib_formats('svg')

    def set_ax(self, ax, idx):
        """设置matplotlib的轴。"""
        if self.xlabels[idx] is not None:
            ax.set_xlabel(self.xlabels[idx], fontsize=self.fontsize)
        if self.ylabels[idx] is not None:
            ax.set_ylabel(self.ylabels[idx], fontsize=self.fontsize)
        if self.xscales[idx] is not None:
            ax.set_xscale(self.xscales[idx], fontsize=self.fontsize)
        if self.yscales[idx] is not None:
            ax.set_yscale(self.yscales[idx], fontsize=self.fontsize)
        if self.xlims[idx] is not None:
            ax.set_xlim(self.xlims[idx])
        if self.ylims[idx] is not None:
            ax.set_ylim(self.ylims[idx])
        if self.legends[idx] is not None:
            ax.legend(self.legends[idx], fontsize=self.fontsize)
        ax.grid()
        ax.tick_params(top=False, bottom=True, left=True,
                       right=False, labelsize=self.fontsize-1)

    def set_suptitle(self, title):
        self.fig.suptitle(title, fontsize=self.fontsize-1)
