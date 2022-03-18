r"""
Module used to assist in understanding the training process.
"""
from tools.utils import Timer, Accumulator
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
from typing import List, Tuple
from typing import Optional as Op


class Animator:
    r"""
    在动画中绘制数据，用于实时显示训练过程，lengths, xlabels卡的比较严格，必须要输。
    Example: 
    """

    def __init__(self, legends: List[Tuple],  # 必传参数
                 nrows, ncols, figsize=(8, 6),
                 xlabels: Op[List[Op[str]]] = None, ylabels: Op[List[Op[str]]] = None,
                 xlims: Op[List[Op[Tuple]]] = None, ylims: Op[List[Op[Tuple]]] = None,
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


num_epochs = 5
num_batches = 20
batch_size = 32
timer = Timer()
animator = Animator(legends=[('train loss',), ('train acc', 'valid acc')],
                    nrows=1, ncols=2, figsize=(16, 8),
                    xlabels=['epoch' for _ in range(2)],
                    ylabels=['loss', 'acc'],
                    xlims=[(0, num_epochs) for _ in range(2)],
                    ylims=[None, (0, 1)])
recorder = Accumulator(3)
l, acc = 10, 0
for epoch in range(num_epochs):
    # 存储3个指标：sum of train loss, acc, number of examples
    recorder.reset()
    for i in range(num_batches):
        timer.start()
        time.sleep(0.001)
        l, acc = np.random.rand() * num_batches / (i+1), batch_size * \
            (np.random.rand() * 0.5 + 0.5) * i / num_batches
        recorder.add(l, acc, batch_size)
        timer.stop()
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            animator.add(1, 1, epoch + (i + 1) / num_batches,
                         (recorder[0] / recorder[2]))
            animator.add(1, 2, epoch + (i + 1) / num_batches,
                         (recorder[1] / recorder[2], None))
    valid_acc = recorder[1] / recorder[2] - 0.05
    animator.add(1, 2, epoch + 1, (None, valid_acc))
    if epoch == num_epochs - 1:  # 统计一下最终训练效果
        print(f'loss {recorder[0] / recorder[2]:.3f}, final train acc '
              f'{recorder[1] / recorder[2]:.3f}, final valid acc {valid_acc:.3f}')
        print(
            f'{recorder[2] * num_epochs / timer.sum():.1f} examples/sec on device')
