
r"""
metric必须实现method='sum', 'mean'方法，最好返回float #TODO 草，有没有考虑过pytorch有封装过metric，算了，万一以后要自定义metric呢，但是还是学一下pytorch怎么封装的吧 

"""

def check_metrics(metrics):
    metric_dict = {'accuracy': Accuracy}
    for i, metric in enumerate(metrics):
        if type(metric) is str:
            assert metric in metric_dict.keys(
            ), f"`{metric}` is an unsupported metric"
            metrics[i] = metric_dict[metric]()
        else:
            assert isinstance(metric, Metric)
    return metrics


class Metric:
    def __call__(self, *args, **kwds):
        raise NotImplementedError('Please define "a call method"')



class Accuracy(Metric):
    def __init__(self):
        self.name = 'accuracy'

    def __call__(self, y_pred, y_true, method='mean'):
        return accuracy(y_pred, y_true, method=method)



def accuracy(y_pred, y_true, method='mean'):
    """计算准确率或预测正确的数量。"""
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred.type(y_true.dtype) == y_true
    if method == 'sum':
        return float(cmp.type(y_true.dtype).sum())
    elif method == 'mean':
        return float(cmp.type(y_true.dtype).sum()) / y_true.numel()
    else:
        raise NotImplementedError


# def evaluate_accuracy(net, data_iter, device=None): 
#     net.eval()  # 设置为评估模式
#     if not device:
#         device = next(iter(net.parameters())).device
#     # 正确预测的数量，总预测的数量
#     metric = Accumulator(2)
#     for X, y_true in data_iter:
#         X = X.to(device)
#         y_true = y_true.to(device)
#         metric.add(accuracy_sum(net(X), y_true), y_true.numel())
#     return metric[0] / metric[1]
