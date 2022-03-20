
r"""
metric必须实现method='sum', 'mean'方法，最好返回float pytorch好像没有像tensorflow那样把metric都封装好，分类的话可以借助sklearn.metric

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
    def __init__(self, *args, **kwds):
        raise NotImplementedError('Please define "a init method"')

    def __call__(self, *args, **kwds):
        raise NotImplementedError('Please define "a call method"')



class Accuracy(Metric):
    def __init__(self, name='Accuracy'):
        self.name = name

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

