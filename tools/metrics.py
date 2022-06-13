
import numpy as np
__all__ = ['SegmentationMetric']

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


"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py

confusionMetric
P\L     P    N

P      TP    FP

N      FN    TN

"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, mIoU)

